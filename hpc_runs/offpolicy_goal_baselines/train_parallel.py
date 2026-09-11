"""Parallel DMLab collection with batched frozen encoding and replay learning."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import gymnasium as gym
import numpy as np
import torch
from tensorboardX import SummaryWriter

from hpc_runs.offpolicy_goal_baselines.model import ContrastiveGoalAgent
from hpc_runs.offpolicy_goal_baselines.planner import LandmarkPlanner
from hpc_runs.offpolicy_goal_baselines.replay import EpisodeReplay
from hpc_runs.offpolicy_goal_baselines.train import (
    FrozenVisualFeatures,
    _baseline_parser,
    _save_checkpoint,
    _tensor,
    _workspace_output,
    _write_json,
)


@dataclass(frozen=True)
class VectorConfig:
    num_envs: int
    context: str


class DmlabEnvFactory:
    """Pickle-safe factory consumed by Gymnasium's existing worker pool."""

    def __init__(self, cfg, index: int) -> None:
        self.cfg = cfg
        self.index = int(index)

    def __call__(self):
        from sf_working_directories.IntrMotiv.dmlab.dmlab_env import make_dmlab_env

        env_config = {
            "env_id": self.index,
            "vector_index": self.index,
            "worker_index": self.index,
        }
        return make_dmlab_env(
            self.cfg.env,
            self.cfg,
            env_config,
            dmlab_level_caches_per_policy=None,
        )


def parse_args(argv=None):
    vector_parser = argparse.ArgumentParser(add_help=False)
    vector_parser.add_argument("--baseline-num-envs", type=int, default=16)
    vector_parser.add_argument(
        "--baseline-vector-context", choices=("spawn", "forkserver"), default="forkserver"
    )
    vector_ns, remaining = vector_parser.parse_known_args(argv)
    baseline_ns, remaining = _baseline_parser().parse_known_args(remaining)
    from sf_working_directories.IntrMotiv.dmlab.train_hipposlam import parse_dmlab_args

    cfg = parse_dmlab_args(remaining)
    from hpc_runs.offpolicy_goal_baselines.train import BaselineConfig

    baseline = BaselineConfig(
        **{key.removeprefix("baseline_"): value for key, value in vars(baseline_ns).items()}
    )
    vector = VectorConfig(
        num_envs=int(vector_ns.baseline_num_envs), context=str(vector_ns.baseline_vector_context)
    )
    if vector.num_envs <= 1:
        raise ValueError("parallel trainer requires --baseline-num-envs > 1")
    if cfg.encoder_conv_architecture != "layer2_resnet18" or bool(cfg.with_pos_obs):
        raise ValueError("parallel baselines require frozen layer2_resnet18 and no pose input")
    if not bool(cfg.online_spatial_telemetry) or not bool(cfg.exploration_coverage_telemetry):
        raise ValueError("coverage and pose telemetry are required for evaluation")
    return baseline, vector, cfg


def _poses(observations: dict, num_envs: int) -> np.ndarray:
    pose = np.asarray(observations.get("telemetry_pose"), dtype=np.float32)
    if pose.shape != (num_envs, 3) or not np.isfinite(pose).all():
        raise RuntimeError(f"valid batched telemetry_pose is required, got {pose.shape}")
    return pose


def _select_info(infos: dict, index: int, num_envs: int) -> dict:
    """Convert Gymnasium's dict-of-arrays vector info into one environment info."""
    if bool(np.asarray(infos.get("_final_info", np.zeros(num_envs, dtype=bool)))[index]):
        final = np.asarray(infos["final_info"], dtype=object)[index]
        return {} if final is None else dict(final)
    selected = {}
    for key, value in infos.items():
        if key.startswith("_") or key in ("final_info", "final_observation"):
            continue
        mask = infos.get(f"_{key}")
        if mask is not None and not bool(np.asarray(mask)[index]):
            continue
        if isinstance(value, dict):
            selected[key] = _select_nested(value, index, num_envs)
        else:
            array = np.asarray(value)
            selected[key] = array[index] if array.ndim > 0 and len(array) == num_envs else value
    return selected


def _select_nested(values: dict, index: int, num_envs: int) -> dict:
    selected = {}
    for key, value in values.items():
        if key.startswith("_"):
            continue
        mask = values.get(f"_{key}")
        if mask is not None and not bool(np.asarray(mask)[index]):
            continue
        if isinstance(value, dict):
            selected[key] = _select_nested(value, index, num_envs)
        else:
            array = np.asarray(value)
            selected[key] = array[index] if array.ndim > 0 and len(array) == num_envs else value
    return selected


def _optimize(
    baseline,
    replay,
    agent,
    device,
    critic_parameters,
    critic_optimizer,
    actor_optimizer,
    alpha_optimizer,
    log_alpha,
    target_entropy,
):
    batch = replay.sample(baseline.batch_size, baseline.max_future, baseline.discount)
    agent.train()
    critic_optimizer.zero_grad(set_to_none=True)
    actor_optimizer.zero_grad(set_to_none=True)
    critic_loss, actor_loss, metrics = agent.losses(
        _tensor(batch, "state", device),
        _tensor(batch, "action", device),
        _tensor(batch, "future_goal", device),
        _tensor(batch, "offset", device),
        _tensor(batch, "random_goal", device),
        entropy_coeff=float(log_alpha.exp().detach()),
        logsumexp_coeff=baseline.logsumexp_coeff,
        max_future=baseline.max_future,
    )
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic_parameters, 10.0)
    critic_optimizer.step()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 10.0)
    actor_optimizer.step()
    alpha_optimizer.zero_grad(set_to_none=True)
    observed_entropy = torch.as_tensor(metrics["policy_entropy"], device=device)
    alpha_loss = log_alpha.exp() * (observed_entropy - target_entropy)
    alpha_loss.backward()
    alpha_optimizer.step()
    metrics["entropy_alpha"] = float(log_alpha.exp().detach())
    metrics["alpha_loss"] = float(alpha_loss.detach())
    return metrics


def train(argv=None) -> int:
    baseline, vector, cfg = parse_args(argv)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(baseline.torch_threads)

    # Start workers before constructing a CUDA context. AsyncVectorEnv supplies
    # shared-memory observations and its established reset/exception handling.
    env = gym.vector.AsyncVectorEnv(
        [DmlabEnvFactory(cfg, index) for index in range(vector.num_envs)],
        shared_memory=True,
        context=vector.context,
    )
    observations, _ = env.reset(seed=[int(cfg.seed) + i for i in range(vector.num_envs)])
    device = torch.device("cpu" if str(cfg.device).lower() == "cpu" else "cuda")
    extractor = FrozenVisualFeatures(cfg, env.single_observation_space, device)
    current_features = extractor.batch(observations)
    current_poses = _poses(observations, vector.num_envs)
    feature_dim = current_features.shape[1]
    num_actions = int(env.single_action_space.n)

    agent = ContrastiveGoalAgent(
        feature_dim, num_actions, hidden_dim=baseline.hidden_dim, repr_dim=baseline.repr_dim
    ).to(device)
    critic_parameters = [
        parameter for name, parameter in agent.named_parameters() if not name.startswith("actor.")
    ]
    critic_optimizer = torch.optim.Adam(critic_parameters, lr=baseline.learning_rate)
    actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=baseline.learning_rate)
    log_alpha = torch.tensor(
        math.log(baseline.entropy_coeff), device=device, dtype=torch.float32, requires_grad=True
    )
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=baseline.learning_rate)
    target_entropy = baseline.target_entropy_fraction * math.log(num_actions)
    replay = EpisodeReplay(baseline.replay_capacity, seed=cfg.seed)
    planner = LandmarkPlanner(
        baseline.landmark_count,
        baseline.landmark_candidates,
        baseline.landmark_neighbors,
        baseline.landmark_local_horizon,
    )

    output = _workspace_output(cfg)
    _write_json(
        output / "run_config.json",
        {
            "schema": "intrmotiv/offpolicy-goal-baseline/v1",
            "baseline": asdict(baseline),
            "parallel": asdict(vector),
            "seed": int(cfg.seed),
            "environment": cfg.env,
            "action_count": num_actions,
            "action_repeat": int(cfg.env_frameskip),
            "visual_encoder": "layer2_resnet18_imagenet_frozen_batched",
            "policy_pose_input": False,
            "feature_dim": int(feature_dim),
        },
    )
    writer = SummaryWriter(str(output / ".summary" / "0"))
    metrics_file = (output / "metrics.jsonl").open("a", buffering=1)

    episode_features = [[current_features[i].copy()] for i in range(vector.num_envs)]
    episode_poses = [[current_poses[i].copy()] for i in range(vector.num_envs)]
    episode_actions = [[] for _ in range(vector.num_envs)]
    final_goals = [None] * vector.num_envs
    goal_poses = [None] * vector.num_envs
    active_goals = [None] * vector.num_envs
    option_steps = np.zeros(vector.num_envs, dtype=np.int64)
    planner_steps = np.zeros(vector.num_envs, dtype=np.int64)
    frames = decisions = updates = update_buckets = 0
    option_attempts = option_successes = 0
    next_checkpoint = baseline.checkpoint_frames
    next_planner_rebuild = baseline.planner_rebuild_frames
    start_time = last_log = time.monotonic()
    train_metrics = {}

    try:
        while frames < baseline.total_frames:
            ready = replay.size >= baseline.replay_min
            if not ready:
                actions = np.asarray(
                    [env.single_action_space.sample() for _ in range(vector.num_envs)], dtype=np.int64
                )
            else:
                for i in range(vector.num_envs):
                    if final_goals[i] is None or option_steps[i] >= baseline.goal_horizon:
                        final_goals[i], goal_poses[i] = replay.sample_goal()
                        active_goals[i] = final_goals[i]
                        option_steps[i] = planner_steps[i] = 0
                        option_attempts += 1
                    if baseline.method == "l3p" and planner_steps[i] >= baseline.planner_horizon:
                        active_goals[i] = planner.subgoal(
                            current_features[i], final_goals[i], agent, device
                        )
                        planner_steps[i] = 0
                agent.eval()
                with torch.no_grad():
                    actions = agent.policy(
                        torch.as_tensor(current_features, device=device),
                        torch.as_tensor(np.stack(active_goals), device=device),
                    ).cpu().numpy()

            next_observations, _, terminated, truncated, infos = env.step(actions)
            dones = np.asarray(terminated) | np.asarray(truncated)
            next_features = extractor.batch(next_observations)
            next_poses = _poses(next_observations, vector.num_envs)
            per_info = [_select_info(infos, i, vector.num_envs) for i in range(vector.num_envs)]
            step_frames = [int(info.get("num_frames", cfg.env_frameskip)) for info in per_info]
            frames += sum(step_frames)
            decisions += vector.num_envs
            option_steps += 1
            planner_steps += 1

            for i in range(vector.num_envs):
                episode_actions[i].append(int(actions[i]))
                if dones[i]:
                    # The vector worker has already reset. Close replay with an
                    # absorbing copy of the last valid observation, then start
                    # the next episode from the returned reset observation.
                    episode_features[i].append(current_features[i].copy())
                    episode_poses[i].append(current_poses[i].copy())
                    replay.add(
                        np.asarray(episode_features[i]),
                        np.asarray(episode_actions[i]),
                        np.asarray(episode_poses[i]),
                    )
                    for key, value in per_info[i].get("episode_extra_stats", {}).items():
                        writer.add_scalar(key, float(value), frames)
                    episode_features[i] = [next_features[i].copy()]
                    episode_poses[i] = [next_poses[i].copy()]
                    episode_actions[i] = []
                    final_goals[i] = goal_poses[i] = active_goals[i] = None
                    option_steps[i] = planner_steps[i] = 0
                else:
                    episode_features[i].append(next_features[i].copy())
                    episode_poses[i].append(next_poses[i].copy())
                    if ready and goal_poses[i] is not None:
                        distance = float(np.linalg.norm(next_poses[i, :2] - goal_poses[i][:2]))
                        if distance <= float(cfg.exploration_coverage_grid_size):
                            option_successes += 1
                            final_goals[i] = None
                for key, value in per_info[i].get("periodic_stats", {}).items():
                    writer.add_scalar(key, float(value), frames)
            current_features, current_poses = next_features, next_poses

            desired_buckets = decisions // baseline.update_every_steps
            if replay.size >= baseline.replay_min:
                new_buckets = desired_buckets - update_buckets
                for _ in range(new_buckets * baseline.updates_per_step):
                    train_metrics = _optimize(
                        baseline,
                        replay,
                        agent,
                        device,
                        critic_parameters,
                        critic_optimizer,
                        actor_optimizer,
                        alpha_optimizer,
                        log_alpha,
                        target_entropy,
                    )
                    updates += 1
            update_buckets = desired_buckets

            while (
                replay.size >= baseline.replay_min
                and baseline.method == "l3p"
                and frames >= next_planner_rebuild
            ):
                planner.rebuild(replay, agent, device)
                next_planner_rebuild += baseline.planner_rebuild_frames

            now = time.monotonic()
            if now - last_log >= 30.0:
                record = {
                    "frames": frames,
                    "decisions": decisions,
                    "updates": updates,
                    "replay_size": replay.size,
                    "collector_envs": vector.num_envs,
                    "throughput_fps": frames / max(now - start_time, 1e-6),
                    "option_attempts": option_attempts,
                    "option_success_rate": option_successes / max(option_attempts, 1),
                    **train_metrics,
                }
                if baseline.method == "l3p":
                    record.update(
                        planner_rebuilds=planner.rebuild_count,
                        planner_finite_edges=planner.finite_edges,
                        planner_queries=planner.subgoal_queries,
                        planner_landmark_fraction=(
                            planner.landmark_subgoals / max(planner.subgoal_queries, 1)
                        ),
                    )
                metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                for key, value in record.items():
                    if key != "frames":
                        writer.add_scalar(f"offpolicy/{key}", float(value), frames)
                writer.add_scalar("train/env_steps", frames, frames)
                writer.flush()
                last_log = now

            while frames >= next_checkpoint:
                _save_checkpoint(
                    output / f"checkpoint_{next_checkpoint:012d}.pt",
                    frames,
                    agent,
                    critic_optimizer,
                    actor_optimizer,
                    alpha_optimizer,
                    log_alpha,
                    baseline,
                    cfg,
                )
                next_checkpoint += baseline.checkpoint_frames
    finally:
        _save_checkpoint(
            output / f"checkpoint_{frames:012d}_final.pt",
            frames,
            agent,
            critic_optimizer,
            actor_optimizer,
            alpha_optimizer,
            log_alpha,
            baseline,
            cfg,
        )
        metrics_file.close()
        writer.close()
        env.close(terminate=True)
    return 0


def main() -> None:
    sys.exit(train())


if __name__ == "__main__":
    main()
