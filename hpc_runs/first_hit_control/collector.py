"""Frozen reward-free replay using the canonical IntrMotiv evaluator loader.

Execute on a NEMO2 compute node. Real observations advance CA3 exactly once;
the next iteration finalizes a pending nonterminal transition. Certified
terminal observations advance a cloned pre-reset core state, never the acting
state and never an autoreset observation. This module does not train or deploy
a controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from hpc_runs.first_hit_control.eligibility import certify_slice_only_relabel


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_hash(actor):
    digest = hashlib.sha256()
    for name, value in sorted(actor.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class PendingReplay:
    """Single-stream factual pairing with explicit missing-terminal semantics."""

    def __init__(self, count, width, goals):
        self.count, self.width, self.goals = count, width, goals
        self.index, self.pending = 0, None
        self.data = {
            "core": np.zeros((count, width), np.float32),
            "action": np.zeros(count, np.int64),
            "successor_core": np.zeros((count, width), np.float32),
            "successor_dg": np.zeros((count, goals), np.float32),
            "physical_done": np.zeros(count, bool),
            "successor_valid": np.zeros(count, bool),
            "successor_core_valid": np.zeros(count, bool),
            "episode_id": np.zeros(count, np.int64),
            "time_in_episode": np.zeros(count, np.int64),
            "source_goal": np.full(count, -1, np.int64),
            "random_action": np.zeros(count, bool),
        }

    def begin(self, core, action, episode, step, source_goal, randomized=False):
        if self.pending is not None or self.index >= self.count:
            raise RuntimeError("Previous transition must be finalized before a new action")
        core = np.asarray(core, dtype=np.float32)
        if core.shape != (self.width,) or not np.isfinite(core).all():
            raise ValueError("Invalid factual core output")
        i = self.index
        self.data["core"][i] = core
        for key, value in (("action", action), ("episode_id", episode), ("time_in_episode", step),
                           ("source_goal", source_goal), ("random_action", randomized)):
            self.data[key][i] = value
        self.pending = (episode, step)

    def finish(self, *, physical_done, successor_core=None, successor_dg=None, successor_episode=None):
        if self.pending is None:
            raise RuntimeError("No factual action is waiting for its successor")
        if not physical_done and successor_episode != self.pending[0]:
            raise ValueError("A reset observation cannot be a live successor")
        i = self.index
        self.data["physical_done"][i] = physical_done
        if successor_dg is not None:
            dg = np.asarray(successor_dg, dtype=np.float32)
            if dg.shape != (self.goals,) or not np.isfinite(dg).all() or (dg < 0).any():
                raise ValueError("Invalid successor DG")
            self.data["successor_dg"][i] = dg
            self.data["successor_valid"][i] = True
        if successor_core is not None:
            core = np.asarray(successor_core, dtype=np.float32)
            if core.shape != (self.width,) or not np.isfinite(core).all() or successor_dg is None:
                raise ValueError("Successor core requires its genuine DG observation")
            self.data["successor_core"][i] = core
            self.data["successor_core_valid"][i] = True
        if not physical_done and successor_core is None:
            raise ValueError("Live transitions require a complete factual successor")
        self.index += 1
        self.pending = None


def install_final_capture(env):
    """Capture genuine final observations and pre-shaping rewards without mutation."""
    import gymnasium as gym
    from sample_factory.algo.utils.make_env import BatchedMultiAgentWrapper
    from sf_working_directories.IntrMotiv.dmlab.wrappers.reward_shaping import DmlabRewardShapingWrapper

    class RawRewardCapture(gym.Wrapper):
        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            raw = np.asarray(reward)
            if raw.size != 1:
                raise RuntimeError("Qualification requires a single native reward scalar")
            return obs, reward, terminated, truncated, dict(info, first_hit_raw_reward=float(raw.item()))

    class FinalCapture(gym.Wrapper):
        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                valid = bool(info.get("intrmotiv_final_observation_valid", False))
                info = dict(info, controller_final_observation_valid=valid)
                if valid:
                    info["controller_final_observation"] = {k: np.array(v, copy=True) for k, v in obs.items()}
            return obs, reward, terminated, truncated, info

    node = env
    while not isinstance(node, BatchedMultiAgentWrapper):
        if not hasattr(node, "env"):
            raise RuntimeError("Cannot certify placement before Sample Factory autoreset")
        node = node.env
    node.env = FinalCapture(node.env)
    node = node.env
    while not isinstance(node, DmlabRewardShapingWrapper):
        if not hasattr(node, "env"):
            raise RuntimeError("Cannot certify reward capture before DMLab reward shaping")
        node = node.env
    node.env = RawRewardCapture(node.env)
    if hasattr(env.unwrapped, "capture_terminal_observation"):
        env.unwrapped.capture_terminal_observation = True


class RewardAudit:
    """Per-decision native reward checks; signed cancellation cannot pass."""

    def __init__(self):
        self.raw_sum = self.raw_abs_sum = self.wrapped_sum = 0.0
        self.raw_nonzero_count = self.steps = 0

    def record(self, info, wrapped_reward):
        if "first_hit_raw_reward" not in info:
            raise RuntimeError("Missing native reward captured before shaping")
        raw, wrapped = float(info["first_hit_raw_reward"]), float(wrapped_reward)
        if not np.isfinite(raw) or not np.isfinite(wrapped):
            raise RuntimeError("Nonfinite native or wrapped reward")
        self.steps += 1
        self.raw_sum += raw
        self.raw_abs_sum += abs(raw)
        self.raw_nonzero_count += int(raw != 0.0)
        self.wrapped_sum += wrapped


def encode_terminal_core(actor, normalized_observation, pre_reset_state):
    """Use actual core detector processing on a clone; never advance acting state."""
    head = actor.forward_head(normalized_observation)
    core, _ = actor.forward_core(head, pre_reset_state.clone())
    return core


def collect(args):
    # No native environment is imported or run by module-level tests.
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Native collection must run inside an allocated Slurm compute job")
    sys.path.insert(0, str(args.source_root.resolve(strict=True)))
    import deepmind_lab
    from sample_factory.algo.sampling.batched_sampling import preprocess_actions
    from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
    from sample_factory.model.model_utils import get_rnn_size
    from sf_working_directories.IntrMotiv.evaluation.place_fields import load_policy_env

    workspace = Path(os.environ.get("INTRMOTIV_WORKSPACE_ROOT", "/work/classic/fr_xl1014-corridor-geometry")).resolve()
    output = args.output.resolve()
    if not output.is_relative_to(workspace) or output.exists():
        raise ValueError("Use a fresh output directory in the active allocated workspace")
    output.mkdir(parents=True)
    source_config = args.run_dir / "config.json"
    if file_hash(source_config) != args.config_sha256:
        raise ValueError("Source configuration differs from the reviewed donor")
    original = json.loads(source_config.read_text())
    if original.get("env") != "openfield_map2_fixed_loc3_fixedlength_noreward":
        raise ValueError("Qualification requires the selected reward-free parent level")
    if file_hash(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("Checkpoint identity differs from the reviewed donor")
    loader = output / "loader"
    loader.mkdir()
    overrides = dict(dmlab_use_level_cache=False, dmlab_level_cache_path=str(output / "cache"),
                     dmlab_runfiles_path=str(args.runfiles.resolve(strict=True)), online_spatial_telemetry=False,
                     with_wandb=False, transfer_freeze_dg=True)
    (loader / "config.json").write_text(json.dumps(dict(original, **overrides), indent=2) + "\n")
    deepmind_lab.set_runfiles_path(str(args.runfiles.resolve(strict=True)))
    cfg, env, env_info, actor, checkpoint, device = load_policy_env(loader, args.decisions, False, 0, args.checkpoint)
    if env.num_agents != 1:
        env.close()
        raise ValueError("Only one factual environment stream is supported")
    actor.eval().requires_grad_(False)
    try:
        certificate = certify_slice_only_relabel(actor.core, actor.decoder, source_root=args.source_root,
                                                input_kind="canonical_core_output")
        install_final_capture(env)
    except BaseException:
        env.close()
        raise
    width, goals = certificate["layout"]["width"], certificate["layout"]["goals"]
    start = certificate["layout"]["goal_start"]
    expanded = int(actor.core.expanded_length)
    replay = PendingReplay(args.decisions, width, goals)
    frozen_before = model_hash(actor)
    checkpoint_stat = checkpoint.stat()
    random = np.random.default_rng(args.seed + 17011)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    episode, step, observations_encoded, terminal_encoded = 0, 0, 0, 0
    rewards = RewardAudit()
    privileged = {"pos", "rot", "GEOMETRY.ENTITY_LAYER", "GEOMETRY.CUE_MANIFEST", "GEOMETRY.GEOMETRY"}

    def normalized(obs, *, terminal=False):
        # Debug pose is evaluation telemetry, never a decoder or encoder input.
        selected = {k: v for k, v in obs.items() if k not in privileged and not k.startswith("GEOMETRY.")}
        if terminal:
            selected = {k: torch.as_tensor(v).unsqueeze(0) for k, v in selected.items()}
        return prepare_and_normalize_obs(actor, selected)

    def encode(obs, state):
        nonlocal observations_encoded
        head = actor.forward_head(normalized(obs))
        core, state = actor.forward_core(head, state)
        observations_encoded += 1
        return core, state

    try:
        env.unwrapped.seed(args.seed)
        obs, _ = env.reset()
        state = torch.zeros(1, get_rnn_size(cfg), dtype=torch.float32, device=device)
        with torch.no_grad():
            for index in range(args.decisions):
                core, state = encode(obs, state)
                current = core[0].cpu().numpy().copy()
                current_dg = current[: goals * expanded : expanded].copy()
                if replay.pending is not None:
                    replay.finish(physical_done=False, successor_core=current, successor_dg=current_dg, successor_episode=episode)
                result = actor.forward_tail(core, values_only=False, sample_actions=True)
                action = result["actions"].reshape(1, 1)
                randomized = random.random() < args.random_action_probability
                if randomized:
                    action = torch.full_like(action, int(random.integers(actor.action_space.n)))
                command = current[start : start + goals]
                source_goal = int(command.argmax()) if command.sum() > 0 else -1
                replay.begin(current, int(action.item()), episode, step, source_goal, randomized)
                obs, reward, terminated, truncated, infos = env.step(preprocess_actions(env_info, action))
                info = infos[0]
                rewards.record(info, float(torch.as_tensor(reward).sum()))
                done = bool(torch.as_tensor(make_dones(terminated, truncated)).reshape(-1)[0])
                if done:
                    terminal_dg = terminal_core = None
                    if bool(info.get("controller_final_observation_valid", False)):
                        if "controller_final_observation" not in info:
                            raise RuntimeError("Certified terminal flag lacks its genuine observation")
                        final_core = encode_terminal_core(
                            actor, normalized(info["controller_final_observation"], terminal=True), state
                        )
                        terminal_core = final_core[0].cpu().numpy().copy()
                        terminal_dg = terminal_core[: goals * expanded : expanded].copy()
                        terminal_encoded += 1
                    replay.finish(physical_done=True, successor_core=terminal_core, successor_dg=terminal_dg)
                    state.zero_()
                    episode, step = episode + 1, 0
                else:
                    step += 1
                if (index + 1) % 1000 == 0:
                    print(f"Collected {index + 1}/{args.decisions} factual decisions", flush=True)
            if replay.pending is not None:
                # The final returned live observation is genuine; encode it
                # once to retain the last transition without taking an action.
                core, _ = encode(obs, state)
                successor = core[0].cpu().numpy().copy()
                replay.finish(physical_done=False, successor_core=successor,
                              successor_dg=successor[: goals * expanded : expanded], successor_episode=episode)
    finally:
        env.close()
    frozen_after = model_hash(actor)
    if (frozen_after != frozen_before or rewards.raw_nonzero_count or rewards.steps != args.decisions
            or replay.index != args.decisions):
        raise RuntimeError("Frozen-model, reward-free, or complete-transition qualification failed")
    if (checkpoint.stat().st_mtime_ns != checkpoint_stat.st_mtime_ns or file_hash(checkpoint) != args.checkpoint_sha256
            or file_hash(source_config) != args.config_sha256):
        raise RuntimeError("Donor checkpoint changed during collection")
    dataset = output / "replay.npz"
    np.savez_compressed(dataset, **replay.data)
    decoder_path = output / "decoder_state.pt"
    torch.save({
        "decoder_state_dict": {k: v.detach().cpu().clone() for k, v in actor.decoder.state_dict().items()},
        "action_head_state_dict": {k: v.detach().cpu().clone() for k, v in actor.action_parameterization.state_dict().items()},
        "decoder_class": type(actor.decoder).__name__, "action_head_class": type(actor.action_parameterization).__name__,
        "decoder_config": {key: getattr(cfg, key) for key in ("decoder_mlp_layers", "nonlinearity", "use_jit")},
        "layout": certificate["layout"], "action_count": actor.action_space.n,
    }, decoder_path)
    metadata = dict(
        schema="intrmotiv/frozen-first-hit-replay/v1", source_run=str(args.run_dir), checkpoint=str(checkpoint),
        checkpoint_sha256=args.checkpoint_sha256, source_config_sha256=file_hash(source_config),
        evaluation_overrides=overrides, original_use_jit=original.get("use_jit"), evaluation_use_jit=cfg.use_jit,
        eligibility=certificate, model_sha256_before=frozen_before, model_sha256_after=frozen_after,
        model_unchanged=True, decisions=args.decisions, completed_episodes=episode,
        observation_core_encodes=observations_encoded, terminal_cloned_core_encodes=terminal_encoded,
        missing_terminal_successors=int((replay.data["physical_done"] & ~replay.data["successor_valid"]).sum()),
        reset_observations_used_as_successors=0, native_reward_sum=rewards.raw_sum,
        native_reward_abs_sum=rewards.raw_abs_sum, native_reward_nonzero_count=rewards.raw_nonzero_count,
        native_reward_zero_every_step=rewards.raw_nonzero_count == 0 and rewards.steps == args.decisions,
        native_reward_decisions=rewards.steps, wrapped_reward_sum=rewards.wrapped_sum,
        native_reward_capture="Immediately below DmlabRewardShapingWrapper, one raw scalar per decision",
        seed=args.seed, random_action_probability=args.random_action_probability,
        random_actions=int(replay.data["random_action"].sum()), action_count=actor.action_space.n,
        action_counts=np.bincount(replay.data["action"], minlength=actor.action_space.n).tolist(),
        primary_detector="winner", secondary_detector="exclusive", goal_sampling="none; label every goal after collection",
        env=cfg.env, frameskip=cfg.env_frameskip, decoder_class=type(actor.decoder).__name__,
        layout=certificate["layout"], dataset_sha256=file_hash(dataset), decoder_state_sha256=file_hash(decoder_path),
        collector_sha256=file_hash(__file__), scope="Frozen factual reward-free replay; no goal-control claim",
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(output), "decisions": args.decisions, "missing_terminal_successors": metadata["missing_terminal_successors"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "checkpoint", "source-root", "runfiles", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--decisions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=73001)
    parser.add_argument("--random-action-probability", type=float, default=0.2)
    args = parser.parse_args()
    if args.decisions < 1 or not 0 <= args.random_action_probability <= 1:
        parser.error("Require positive decisions and a mixture probability in [0,1]")
    collect(args)


if __name__ == "__main__":
    main()
