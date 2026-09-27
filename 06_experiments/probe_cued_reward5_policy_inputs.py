#!/usr/bin/env python3
"""Measure one-step policy sensitivity to CA3 and depth on real rollouts.

The policy and environment are loaded through the standard place-field loader.
Recorded core outputs are replayed through the frozen action head after swapping
one input group across observations. The task cue and selected DG goal stay fixed.
This diagnoses action use, not closed-loop reward performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.model.model_utils import get_rnn_size
from sf_working_directories.IntrMotiv.evaluation.place_fields import load_policy_env


def action_probabilities(actor_critic, core_outputs: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
    chunks = []
    with torch.no_grad():
        for batch in core_outputs.split(batch_size):
            output = actor_critic.forward_tail(batch, values_only=False, sample_actions=False)
            chunks.append(F.softmax(output["action_logits"], dim=-1).cpu())
    return torch.cat(chunks)


def sensitivity(reference: torch.Tensor, altered: torch.Tensor) -> dict[str, float]:
    return {
        "mean_total_variation": float((reference - altered).abs().sum(-1).mul(0.5).mean()),
        "argmax_change_fraction": float((reference.argmax(-1) != altered.argmax(-1)).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--decisions", type=int, default=1024)
    parser.add_argument("--shuffles", type=int, default=5)
    args = parser.parse_args()
    if args.decisions < 2 or args.shuffles < 1:
        parser.error("decisions must be >= 2 and shuffles must be >= 1")

    cfg, env, _env_info, actor_critic, checkpoint, device = load_policy_env(
        args.run_dir, args.decisions, True, 0, args.checkpoint
    )
    env_info = extract_env_info(env, cfg)
    core = actor_critic.core
    canonical_ca3_end = int(core.core_output_size)
    depth_end = canonical_ca3_end + int(actor_critic.encoder.depth_encoder.get_out_size())
    if not bool(actor_critic.encoder.depth_sensor) or depth_end > int(core.target_condition_start):
        raise ValueError("Expected a depth bypass before the goal condition")
    if getattr(core, "dg_goal_modulation", None) is not None:
        # Goal-write cores append the trace seen by the worker. Their canonical
        # trace is retained for the manager but is not read by the action head.
        ca3_start = int(core.total_output_size)
        ca3_end = ca3_start + canonical_ca3_end
        ca3_path = "goal_written_worker_trace"
    else:
        ca3_start, ca3_end = 0, canonical_ca3_end
        ca3_path = "canonical_trace"

    observed_core: list[torch.Tensor] = []
    observed_logits: list[torch.Tensor] = []

    def capture_core(_module, _inputs, output):
        state = output[0] if isinstance(output, (tuple, list)) else output
        observed_core.append(state.detach().cpu())

    hook = core.register_forward_hook(capture_core)
    try:
        obs, _ = env.reset()
        rnn_states = torch.zeros((env.num_agents, get_rnn_size(cfg)), dtype=torch.float32, device=device)
        with torch.no_grad():
            for _ in range(args.decisions):
                normalized = prepare_and_normalize_obs(actor_critic, obs)
                output = actor_critic(normalized, rnn_states)
                observed_logits.append(output["action_logits"].detach().cpu())
                actions = argmax_actions(actor_critic.action_distribution())
                if actions.ndim == 1:
                    actions = unsqueeze_tensor(actions, dim=-1)
                obs, _, terminated, truncated, _ = env.step(preprocess_actions(env_info, actions))
                rnn_states = output["new_rnn_states"].clone()
                rnn_states[make_dones(terminated, truncated).bool()] = 0
    finally:
        hook.remove()
        env.close()

    states = torch.cat(observed_core, dim=0)
    if len(states) < 2:
        raise ValueError("Rollout produced too few core states")
    reference = action_probabilities(actor_critic, states)
    observed_reference = F.softmax(torch.cat(observed_logits, dim=0), dim=-1)
    replay_max_error = float((reference - observed_reference).abs().max())
    if replay_max_error > 1e-5:
        raise ValueError(f"Frozen action-head replay disagrees with rollout: {replay_max_error}")
    generator = torch.Generator().manual_seed(int(cfg.seed) + 1971)
    results = {"ca3": [], "depth": []}
    for _ in range(args.shuffles):
        permutation = torch.randperm(len(states), generator=generator)
        for name, start, end in (
            ("ca3", ca3_start, ca3_end),
            ("depth", canonical_ca3_end, depth_end),
        ):
            counterfactual = states.clone()
            counterfactual[:, start:end] = states[permutation, start:end]
            results[name].append(sensitivity(reference, action_probabilities(actor_critic, counterfactual)))
    summary = {
        "run_dir": str(args.run_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "decisions": len(states),
        "shuffles": args.shuffles,
        "ca3_width": ca3_end - ca3_start,
        "ca3_path": ca3_path,
        "depth_width": depth_end - canonical_ca3_end,
        "action_replay_max_probability_error": replay_max_error,
        "method": "real-rollout core states, within-run marginal shuffles, frozen one-step action head",
        "sensitivity": {
            name: {
                metric: sum(item[metric] for item in entries) / len(entries)
                for metric in entries[0]
            }
            for name, entries in results.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
