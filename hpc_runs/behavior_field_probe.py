"""Frozen IntrMotiv rollout with aligned multilayer traces and action controls.

Run only inside an ordinary NEMO2 compute job. Pose is read for telemetry and
never passed to the actor. The model is evaluated once per decision in all arms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


POLICIES = ("own", "uniform", "persistent8")


def choose_executed_action(policy: str, proposed: int, rng: np.random.Generator,
                           n_actions: int, decision: int, previous: int | None) -> int:
    if policy == "own":
        return proposed
    if policy == "uniform":
        return int(rng.integers(n_actions))
    if policy == "persistent8":
        return int(rng.integers(n_actions)) if previous is None or decision % 8 == 0 else previous
    raise ValueError(f"Unknown policy {policy}")


def frozen_digest(actor: nn.Module) -> str:
    """Hash learned parameters and BN running statistics, excluding live graph state."""
    digest = hashlib.sha256()
    for name, value in actor.named_parameters():
        digest.update(name.encode() + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    for name, module in actor.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            for field in ("running_mean", "running_var", "num_batches_tracked"):
                value = getattr(module, field, None)
                if value is not None:
                    digest.update(f"{name}.{field}".encode() + b"\0")
                    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reset_recurrent_state(state: torch.Tensor, done: bool) -> torch.Tensor:
    if done:
        state.zero_()
    return state


def assert_alignment(records: dict[str, list], decisions: int) -> None:
    lengths = {name: len(values) for name, values in records.items()}
    if set(lengths.values()) != {decisions}:
        raise RuntimeError(f"Trace alignment or exact decision count failed: {lengths}")


def configure_workspace_paths(cfg: Any, run_dir: Path, output_root: Path,
                              workspace_root: Path) -> dict[str, str]:
    """Override retired paths from historical configs before environment creation."""
    root = workspace_root.resolve(strict=True)
    paths = {
        "train_dir": run_dir.parent.resolve(),
        "dmlab_level_cache_path": (output_root / "dmlab_cache").resolve(),
        "wandb_dir": (output_root / "wandb").resolve(),
    }
    for name, path in paths.items():
        if not path.is_relative_to(root):
            raise ValueError(f"{name} outside active workspace: {path}")
        setattr(cfg, name, str(path))
    cfg.with_wandb = False
    return {key: str(value) for key, value in paths.items()}


def capture_decoder_module(actor: nn.Module) -> tuple[str, nn.Module]:
    """Use the poster's first activation hook, which fires at both MLP layers."""
    for name, module in actor.decoder.named_modules():
        if isinstance(module, (nn.ReLU, nn.ELU, nn.GELU, nn.Tanh, nn.LeakyReLU, nn.SiLU)):
            return f"decoder.{name}", module
    raise RuntimeError("Decoder has no supported hidden-layer activation")


def collect(run_dir: Path, checkpoint: Path, output: Path, policy: str,
            decisions: int, eval_seed: int, workspace_root: Path) -> dict[str, object]:
    from sample_factory.algo.sampling.batched_sampling import preprocess_actions
    from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
    from sample_factory.model.model_utils import get_rnn_size
    from sf_working_directories.IntrMotiv.evaluation import place_fields

    if policy not in POLICIES or decisions < 1 or output.exists():
        raise ValueError("Invalid policy, decision count, or existing output")
    root = workspace_root.resolve(strict=True)
    for path in (run_dir, checkpoint, output):
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Probe input/output outside active workspace: {path}")
    output_root = output.parent.parent
    cache_path = output_root / "dmlab_cache"
    wandb_path = output_root / "wandb"
    for path in (cache_path, wandb_path):
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Runtime output outside active workspace: {path}")

    # The historical config restores retired training/cache/W&B paths. Reuse
    # the canonical loader, but replace those paths immediately before its
    # environment constructor sees the loaded config.
    original_factory = place_fields.make_env_func_batched
    configured_paths: dict[str, str] = {}

    def workspace_factory(cfg, *args, **kwargs):
        configured_paths.update(configure_workspace_paths(cfg, run_dir, output_root, root))
        return original_factory(cfg, *args, **kwargs)

    torch.set_num_threads(1)
    place_fields.make_env_func_batched = workspace_factory
    try:
        cfg, env, env_info, actor, selected, device = place_fields.load_policy_env(
            run_dir, decisions, False, 0, checkpoint
        )
    finally:
        place_fields.make_env_func_batched = original_factory
    if set(configured_paths) != {"train_dir", "dmlab_level_cache_path", "wandb_dir"}:
        raise RuntimeError("Workspace path override did not run")
    if env.num_agents != 1 or actor.action_space.n < 2:
        raise RuntimeError("Probe requires one agent with discrete actions")
    if not hasattr(env.unwrapped, "seed"):
        raise RuntimeError("Environment does not expose reproducible reset seeds")
    env.unwrapped.seed(eval_seed)
    torch.manual_seed(eval_seed + 10_000)
    action_rng = np.random.default_rng(eval_seed + 20_000)
    actor.eval()
    before = frozen_digest(actor)
    decoder_name, decoder_module = capture_decoder_module(actor)
    captured: dict[str, list[np.ndarray]] = {"core": [], "decoder": []}

    def core_hook(_module, _inputs, result):
        output = result[0] if isinstance(result, (tuple, list)) else result
        captured["core"].append(output.detach().cpu().numpy().reshape(-1).copy())

    def decoder_hook(_module, _inputs, result):
        output = result[0] if isinstance(result, (tuple, list)) else result
        captured["decoder"].append(output.detach().cpu().numpy().reshape(-1).copy())

    core_handle = actor.core.register_forward_hook(core_hook)
    decoder_handle = decoder_module.register_forward_hook(decoder_hook)
    records: dict[str, list] = {key: [] for key in (
        "pose", "dg", "dg_pre_threshold", "ca3", "decoder_1", "decoder_2",
        "value", "action_logits", "action_probabilities", "proposed_action",
        "executed_action", "reward", "done", "episode_id",
    )}
    previous_random: int | None = None
    episode = 0
    state = torch.zeros((1, get_rnn_size(cfg)), dtype=torch.float32, device=device)
    try:
        obs, _ = env.reset()
        with torch.no_grad():
            for t in range(decisions):
                captured["core"].clear()
                captured["decoder"].clear()
                pose = np.asarray([
                    float(obs["pos"][0, 0]), float(obs["pos"][0, 1]),
                    float(obs["rot"][0, 1]),
                ], dtype=np.float32)
                outputs = actor(prepare_and_normalize_obs(actor, obs), state)
                if len(captured["core"]) != 1 or len(captured["decoder"]) != 2:
                    raise RuntimeError(f"Unexpected hook calls: core={len(captured['core'])}, "
                                       f"decoder={len(captured['decoder'])}")
                core = captured["core"][0]
                width = int(cfg.Hippo_n_feature)
                expanded = int(cfg.Hippo_R + cfg.Hippo_L - 1)
                ca3 = core[:width * expanded]
                if ca3.size != width * expanded:
                    raise RuntimeError("CA3 trace shorter than declared")
                pre = actor.encoder.DG_projection.last_pre_threshold_logits
                if pre is None:
                    raise RuntimeError("DG pre-threshold trace unavailable")
                pre = pre.detach().cpu().numpy().reshape(-1).copy()
                if pre.size != width:
                    raise RuntimeError("DG logit width mismatch")
                logits = outputs["action_logits"].detach().cpu().numpy().reshape(-1)
                if logits.size != actor.action_space.n:
                    raise RuntimeError("Action logit width mismatch")
                probabilities = torch.softmax(outputs["action_logits"], dim=-1)
                proposed = int(outputs["actions"].reshape(-1)[0])
                executed = choose_executed_action(
                    policy, proposed, action_rng, actor.action_space.n, t, previous_random
                )
                previous_random = executed
                records["pose"].append(pose)
                records["dg"].append(core[:width * expanded:expanded].copy())
                records["dg_pre_threshold"].append(pre)
                records["ca3"].append(ca3.copy())
                records["decoder_1"].append(captured["decoder"][0])
                records["decoder_2"].append(captured["decoder"][1])
                records["value"].append(float(outputs["values"].reshape(-1)[0]))
                records["action_logits"].append(logits.copy())
                records["action_probabilities"].append(probabilities.detach().cpu().numpy().reshape(-1).copy())
                records["proposed_action"].append(proposed)
                records["executed_action"].append(executed)
                records["episode_id"].append(episode)
                action = torch.tensor([[executed]], dtype=torch.int64, device=device)
                obs, reward, terminated, truncated, _ = env.step(preprocess_actions(env_info, action))
                done = bool(make_dones(terminated, truncated).reshape(-1)[0])
                records["reward"].append(float(reward.reshape(-1)[0]))
                records["done"].append(done)
                state = outputs["new_rnn_states"]
                if done:
                    state = reset_recurrent_state(state, done)
                    previous_random = None
                    episode += 1
    finally:
        core_handle.remove()
        decoder_handle.remove()
        env.close()
    after = frozen_digest(actor)
    if before != after:
        raise RuntimeError("Model parameters or BatchNorm running statistics changed")
    assert_alignment(records, decisions)
    arrays = {key: np.asarray(value) for key, value in records.items()}
    output.mkdir(parents=True)
    np.savez_compressed(output / "trace.npz", **arrays)
    metadata: dict[str, object] = {
        "schema": "intrmotiv/behavior-field-probe/v1",
        "run_dir": str(run_dir), "checkpoint": str(selected),
        "checkpoint_sha256": checkpoint_digest(selected),
        "frozen_model_sha256": before,
        "policy": policy, "decisions": decisions, "eval_seed": eval_seed,
        "environment": cfg.env, "action_repeat": int(cfg.env_frameskip),
        "action_count": int(actor.action_space.n), "episodes_completed": episode,
        "decoder_hook": decoder_name,
        "runtime_paths": configured_paths,
        "layer_widths": {key: int(arrays[key].shape[1]) for key in
                         ("dg", "ca3", "decoder_1", "decoder_2")},
        "proposed_executed_disagreements": int(np.count_nonzero(
            arrays["proposed_action"] != arrays["executed_action"])),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--decisions", type=int, default=50_000)
    parser.add_argument("--eval-seed", type=int, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.run_dir, args.checkpoint, args.output,
                             args.policy, args.decisions, args.eval_seed,
                             args.workspace_root)), flush=True)


if __name__ == "__main__":
    main()
