"""Replay a validated observation panel through frozen IntrMotiv layers.

Each invocation consumes one canonical place-field manifest row on a compute
node. The archived DG evaluator output is an exact-alignment check; downstream
summaries use its occupancy and the same 19-by-19 spatial grid. Raw traces stay
in memory, so only compact, source-identified summaries are exported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from hpc_runs.behavior_field_metrics import comparison_details, information_scores
from hpc_runs.behavior_field_probe import (
    capture_decoder_module, configure_workspace_paths, frozen_digest,
)


LAYERS = ("ca3", "decoder_1", "decoder_2")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def summarize_layer(pose: np.ndarray, activity: np.ndarray) -> tuple[dict, np.ndarray]:
    """Report both gain-invariant and amplitude-weighted spatial information.

    Signed downstream traces are rectified as in the frozen behavior probe.
    Bits per activation averages only field-eligible units; the amplitude score
    averages all units, including zero-valued units, matching the DG evaluator.
    """
    positive = np.maximum(np.asarray(activity, dtype=np.float32), 0)
    details = comparison_details(pose, positive)
    occupancy = details["occupancy"]
    eligible = details["field_eligible"]
    maps = details["rate_maps"]
    normalized, amplitude = [], []
    for unit in range(positive.shape[1]):
        bits, weighted = information_scores(maps[:, :, unit], occupancy)
        normalized.append(bits)
        amplitude.append(0.0 if not np.isfinite(weighted) else weighted)
    normalized = np.asarray(normalized)
    amplitude = np.asarray(amplitude)
    selected = normalized[eligible & np.isfinite(normalized)]
    selected_amplitude = amplitude[eligible & np.isfinite(normalized)]
    result = {
        "units": int(positive.shape[1]),
        "eligible_units": int(eligible.sum()),
        "silent_units": int(np.count_nonzero(~(positive > 0).any(axis=0))),
        "mean_positive_fraction": float(np.mean(positive > 0)),
        "mean_rectified_activation": float(positive.mean()),
        "mean_bits_per_activation_eligible": float(selected.mean()) if selected.size else None,
        "mean_bits_per_step_eligible": float(selected_amplitude.mean()) if selected_amplitude.size else None,
        "mean_amplitude_weighted_bits_all": float(amplitude.mean()),
        "median_bits_per_activation_eligible": float(np.median(selected)) if selected.size else None,
        "visited_bins": int(np.count_nonzero(occupancy)),
        "observations_in_bounds": int(occupancy.sum()),
    }
    return result, occupancy


def load_row(manifest: Path, index: int) -> dict[str, str]:
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not 0 <= index < len(rows):
        raise IndexError(index)
    row = rows[index]
    if not row["condition"].startswith(("COS27_C01_", "COS27_C05_", "COS27_C15_")) or int(row["target_frames"]) != 100_000_000:
        raise ValueError("Expected a fresh COSYNE terminal row")
    return row


def replay(row: dict[str, str], panel: Path, dg_reference: Path,
           output_root: Path, workspace_root: Path) -> dict:
    from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
    from sample_factory.model.model_utils import get_rnn_size
    from sf_working_directories.IntrMotiv.evaluation import place_fields

    root = workspace_root.resolve(strict=True)
    run_dir, checkpoint = Path(row["run_dir"]), Path(row["checkpoint"])
    for path in (run_dir, checkpoint, panel, dg_reference):
        if not path.resolve(strict=True).is_relative_to(root):
            raise ValueError(f"Input outside active workspace: {path}")
    with np.load(panel, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    keys = [key for key in data if key.startswith("obs_")]
    if "obs_pos" not in keys or "obs_rot" not in keys or "dones" not in data:
        raise ValueError("Panel lacks pose or boundaries")
    count = len(data["dones"])
    if count != 10_001 or any(len(data[key]) != count for key in keys):
        raise ValueError("Unexpected panel length or misaligned observations")
    if data["dones"].shape[1] != 1:
        raise ValueError("Expected a one-agent panel")

    configured_paths: dict[str, str] = {}
    original_factory = place_fields.make_env_func_batched

    def workspace_factory(cfg, *args, **kwargs):
        configured_paths.update(configure_workspace_paths(cfg, run_dir, output_root, root))
        return original_factory(cfg, *args, **kwargs)

    torch.set_num_threads(1)
    place_fields.make_env_func_batched = workspace_factory
    try:
        cfg, env, _env_info, actor, selected, device = place_fields.load_policy_env(
            run_dir, count, False, 0, checkpoint
        )
    finally:
        place_fields.make_env_func_batched = original_factory
    if set(configured_paths) != {"train_dir", "dmlab_level_cache_path", "wandb_dir"}:
        raise RuntimeError("Workspace overrides did not run")
    actor.eval()
    before = frozen_digest(actor)
    decoder_name, decoder_module = capture_decoder_module(actor)
    captured: list[np.ndarray] = []
    captured_core: list[np.ndarray] = []

    def core_hook(_module, _inputs, result):
        value = result[0] if isinstance(result, (tuple, list)) else result
        captured_core.append(value.detach().cpu().numpy().reshape(-1).copy())

    def decoder_hook(_module, _inputs, result):
        value = result[0] if isinstance(result, (tuple, list)) else result
        captured.append(value.detach().cpu().numpy().reshape(-1).copy())

    hook = decoder_module.register_forward_hook(decoder_hook)
    core_handle = actor.core.register_forward_hook(core_hook)
    state = torch.zeros((1, get_rnn_size(cfg)), dtype=torch.float32, device=device)
    width = int(cfg.Hippo_n_feature)
    expanded = int(cfg.Hippo_R + cfg.Hippo_L - 1)
    traces: dict[str, list[np.ndarray]] = {key: [] for key in ("dg",) + LAYERS}
    try:
        with torch.no_grad():
            for t in range(count):
                obs = {key[4:]: torch.as_tensor(data[key][t], device=device) for key in keys}
                captured.clear()
                captured_core.clear()
                output = actor(prepare_and_normalize_obs(actor, obs), state)
                if len(captured) != 2 or len(captured_core) != 1:
                    raise RuntimeError("Unexpected core or decoder hook calls")
                ca3 = captured_core[0][:width * expanded].copy()
                if ca3.size != width * expanded:
                    raise RuntimeError("CA3 width mismatch")
                traces["ca3"].append(ca3)
                traces["dg"].append(ca3[::expanded].copy())
                traces["decoder_1"].append(captured[0])
                traces["decoder_2"].append(captured[1])
                state = output["new_rnn_states"]
                state[torch.as_tensor(data["dones"][t], device=device).bool()] = 0
    finally:
        hook.remove()
        core_handle.remove()
        env.close()
    if frozen_digest(actor) != before:
        raise RuntimeError("Frozen model parameters or BatchNorm statistics changed")

    pose = np.stack((data["obs_pos"][:, 0, 0], data["obs_pos"][:, 0, 1],
                     data["obs_rot"][:, 0, 1]), axis=1).astype(np.float32)
    summaries = {}
    dg, occupancy = summarize_layer(pose, np.asarray(traces["dg"]))
    dg_details = comparison_details(pose, np.asarray(traces["dg"]))
    dg_per_unit = np.nan_to_num(np.asarray([
        information_scores(dg_details["rate_maps"][:, :, unit], occupancy)[1]
        for unit in range(width)
    ]))
    with np.load(dg_reference, allow_pickle=False) as archive:
        reference_occupancy = archive["occupancy"]
        reference_per_unit = archive["spatial_information"]
        reference_score = float(np.mean(reference_per_unit))
        reference_checkpoint = str(archive["checkpoint"].item())
    # The historical DG evaluator stores grid axes as (x, y); the shared
    # spatial_contract and behavior probe use (y, x). The counts are identical
    # after transpose, and spatial information is invariant to this reordering.
    same_occupancy = np.array_equal(occupancy.T, reference_occupancy)
    same_unit_scores = np.allclose(dg_per_unit, reference_per_unit, atol=1e-5, rtol=1e-5)
    if not same_occupancy or abs(
        dg["mean_amplitude_weighted_bits_all"] - reference_score
    ) > 1e-4 or not same_unit_scores or reference_checkpoint != str(selected):
        raise RuntimeError(
            "Replay does not reproduce archived DG field result: "
            f"occupancy_equal={same_occupancy}, visited={dg['visited_bins']}, "
            f"unit_scores_equal={same_unit_scores}, "
            f"replay_score={dg['mean_amplitude_weighted_bits_all']:.9f}, "
            f"reference_score={reference_score:.9f}, "
            f"checkpoint_equal={reference_checkpoint == str(selected)}"
        )
    for layer in LAYERS:
        summaries[layer], layer_occupancy = summarize_layer(pose, np.asarray(traces[layer]))
        if not np.array_equal(layer_occupancy, occupancy):
            raise RuntimeError(f"Occupancy differs for {layer}")
    return {
        "schema": "intrmotiv/common-panel-layers/v1",
        "condition": row["condition"], "seed": int(row["seed"]),
        "checkpoint_frames": int(row["checkpoint_frames"]),
        "checkpoint": str(selected), "checkpoint_sha256": digest(selected),
        "panel": str(panel), "panel_sha256": digest(panel),
        "dg_reference": str(dg_reference), "dg_reference_sha256": digest(dg_reference),
        "dg_reference_mean_amplitude_weighted_bits": reference_score,
        "dg_replay_mean_amplitude_weighted_bits": dg["mean_amplitude_weighted_bits_all"],
        "dg_max_unit_score_absolute_error": float(np.max(np.abs(dg_per_unit - reference_per_unit))),
        "archived_dg_occupancy_orientation": "x_y; transpose matches replay y_x",
        "frozen_model_sha256": before, "decoder_hook": decoder_name,
        "runtime_paths": configured_paths, "layers": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--dg-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    args = parser.parse_args()
    row = load_row(args.manifest, args.row)
    label = f'{row["condition"]}__s{row["seed"]}__f{row["checkpoint_frames"]}'
    reference = args.dg_root / "raw" / f'{Path(row["run_dir"]).parent.name.rstrip("_")}__{Path(row["run_dir"]).name}__{row["label_suffix"]}' / "place_fields.npz"
    output = args.output_root / "raw" / f"{label}.json"
    if output.exists() or not output.resolve().is_relative_to(args.workspace_root.resolve(strict=True)):
        raise ValueError(f"Existing or non-workspace output: {output}")
    result = replay(row, args.panel, reference, args.output_root, args.workspace_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified {label}: {output}", flush=True)


if __name__ == "__main__":
    main()
