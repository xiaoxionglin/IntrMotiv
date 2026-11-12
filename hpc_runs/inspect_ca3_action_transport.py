"""Post-hoc kinematic coordinates for saved linear-readout CA3 transports.

No fitting, labels, or model changes are involved. Recover a coordinate map
from the east-anchor position readout and the learned forward matrix, then
check all action matrices against independently specified planar kinematics.
Canonical matrices are analysis references and never training inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ca3_action_transport import TransportReadout
from parallel_action_register_probe import ROOT, digest


def planar_action_maps() -> dict[str, np.ndarray]:
    """Column state (p_x,p_y,h_x,h_y), unit forward and quarter turns."""
    identity, zero = np.eye(2), np.zeros((2, 2))
    left = np.array([[0., -1.], [1., 0.]])
    return {
        "F": np.block([[identity, identity], [zero, identity]]),
        "L": np.block([[identity, zero], [zero, left]]),
        "R": np.block([[identity, zero], [zero, left.T]]),
    }


def recover_coordinates(matrices: dict[str, np.ndarray], readout: np.ndarray) -> dict:
    """C reads position; C(T_F-I) reads the effect of a forward command.

    The resulting heading interpretation is tested by its turn dynamics and
    its invariance under forward motion. The forward matrix's upper blocks
    are fixed by this construction, so those blocks alone are not evidence.
    A position offset can be subtracted after the linear coordinate change.
    """
    forward = np.asarray(matrices["F"], dtype=np.float64)
    readout = np.asarray(readout, dtype=np.float64)
    coordinates = np.concatenate((readout, readout @ (forward - np.eye(4))), axis=0)
    if np.linalg.matrix_rank(coordinates) != 4:
        raise ValueError("Readout and forward effect do not define four independent coordinates")
    inverse = np.linalg.inv(coordinates)
    transformed = {name: coordinates @ matrix @ inverse for name, matrix in matrices.items()}
    canonical = planar_action_maps()
    return {
        "coordinate_map": coordinates.tolist(),
        "coordinate_map_condition": float(np.linalg.cond(coordinates)),
        "coordinate_matrices": {name: value.tolist() for name, value in transformed.items()},
        "canonical_frobenius_errors": {name: float(np.linalg.norm(value - canonical[name]))
                                       for name, value in transformed.items()},
        "delta_singular_values": {name: np.linalg.svd(matrix - np.eye(4), compute_uv=False).tolist()
                                  for name, matrix in matrices.items()},
    }


def trace_word(word: str, matrices: dict[str, np.ndarray], initial: np.ndarray, coordinates: np.ndarray) -> dict:
    """Constructed word illustration; not a held-out performance statistic."""
    tag, initial_coordinates = initial.copy(), coordinates @ initial
    trajectory = [initial_coordinates]
    for action in word:
        tag = matrices[action] @ tag
        trajectory.append(coordinates @ tag)
    trajectory = np.asarray(trajectory)
    trajectory[:, :2] -= initial_coordinates[:2]
    return {"word": word, "position": trajectory[:, :2].tolist(), "heading": trajectory[:, 2:].tolist()}


def plot_paths(result: dict, output: Path) -> dict:
    """Show two action orders using a declared representative fit, seed 99."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = Path(font_manager.findfont("DejaVu Sans", fallback_to_default=False))
    if not font.is_file() or font.suffix.lower() not in (".ttf", ".otf"):
        raise RuntimeError("A scalable TTF/OTF font is required")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    exemplar = next(item for item in result["checkpoints"] if item["seed"] == 99)
    reference = planar_action_maps()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8), sharex=True, sharey=True)
    for axis, word, title in zip(axes, ("FLF", "LFF"),
                                  ("Forward → left → forward", "Left → forward → forward")):
        learned = exemplar["examples"][word]
        position = np.asarray(learned["position"])
        heading = np.asarray(learned["heading"])
        expected = trace_word(word, reference, np.array([0., 0., 1., 0.]), np.eye(4))
        canonical = np.asarray(expected["position"])
        axis.plot(canonical[:, 0], canonical[:, 1], color=".65", lw=3, label="Analytic motion")
        axis.scatter(position[:, 0], position[:, 1], color="#0072B2", s=60, zorder=4,
                     label="Learned tag (seed 99)")
        # Initial and final headings are arrows, distinct from the path.
        for index, color in ((0, "#666666"), (-1, "#0072B2")):
            start = position[index]
            axis.arrow(*start, *(heading[index] * .35), width=.018,
                       head_width=.12, color=color, length_includes_head=True, zorder=5)
        turn_step = word.index("L") + 1
        turn_position = position[turn_step]
        axis.scatter(*turn_position, facecolors="none", edgecolors="#D55E00", s=200, lw=2, zorder=6)
        axis.annotate("Turn in place", xy=turn_position,
                      xytext=(1.05, -.48) if word == "FLF" else (.55, .45),
                      arrowprops={"arrowstyle": "->", "color": "#D55E00"}, color="#A54100", fontsize=12)
        endpoint = canonical[-1].astype(int)
        axis.text(position[-1, 0] + .12, position[-1, 1] + .12,
                  f"End ({endpoint[0]}, {endpoint[1]})", fontsize=13)
        axis.set_title(title, fontsize=14, pad=14)
        axis.set_aspect("equal")
        axis.set_xlim(-.45, 2.5)
        axis.set_ylim(-.6, 2.5)
        axis.set_xticks((0, 1, 2))
        axis.set_yticks((0, 1, 2))
        axis.grid(alpha=.2)
        axis.set_xlabel("Displacement x (grid steps)")
    axes[0].set_ylabel("Displacement y (grid steps)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=13,
               bbox_to_anchor=(.5, .99))
    fig.subplots_adjust(top=.79, bottom=.20, left=.08, right=.98, wspace=.18)
    fig.text(.08, .045, "Same start, same action counts. Turning changes heading; the next forward command uses that heading.", fontsize=12)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / f"kinematics_paths.{extension}", dpi=140 if extension == "png" else 180)
    plt.close(fig)
    svg = output / "kinematics_paths.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    return {"font": str(font), "representative_seed": 99,
            "scope": "Two constructed words illustrate recovered action maps; not a held-out accuracy figure."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    output = args.directory.resolve()
    spec = json.loads((output / "spec.json").read_text())
    provenance = json.loads((output / "provenance.json").read_text())
    if digest(spec) != provenance["spec_sha256"]:
        raise ValueError("Comparison specification changed since fitting")
    inputs = {**provenance["sources"], **provenance["parent_artifacts"]}
    for name, expected in inputs.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source or parent artifact changed since fitting: {name}")
    parent_config = json.loads((ROOT / spec["parent_directory"] / "config.json").read_text())
    replay = json.loads((output / "replay_audit.json").read_text())
    variant = next(item for item in spec["variants"] if item["mode"] == "per_action" and item["rank"] == 2 and item["hidden"] == 0)
    base_size = 5 * (parent_config["steps"] + parent_config["repeat"])
    tag_dim, scale = spec["tag_dim"], parent_config["target_scale"]
    records = []
    for seed in parent_config["fit_seeds"]:
        path = output / "weights" / f"{variant['id']}__seed{seed}.pt"
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != replay["weight_sha256"][path.name]:
            raise ValueError(f"Checkpoint changed since replay audit: {path.name}")
        inputs[str(path.relative_to(ROOT))] = sha
        model = TransportReadout(base_size, 5, tag_dim, 2, parent_config["repeat"],
                                 parent_config["expanded"], "per_action", 0)
        model.load_state_dict(torch.load(path, weights_only=True))
        matrices = dict(zip("FLR", model.core.transport.matrices().detach().numpy()[:3].astype(np.float64)))
        weight = model.head.weight.detach().numpy().astype(np.float64) * scale
        c0 = model.core.initial_tag.numpy().astype(np.float64)
        readout = weight[:, base_size : base_size + tag_dim]
        record = {"seed": seed, "checkpoint_sha256": sha, **recover_coordinates(matrices, readout)}
        coordinates = np.asarray(record["coordinate_map"])
        record["initial_tag_in_coordinates"] = (coordinates @ c0).tolist()
        record["nuisance_tag_readout_norm"] = float(np.linalg.norm(weight[:, base_size + 4 * tag_dim:]))
        rotation = np.array([[0., -1.], [1., 0.]])
        record["other_landmark_readout_rotation_errors"] = [
            float(np.linalg.norm(weight[:, base_size + i * tag_dim:base_size + (i + 1) * tag_dim]
                                 - np.linalg.matrix_power(rotation, i) @ readout)) for i in (1, 2, 3)]
        record["examples"] = {word: trace_word(word, matrices, c0, coordinates) for word in ("FLF", "LFF")}
        records.append(record)
    result = {"schema": "intrmotiv/kinematic-weight-inspection/v1", "checkpoints": records,
              "coordinate_construction": "Q = stack(C, C(T_F-I)); C is the scaled east-anchor tag readout. No fit or target labels.",
              "validation": "Forward lower blocks and all left/right blocks are independent checks; forward upper blocks follow by construction.",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "inputs": inputs}
    figure = plot_paths(result, output)
    (output / "weight_inspection.json").write_text(json.dumps(result, indent=2) + "\n")
    figure["inspection_source_sha256"] = result["source_sha256"]
    figure["inspection_sha256"] = hashlib.sha256((output / "weight_inspection.json").read_bytes()).hexdigest()
    (output / "kinematics_figure_manifest.json").write_text(json.dumps(figure, indent=2) + "\n")
    print(json.dumps([{ "seed": item["seed"], "basis_condition": item["coordinate_map_condition"],
                        "matrix_errors": item["canonical_frobenius_errors"],
                        "landmark_rotation_errors": item["other_landmark_readout_rotation_errors"]} for item in records], indent=2))


if __name__ == "__main__":
    main()
