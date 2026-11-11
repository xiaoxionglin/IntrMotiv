"""Offline, synthetic test of landmark/action binding in unchanged CA3 memory.

This is a readout experiment, not Sample Factory training or DMLab telemetry.
Reuse the fingerprinted prediction kit's pure state adapters without editing
that export. All word occurrences (headings included) share a split, and model
selection uses validation words only. Run with --help for the reproducible CLI.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "exports/intrmotiv_prediction_starter"
sys.path.insert(0, str(KIT))
from intrmotiv_transfer.state import advance_action_history, advance_ca3  # noqa: E402

ACTIONS = "FLR"
DIRECTIONS = np.array(((1, 0), (0, 1), (-1, 0), (0, -1)), dtype=np.int64)
MODELS = (
    "landmark_only_mlp64", "action_only_mlp64", "unordered_mlp64",
    "parallel_linear", "parallel_mlp64", "parallel_mlp128",
)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def endpoint(word: str, heading: int) -> np.ndarray:
    """Exact displacement since an anchor cue; turns and moves do not commute."""
    position = np.zeros(2, dtype=np.int64)
    for action in word:
        if action == "W":
            continue
        if action == "F":
            position += DIRECTIONS[heading]
        else:
            heading = (heading + (1 if action == "L" else -1)) % 4
    return position


def words_by_split(config: dict) -> dict[str, list[str]]:
    """Hold out complete words, not rows or initial headings of the same word."""
    splits = {name: [] for name in ("train", "validation", "unseen_words", "longer_words")}
    for length in range(config["train_min_length"], config["train_max_length"] + 1):
        words = ["".join(chars) for chars in itertools.product(ACTIONS, repeat=length)]
        # Sorting hashes gives deterministic per-length proportions, independent
        # of Python hash randomization, target labels, and optimizer seeds.
        words.sort(key=lambda word: digest((config["split_seed"], word)))
        count = max(1, round(len(words) * config["holdout_fraction"]))
        splits["validation"].extend(words[:count])
        splits["unseen_words"].extend(words[count : 2 * count])
        splits["train"].extend(words[2 * count :])
    rng = np.random.default_rng(config["split_seed"])
    for length in config["longer_lengths"]:
        words = ["".join(chars) for chars in itertools.product(ACTIONS, repeat=length)]
        chosen = rng.choice(len(words), min(config["longer_words_per_length"], len(words)), replace=False)
        splits["longer_words"].extend(words[index] for index in chosen)
    return splits


def build_dataset(config: dict) -> dict:
    """Construct actual registers before the endpoint label is ever revealed.

    A nuisance DG cue at t=0 precedes the task anchor. The anchor's four
    identities specify initial heading E/N/W/S. Its age determines which
    actions belong to this local displacement query. Earlier actions are
    nuisance history. No pose or endpoint enters the input.
    """
    rows = []
    for split, words in words_by_split(config).items():
        for word in words:
            counts = tuple(word.count(action) for action in ACTIONS)
            # Identical nuisance actions within each anagram family. Matched
            # counterfactuals differ ONLY in order after the same anchor.
            family = (len(word), counts)
            rng = np.random.default_rng(int(digest((config["split_seed"], family))[:8], 16))
            age = config.get("anchor_age") or len(word)
            if age < len(word) or age >= config["steps"]:
                raise ValueError("Anchor age must cover the word and leave a nuisance prefix")
            prefix = "".join(rng.choice(list(ACTIONS), config["steps"] - age))
            # Optional age-matched control: interleave zero-encoded no-ops.
            # Every trained action position is covered while the anchor age is
            # identical for short training and longer held-out action words.
            suffix = ["W"] * age
            positions = sorted(rng.choice(age, len(word), replace=False))
            for position, action in zip(positions, word):
                suffix[position] = action
            for heading in range(4):
                rows.append({"split": split, "word": word, "heading": heading,
                             "prefix": prefix, "suffix": "".join(suffix), "counts": counts})
    count, steps = len(rows), config["steps"]
    state = torch.zeros(count, config["dg_count"], config["expanded"])
    actions = torch.zeros(count, config["expanded"], len(ACTIONS))
    dg = torch.zeros(count, config["dg_count"])
    dg[:, -1] = 1  # nuisance landmark, shared across all examples
    state = advance_ca3(state, dg, config["repeat"])
    headings = torch.tensor([row["heading"] for row in rows])
    anchor_times = torch.tensor([steps - len(row["suffix"]) for row in rows])
    for step in range(steps):
        # The action produces observation step+1. An anchor at time tau is
        # observed after prefix action tau-1, before suffix action tau.
        action = torch.tensor([ACTIONS.index(char) if char in ACTIONS else len(ACTIONS)
                               for row in rows for char in [(row["prefix"] + row["suffix"])[step]]])
        actions = advance_action_history(actions, action)
        dg = torch.zeros_like(dg)
        selected = torch.where(anchor_times == step + 1)[0]
        dg[selected, headings[selected]] = 1
        state = advance_ca3(state, dg, config["repeat"])
    # Remove only analytically constant-zero coordinates. These supports are
    # specified by the generator, not learned from validation/test labels.
    dg_features = state[:, [0, 1, 2, 3, config["dg_count"] - 1], : steps + config["repeat"]].flatten(1)
    action_features = actions[:, -steps:].flatten(1)
    bag_features = torch.tensor([row["counts"] for row in rows], dtype=torch.float32)
    features = {
        "landmark": dg_features,
        "action": action_features,
        "unordered": torch.cat((dg_features, bag_features), dim=1),
        "parallel": torch.cat((dg_features, action_features), dim=1),
    }
    targets = torch.tensor(np.stack([endpoint(row["word"], row["heading"]) for row in rows]), dtype=torch.float32)
    indices = {name: np.array([i for i, row in enumerate(rows) if row["split"] == name])
               for name in words_by_split(config)}
    return {"rows": rows, "features": features, "targets": targets,
            "indices": indices, "dg_state": state, "action_state": actions}


def decode_registers(data: dict, config: dict) -> np.ndarray:
    """Known-dynamics oracle using registers only, not stored words or headings.

    Slot differences recover the isolated cue's age despite R-slot injection.
    This is an information-sufficiency check, NOT a learned-model competitor.
    """
    state = data["dg_state"].numpy()
    action_state = data["action_state"].numpy()
    output = []
    for landmark, action in zip(state, action_state):
        heading = int(np.argmax(landmark[:4].sum(axis=1)))
        age = int(np.flatnonzero(landmark[heading])[0])
        word = "".join(ACTIONS[int(slot.argmax())] if slot.any() else "W" for slot in action[-age:]) if age else ""
        output.append(endpoint(word, heading))
    return np.stack(output)


def matched_pairs(data: dict, split: str, limit: int = 10000) -> np.ndarray:
    """Same cue history, suffix length/counts, prefix; different true endpoint."""
    groups = defaultdict(list)
    for index in data["indices"][split]:
        row = data["rows"][index]
        groups[(row["heading"], len(row["word"]), row["counts"])].append(index)
    pairs = [(i, j) for group in groups.values() for i, j in itertools.combinations(group, 2)
             if not torch.equal(data["targets"][i], data["targets"][j])]
    result = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    if len(result) > limit:
        rng = np.random.default_rng(0)
        result = result[rng.choice(len(result), limit, replace=False)]
    return result


def metrics(prediction: np.ndarray, data: dict, indices: np.ndarray, pairs: np.ndarray) -> dict:
    target = data["targets"].numpy()
    error = prediction[indices] - target[indices]
    result = {
        "n_examples": len(indices),
        "position_rmse": float(np.sqrt(np.mean(np.sum(error ** 2, axis=1)))),
        "rounded_endpoint_accuracy": float(np.all(np.round(prediction[indices]) == target[indices], axis=1).mean()),
        "n_matched_pairs": len(pairs),
    }
    if len(pairs):
        i, j = pairs.T
        true_difference = target[j] - target[i]
        predicted_difference = prediction[j] - prediction[i]
        dot = (true_difference * predicted_difference).sum(axis=1)
        # Zero response is scored 1/2, not a correct distinction. Direction
        # accuracy tests ordering only; endpoint accuracy tests calibration.
        result.update(
            pair_direction_accuracy=float(np.where(abs(dot) < 1e-6, .5, dot > 0).mean()),
            pair_difference_rmse=float(np.sqrt(np.mean(np.sum((predicted_difference - true_difference) ** 2, axis=1)))),
        )
    return result


def make_model(name: str, size: int) -> nn.Module:
    if name == "parallel_linear":
        return nn.Linear(size, 2)
    width = int(name.rsplit("mlp", 1)[1])
    return nn.Sequential(nn.Linear(size, width), nn.ReLU(), nn.Linear(width, 2))


def fit(name: str, seed: int, data: dict, config: dict) -> tuple[np.ndarray, dict]:
    torch.manual_seed(seed)
    kind = name.split("_", 1)[0]
    x, y = data["features"][kind], data["targets"] / config["target_scale"]
    train = torch.tensor(data["indices"]["train"])
    valid = torch.tensor(data["indices"]["validation"])
    model = make_model(name, x.size(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    best_loss, best_state, best_epoch = float("inf"), None, None
    trace = []
    started = time.monotonic()
    for epoch in range(config["epochs"] + 1):
        if epoch % config["validation_every"] == 0 or epoch == config["epochs"]:
            with torch.no_grad():
                validation_loss = (model(x[valid]) - y[valid]).square().mean().item()
                train_loss = (model(x[train]) - y[train]).square().mean().item()
            trace.append({"epoch": epoch, "train_mse_scaled": train_loss,
                          "validation_mse_scaled": validation_loss})
            if validation_loss < best_loss:
                best_loss, best_epoch = validation_loss, epoch
                best_state = copy.deepcopy(model.state_dict())
        if epoch == config["epochs"]:
            break
        order = train[torch.randperm(len(train))]
        for batch in order.split(config["batch_size"]):
            loss = (model(x[batch]) - y[batch]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.load_state_dict(best_state)
    with torch.no_grad():
        prediction = (model(x) * config["target_scale"]).numpy()
    return prediction, {"model": name, "seed": seed, "parameters": sum(p.numel() for p in model.parameters()),
                        "input_size": x.size(1), "best_epoch": best_epoch,
                        "seconds": time.monotonic() - started, "trace": trace}


def audit(data: dict, config: dict) -> dict:
    word_sets = {split: {data["rows"][i]["word"] for i in indices}
                 for split, indices in data["indices"].items()}
    for left, right in itertools.combinations(word_sets, 2):
        assert word_sets[left].isdisjoint(word_sets[right]), (left, right)
    assert np.array_equal(decode_registers(data, config), data["targets"].numpy())
    pair_counts = {}
    for split in word_sets:
        pairs = matched_pairs(data, split)
        if len(pairs):
            i, j = pairs.T
            assert torch.equal(data["dg_state"][i], data["dg_state"][j])
            assert torch.equal(data["features"]["unordered"][i], data["features"]["unordered"][j])
            assert not torch.any(torch.all(data["features"]["action"][i] == data["features"]["action"][j], dim=1))
        pair_counts[split] = len(pairs)
    return {"disjoint_word_splits": True, "oracle_exact": True,
            "matched_histories_and_counts_exact": True, "matched_pair_counts": pair_counts,
            "word_counts": {split: len(words) for split, words in word_sets.items()},
            "dataset_sha256": digest(data["rows"])}


def run(config: dict, output: Path) -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=True)
    data = build_dataset(config)
    checks = audit(data, config)
    print(json.dumps(checks), flush=True)
    predictions, records, scores = {}, [], []
    pairs = {split: matched_pairs(data, split) for split in data["indices"]}
    for name in config["models"]:
        for seed in config["fit_seeds"]:
            prediction, record = fit(name, seed, data, config)
            key = f"{name}__seed{seed}"
            predictions[key] = prediction
            records.append(record)
            for split, indices in data["indices"].items():
                scores.append({"model": name, "seed": seed, "split": split,
                               **metrics(prediction, data, indices, pairs[split])})
            print(json.dumps({k: v for k, v in record.items() if k != "trace"}), flush=True)
    zero = np.zeros_like(data["targets"].numpy())
    oracle = decode_registers(data, config)
    for name, prediction in (("zero", zero), ("known_dynamics_oracle", oracle)):
        for split, indices in data["indices"].items():
            scores.append({"model": name, "seed": -1, "split": split,
                           **metrics(prediction, data, indices, pairs[split])})
    np.savez_compressed(output / "predictions.npz", targets=data["targets"].numpy(), **predictions)
    # One example per line keeps this reproducible manifest reviewable in Git.
    (output / "dataset.json").write_text("[\n" + ",\n".join(json.dumps(row) for row in data["rows"]) + "\n]\n")
    for filename, value in (("config.json", config),
                            ("fit_records.json", records), ("audit.json", checks)):
        (output / filename).write_text(json.dumps(value, indent=2) + "\n")
    with (output / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scores[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(scores)
    provenance = {"schema": "intrmotiv/synthetic-readout-probe/v1", "torch": torch.__version__,
                  "numpy": np.__version__, "python": sys.version, "config_sha256": digest(config),
                  "sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in (Path(__file__), KIT / "intrmotiv_transfer/state.py")},
                  "scope": "Offline supervised readout only; no DMLab, learned dynamics, recursive rollout or policy test."}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
