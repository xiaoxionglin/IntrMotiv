"""Thin read-only bridge to GeoMem's canonical task-graph contracts.

GeoMem's graph_process arrays describe a fixed-policy JOINT edge distribution.
They validate observations and scheduling, but cannot be used directly as a
conditional action kernel. This module reconstructs P(next | state, action)
from authoritative exported edges and keeps unavailable actions unavailable.
It performs no DMLab import, package installation, or GeoMem filesystem write.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from .provenance import audit_source_snapshots, snapshot_sources


DEFAULT_GEOMEM = Path("/home/xiaoxiong/Desktop/Projects/geomem")


def load_source_module(path, stem):
    """Load a specific source file without modifying package search paths."""
    path = Path(path).resolve()
    name = f"first_hit_geomem_{stem}_{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses consult sys.modules while processing annotations.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def geomem_source_paths(root=DEFAULT_GEOMEM):
    source = Path(root).resolve() / "geomem-env/src/geomem_env"
    return {name: source / f"{name}.py" for name in ("task_graphs", "graph_process")}


def load_geomem_modules(root=DEFAULT_GEOMEM):
    return {name: load_source_module(path, name) for name, path in geomem_source_paths(root).items()}


def task_document(task):
    """Serialize the existing TaskGraph in its canonical ordered-node contract.

    Mirrors only the small serialization boundary used by GeoMem's atlas; it
    does not copy task constructors, transition rules or observation functions.
    """
    states = list(task.graph.nodes)
    index = {state: i for i, state in enumerate(states)}
    nodes = [dict(id=index[state], state=state, **dict(task.graph.nodes[state])) for state in states]
    edges = [dict(source=index[source], target=index[target], **dict(data))
             for source, target, data in task.graph.edges(data=True)]
    return {"name": task.name, "metadata": dict(task.metadata), "nodes": nodes, "edges": edges}


@dataclass(frozen=True)
class ControlledGraph:
    transition: np.ndarray
    available_actions: np.ndarray
    observation_ids: np.ndarray
    sensory_symbols: tuple[str, ...]
    latent_states: tuple[object, ...]
    action_labels: tuple[str, ...]
    terminal: np.ndarray
    initial: np.ndarray
    metadata: dict

    def detector(self, sensory_symbol):
        """A sensory identity denotes its entire alias class, never one node."""
        if sensory_symbol not in self.sensory_symbols:
            raise ValueError("Requested sensory identity is absent")
        return np.asarray([symbol == sensory_symbol for symbol in self.sensory_symbols], dtype=bool)


def conditional_action_arrays(document, graph_process_arrays):
    """Validate and extract a purely controlled graph without policy dilution.

    Event/reset graphs require an explicit scheduler and are intentionally
    refused. No fixed-policy event token is reinterpreted as a chosen action.
    Terminal and unavailable action rows stay zero, accompanied by a mask.
    """
    process, metadata = graph_process_arrays(document)
    kinds = {edge["kind"] for edge in document["edges"]}
    if kinds != {"action"}:
        raise ValueError(f"Explicit scheduler required for non-control edge kinds: {sorted(kinds - {'action'})}")
    actions = tuple(metadata["actions"])
    action_index = {label: i for i, label in enumerate(actions)}
    states = len(document["nodes"])
    transition = np.zeros((states, len(actions), states), dtype=float)
    for edge in document["edges"]:
        transition[edge["source"], action_index[edge["action"]], edge["target"]] += edge["probability"]
    available = transition.sum(-1) > 0
    if not np.allclose(transition.sum(-1)[available], 1):
        raise ValueError("Conditional action probabilities must sum to one")
    return ControlledGraph(
        transition, available, process["observation"].copy(),
        tuple(node["observation"] for node in document["nodes"]),
        tuple(node.get("state", node["id"]) for node in document["nodes"]),
        actions, np.array([node.get("terminal", False) for node in document["nodes"]], bool),
        process["initial"].copy(),
        {**metadata, "kernel_semantics": "conditional P(successor|state,action), not fixed-policy joint probabilities",
         "unavailable_actions": "zero rows with false availability mask; no invented transitions",
         "privileged_latent_states": "audit/evaluation only, not policy observations"},
    )


def _success_action_sets(document):
    """One-step correctness sets solely for a labeled memory-impossibility audit."""
    successful = {node["id"] for node in document["nodes"] if node.get("success", False)}
    by_action = defaultdict(lambda: defaultdict(list))
    for edge in document["edges"]:
        if edge["kind"] == "action":
            by_action[edge["source"]][edge["action"]].append(edge)
    return {state: {action for action, edges in branches.items()
                    if all(edge["target"] in successful for edge in edges)}
            for state, branches in by_action.items()}


def _shortest_prefix(document, target):
    """Any exported initial node to target; never used to train a controller."""
    starts = [n["id"] for n in document["nodes"] if n.get("initial", False)]
    adjacent = defaultdict(list)
    for edge in document["edges"]:
        adjacent[edge["source"]].append(edge["target"])
    queue = deque((node, [node]) for node in starts)
    seen = set(starts)
    while queue:
        node, path = queue.popleft()
        if node == target:
            return path
        for successor in adjacent[node]:
            if successor not in seen:
                seen.add(successor)
                queue.append((successor, path + [successor]))
    return None


def _ca3_word(observation_ids, vocabulary, length, repeat=1):
    """Exact shift/inject algebra for a categorical diagnostic (no trained net)."""
    memory = np.zeros((vocabulary, length), dtype=np.int64)
    for observation in observation_ids:
        memory[:, 1:] = memory[:, :-1].copy()
        memory[:, 0] = 0
        memory[observation, :repeat] += 1
    return memory


def memory_ambiguity_witnesses(document, process):
    """Find same-observation decision states with disjoint correct actions.

    Correctness labels belong to GeoMem's source task and are diagnostic oracle
    information. They are NOT reward-free IntrMotiv learning labels. For each
    pair, equal source probability gives a one-step memoryless ceiling of .5.
    Prefixes and finite CA3 comparisons show whether a public cue can resolve it.
    """
    correct = _success_action_sets(document)
    groups = defaultdict(list)
    for state, choices in correct.items():
        if choices:
            groups[int(process["observation"][state])].append(state)
    witnesses = []
    for observation, group in groups.items():
        for i, left in enumerate(group):
            for right in group[i + 1:]:
                if not correct[left].isdisjoint(correct[right]):
                    continue
                prefixes = [_shortest_prefix(document, state) for state in (left, right)]
                words = [[int(process["observation"][state]) for state in prefix] if prefix else []
                         for prefix in prefixes]
                max_length = max(map(len, words))
                comparisons = []
                for length in (1, 2, 4, 8, max_length):
                    if length < 1 or any(row["memory_length"] == length for row in comparisons):
                        continue
                    same = np.array_equal(*[_ca3_word(word, process["n_observations"], length) for word in words])
                    comparisons.append({"memory_length": length, "repeat": 1, "same_ca3": bool(same)})
                witness = {"left_node": left, "right_node": right,
                                  "observation_id_including_action_availability": observation,
                                  "left_correct_actions": sorted(correct[left]), "right_correct_actions": sorted(correct[right]),
                                  "equal_context_memoryless_success_upper_bound": 0.5,
                                  "prefix_nodes": prefixes, "prefix_observation_ids": words,
                                  "prefixes_observably_distinct": words[0] != words[1],
                                  "finite_ca3_comparisons": comparisons}
                reset_prefixes = []
                for prefix in prefixes:
                    candidates = [edge["source"] for edge in document["edges"]
                                  if edge["kind"] == "reset" and prefix and edge["target"] == prefix[0]
                                  and document["nodes"][edge["source"]].get("success", False)]
                    if candidates:
                        reset_prefixes.append([candidates[0], *prefix])
                if len(reset_prefixes) == 2:
                    reset_words = [[int(process["observation"][state]) for state in prefix]
                                   for prefix in reset_prefixes]
                    witness["observed_previous_outcome_prefixes"] = reset_prefixes
                    witness["reset_preserving_memory_can_distinguish"] = reset_words[0] != reset_words[1]
                    witness["reset_memory_boundary"] = (
                        "Previous public outcome is informative only if memory persists across the exported reset. "
                        "A physical reset that clears memory restores the initial ambiguity."
                    )
                witnesses.append(witness)
    return witnesses


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run_reference_audit(output, geomem_root=DEFAULT_GEOMEM):
    """Export and audit three reused graphs; no training or native environment."""
    output, geomem_root = Path(output).resolve(), Path(geomem_root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = geomem_source_paths(geomem_root)
    captured = snapshot_sources([*paths.values(), Path(__file__), Path(__file__).with_name("provenance.py")], output / "source_snapshot")
    records = {entry["source_path"]: output / "source_snapshot" / entry["snapshot_path"] for entry in captured["sources"]}
    # Import the archived GeoMem files, so later source edits cannot affect this
    # run. Python/networkx/scipy are pre-existing dependencies; nothing installs.
    modules = {name: load_source_module(records[str(path.resolve())], name) for name, path in paths.items()}
    task_graphs, validate = modules["task_graphs"], modules["graph_process"].graph_process_arrays
    config = {"schema": "intrmotiv/geomem-reference-audit/v1", "training": False,
              "native_dmlab": False, "package_installation": False,
              "constructors": [{"name": "delayed_match", "delay": 4}, {"name": "context_decision", "delay": 2}, {"name": "alternation", "length": 4}],
              "goal_semantics": "source-task correctness labels used ONLY for information-limit diagnostics",
              "geomem_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=geomem_root, text=True).strip()}
    _write_json(output / "prerun.json", config)
    graphs = [("delayed_match", task_graphs.delayed_match(delay=4)),
              ("context_decision", task_graphs.context_decision(delay=2)),
              ("alternation", task_graphs.alternation(length=4))]
    summaries = []
    for slug, task in graphs:
        document = task_document(task)
        _write_json(output / f"{slug}.json", document)
        process, metadata = validate(document)
        np.savez_compressed(output / f"{slug}_fixed_policy_process.npz", **process)
        try:
            conditional_action_arrays(document, validate)
            rejection = None
        except ValueError as error:
            rejection = str(error)
        kinds = {kind: sum(edge["kind"] == kind for edge in document["edges"])
                 for kind in sorted({edge["kind"] for edge in document["edges"]})}
        witnesses = memory_ambiguity_witnesses(document, process)
        _write_json(output / f"{slug}_memory_witnesses.json", witnesses)
        summary = {"graph": slug, "nodes": len(document["nodes"]), "edges": len(document["edges"]),
                   "edge_kinds": kinds, "observation_classes_including_action_mask": int(process["n_observations"]),
                   "raw_sensory_classes": len({node["observation"] for node in document["nodes"]}),
                   "initial_distribution_assumption": metadata["initial_assumption"],
                   "pure_control_adapter_accepted": rejection is None, "rejection": rejection,
                   "opposing_action_alias_pairs": len(witnesses),
                   "distinct_observed_prefix_pairs": sum(w["prefixes_observably_distinct"] for w in witnesses),
                   "document_sha256": hashlib.sha256((output / f"{slug}.json").read_bytes()).hexdigest(),
                   "process_sha256": hashlib.sha256((output / f"{slug}_fixed_policy_process.npz").read_bytes()).hexdigest(),
                   "witness_sha256": hashlib.sha256((output / f"{slug}_memory_witnesses.json").read_bytes()).hexdigest(),
                   "reset_semantics": metadata["reset_assumption"]}
        summaries.append(summary)
    _write_json(output / "summary.json", summaries)
    _write_json(output / "source_audit.json", audit_source_snapshots(output / "source_snapshot"))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geomem-root", type=Path, default=DEFAULT_GEOMEM)
    args = parser.parse_args()
    print(json.dumps(run_reference_audit(args.output, args.geomem_root), indent=2))


if __name__ == "__main__":
    main()
