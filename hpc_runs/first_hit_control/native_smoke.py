"""Qualify local native no-reward DMLab replay, without any learned checkpoint.

Runfiles are prepared through the canonical helper. This deliberately tests
infrastructure only: no DG representation, Q policy, or subgoal control is run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from hpc_runs.intrmotiv_study.runfiles import prepare_runfiles


LEVEL = "openfield_map2_fixed_loc3_fixedlength_noreward"
# Exact existing reduced runtime action interface, including strafe and turns
# combined with forward movement. No idealized motion model is substituted.
ACTIONS = np.asarray(
    [[0, 0, 0, 1, 0, 0, 0], [0, 0, -1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0],
     [-20, 0, 0, 1, 0, 0, 0], [20, 0, 0, 1, 0, 0, 0]], dtype=np.intc,
)


def qualify(source: Path, runfiles: Path, output: Path, *, seed: int = 99, decisions: int = 64) -> dict:
    """Run an identical action prefix in two fresh native engines and certify it."""
    import deepmind_lab

    if decisions < 1:
        raise ValueError("decisions must be positive")
    if output.exists():
        raise FileExistsError(f"Preserve the existing qualification: {output}")
    source = source.resolve(strict=True)
    runfiles = prepare_runfiles(source, runfiles)
    deepmind_lab.set_runfiles_path(str(runfiles))
    word = np.random.default_rng(seed).integers(0, len(ACTIONS), decisions)
    signatures, rewards, elapsed = [], [], []
    for _ in range(2):
        start = time.monotonic()
        lab = deepmind_lab.Lab(
            LEVEL, ["RGB_INTERLEAVED", "DEBUG.POS.TRANS"],
            config={"width": "84", "height": "84", "fps": "60"}, renderer="software",
        )
        digest, reward_sum = hashlib.sha256(), 0.0
        try:
            lab.reset(seed=seed)
            for action in word:
                observations = lab.observations()
                for name in sorted(observations):
                    digest.update(name.encode())
                    digest.update(observations[name].tobytes())
                reward_sum += lab.step(ACTIONS[action], num_steps=8)
                if not lab.is_running():
                    raise RuntimeError("Native smoke crossed a physical episode boundary")
            final = lab.observations()
            for name in sorted(final):
                digest.update(name.encode())
                digest.update(final[name].tobytes())
            if final["RGB_INTERLEAVED"].shape != (84, 84, 3) or reward_sum != 0:
                raise RuntimeError("Unexpected native image shape or nonzero external reward")
            signatures.append(digest.hexdigest())
            rewards.append(reward_sum)
        finally:
            lab.close()
        elapsed.append(time.monotonic() - start)
    if signatures[0] != signatures[1]:
        raise RuntimeError("Fresh native engines did not reproduce the identical action prefix")
    provenance = {"source_root": str(source), "files": {}}
    provenance.update(
        schema="intrmotiv/first-hit-local-infrastructure-qualification/v1",
        runtime_commit=subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        runtime_dirty=bool(subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()),
        level=LEVEL, runfiles=str(runfiles), decisions_per_engine=decisions, engines=2, frameskip=8, seed=seed,
        actions=ACTIONS.tolist(), action_word=word.tolist(), observation_hashes=signatures,
        exact_native_prefix=True, reward_sums=rewards, elapsed_seconds=elapsed,
        native_binding=deepmind_lab.__file__,
        scope="infrastructure only; no learned checkpoint, trained DG, Q policy, or goal control tested",
    )
    project = Path(__file__).resolve().parents[2]
    paths = [
        Path(__file__), source / f"deepmindlab_patch/game_scripts/levels/{LEVEL}.lua",
        source / "deepmindlab_patch/game_scripts/levels/openfield_map2_fixed_loc3_noreward.lua",
        project / "hpc_runs/intrmotiv_study/runfiles.py",
    ]
    for path in paths:
        provenance["files"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runfiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--decisions", type=int, default=64)
    args = parser.parse_args()
    result = qualify(args.source, args.runfiles, args.output, seed=args.seed, decisions=args.decisions)
    print(json.dumps({"passed": True, "output": str(args.output), "scope": result["scope"]}))


if __name__ == "__main__":
    main()
