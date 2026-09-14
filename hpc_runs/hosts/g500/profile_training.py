"""Bounded direct-SF throughput trials with host-wide resource sampling.

Consumes a canonical StudySpec and exact run identity. Profiling overrides are
recorded separately; they never alter the scientific study. No Slurm, automatic
process killing, credential persistence, or live-run batch-size changes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import psutil

from hpc_runs.intrmotiv_study.spec import load_study


def resources():
    result = {"time": time.time(), "cpu_percent": psutil.cpu_percent(),
              "load": os.getloadavg(), "available_ram_gib": psutil.virtual_memory().available / 2**30}
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True,
    )
    result["gpus"] = [dict(zip(("index", "free_mib", "utilization", "power_w"), map(float, row.split(","))))
                      for row in query.stdout.strip().splitlines()]
    result["pressure"] = {name: Path(f"/proc/pressure/{name}").read_text().strip()
                          for name in ("cpu", "memory", "io")}
    return result


def frame_count(path):
    if not path.exists():
        return 0
    matches = re.findall(r"Total num frames: (\d+)", path.read_text(errors="replace"))
    return int(matches[-1]) if matches else 0


def measured_fps(samples, warmup_seconds=40):
    """Use completed frame counters after progress starts, excluding warmup."""
    advancing = [s for s in samples if s[1] > 0]
    if not advancing:
        return None
    steady = [s for s in advancing if s[0] >= advancing[0][0] + warmup_seconds]
    if len(steady) < 3 or steady[-1][0] - steady[0][0] < 20:
        return None
    return (steady[-1][1] - steady[0][1]) / (steady[-1][0] - steady[0][0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("run")
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--seconds", type=int, default=900, help="External wall-clock safety limit")
    parser.add_argument("--frames", type=int, default=131072, help="Normal SF frame-count termination")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 64 or args.batch_size % 64:
        parser.error("workers must be positive and batch size a positive multiple of 64")
    root = Path(os.environ.get("INTRMOTIV_ROOT", "/scratch/lin/IntrMotiv")).resolve()
    output = args.output.resolve()
    if not output.is_relative_to(root):
        parser.error("profiling output must be inside INTRMOTIV_ROOT")
    study = load_study(args.study)
    run = next(r for r in study.expand_runs() if r.name == args.run)
    specs = []
    for slot, gpu in enumerate(args.gpus):
        name = f"PROFILE_W{args.workers}_B{args.batch_size}_P{len(args.gpus)}_{output.name}_{slot}"
        overrides = {
            "experiment": name, "train_dir": str(output / "training"), "device": "gpu",
            "num_workers": args.workers, "num_envs_per_worker": 2,
            "batch_size": args.batch_size, "train_for_seconds": args.seconds,
            "train_for_env_steps": args.frames, "decorrelate_experience_max_seconds": 0,
            "decorrelate_envs_on_one_worker": False, "set_workers_cpu_affinity": False,
            "dmlab_level_cache_path": str(root / "cache/dmlab"),
            "wandb_dir": str(root / "logs/wandb"), "with_wandb": True,
            "wandb_project": "SF_IntrMotiv_DGNeighborhood", "wandb_group": "g500_resource_profile_20260914",
            "wandb_tags": "resource_profile", "save_every_sec": 100000,
            "save_best_every_sec": 100000, "save_milestones_sec": 100000,
            "online_spatial_telemetry": True, "online_spatial_workspace_root": str(root),
            "online_spatial_output_root": str(output / "spatial"),
        }
        tokens = [s for s in run.args if s.split("=", 1)[0].removeprefix("--") not in overrides]
        tokens += [f"--{k}={v}" for k, v in overrides.items()]
        command = [sys.executable, "-m", "sf_working_directories.IntrMotiv.dmlab.train_hipposlam", *tokens]
        # No NEMO workspace or home bulk-output paths may survive the overrides.
        if any("/work/classic/" in s or "/home/fr/" in s for s in command):
            raise ValueError("untranslated NEMO path in profiling command")
        specs.append({"name": name, "gpu": gpu, "command": command, "overrides": overrides})
    manifest = {**study.provenance(), "source_run": run.as_dict(), "runs": specs,
                "workers": args.workers, "batch_size": args.batch_size,
                "concurrency": len(args.gpus), "seconds": args.seconds, "frames": args.frames}
    if not args.execute:
        print(json.dumps(manifest, indent=2))
        return
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite profiling output: {output}")
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    available = resources()
    counts = {gpu: args.gpus.count(gpu) for gpu in set(args.gpus)}
    if available["available_ram_gib"] < 64 + 8 * len(args.gpus):
        raise RuntimeError("Insufficient host RAM headroom for profiling admission")
    for gpu, count in counts.items():
        info = next(g for g in available["gpus"] if g["index"] == gpu)
        if info["free_mib"] < 16384 * count + 16384 or info["utilization"] > 50:
            raise RuntimeError(f"GPU {gpu} currently lacks profiling headroom")
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Online W&B credential must be supplied through the environment")
    records = []
    start = time.monotonic()
    with (output / "resources.jsonl").open("w") as sample_file:
        for spec in specs:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(spec["gpu"]), WANDB_MODE="online",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            log_path = output / (spec["name"] + ".log")
            handle = log_path.open("w")
            process = subprocess.Popen(spec["command"], env=env, stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            records.append({"process": process, "handle": handle, "log": log_path, "samples": [], "spec": spec})
        (output / "processes.json").write_text(json.dumps([
            {"pid": r["process"].pid, "create_time": psutil.Process(r["process"].pid).create_time(),
             "name": r["spec"]["name"], "gpu": r["spec"]["gpu"]} for r in records], indent=2))
        deadline_signal_sent = False
        while any(r["process"].poll() is None for r in records):
            sample = resources()
            sample["runs"] = []
            for record in records:
                frames = frame_count(record["log"])
                if record["process"].poll() is None:
                    record["samples"].append((sample["time"], frames))
                sample["runs"].append({"name": record["spec"]["name"], "frames": frames,
                                       "returncode": record["process"].poll()})
            sample_file.write(json.dumps(sample) + "\n")
            sample_file.flush()
            failed = any("Traceback (most recent call last)" in r["log"].read_text(errors="replace")
                         for r in records)
            if (failed or time.monotonic() - start > args.seconds) and not deadline_signal_sent:
                for r in records:
                    if r["process"].poll() is None:
                        # SF's parent requests a controlled stop at worker boundaries.
                        # This runtime does not advance its train_for_seconds counter.
                        os.kill(r["process"].pid, signal.SIGINT)
                deadline_signal_sent = True
            if time.monotonic() - start > args.seconds + 300:
                raise RuntimeError("Profiling process did not shut down; inspect recorded owned PIDs")
            time.sleep(5)
    summary = []
    for record in records:
        record["handle"].close()
        content = record["log"].read_text(errors="replace")
        summary.append({"name": record["spec"]["name"], "returncode": record["process"].returncode,
                        "frames": frame_count(record["log"]), "fps_after_warmup": measured_fps(record["samples"]),
                        "has_traceback": "Traceback (most recent call last)" in content,
                        "wandb_online": "https://wandb.ai/" in content,
                        "deadline_signal_sent": deadline_signal_sent})
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if any(r["returncode"] != 0 or r["deadline_signal_sent"] or r["frames"] < args.frames
           or r["has_traceback"] or not r["wandb_online"] or not r["fps_after_warmup"] for r in summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
