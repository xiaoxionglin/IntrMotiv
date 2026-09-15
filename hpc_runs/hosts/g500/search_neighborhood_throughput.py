"""Bounded, paired sampler search followed by qualified fresh production.

Uses the existing profiler, StudySpec/direct queue, and scientific transition.
Every candidate has identical four-arm seeds, frame budgets and batch geometry.
Outputs are exclusive; failed or interrupted searches require explicit recovery.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from hpc_runs.intrmotiv_study.direct import atomic_json, make_manifest, run_queue
from hpc_runs.intrmotiv_study.spec import load_study
from profile_training import resources

CANDIDATES = [
    (32, 8, 1, 8, 4),
    (32, 8, 2, 8, 4),
    (32, 8, 2, 8, 6),
    (48, 8, 2, 8, 4),
    (48, 8, 2, 8, 6),
    (32, 16, 2, 8, 4),
]
PROFILE_FRAMES = 1_048_576
SLOTS = [0, 1, 0, 1]


def score_candidate(directory, concurrency=4):
    """Require successful online runs and headroom; rank aggregate steady FPS."""
    summary = json.loads((directory / 'summary.json').read_text())
    samples = [json.loads(s) for s in (directory / 'resources.jsonl').read_text().splitlines()]
    valid = len(summary) == concurrency and all(
        x['returncode'] == 0 and not x['has_traceback'] and not x['deadline_signal_sent']
        and x['wandb_online'] and x['frames'] >= PROFILE_FRAMES and (x['fps_after_warmup'] or 0) > 0
        for x in summary)
    ram = min(x['available_ram_gib'] for x in samples)
    gpu = min(g['free_mib'] for x in samples for g in x['gpus'])
    return dict(eligible=valid and ram >= 64 and gpu >= 16384,
                aggregate_fps=sum(x['fps_after_warmup'] or 0 for x in summary),
                per_run_fps=[x['fps_after_warmup'] for x in summary],
                minimum_ram_gib=ram, minimum_gpu_free_mib=gpu)


def revised_study(template, destination, study_id, workers, envs, epochs=1, splits=2, concurrency=4):
    """Change only execution geometry and experiment/output identities."""
    document = json.loads(template.read_text())
    old_id = document['study_id']
    document = json.loads(json.dumps(document).replace(old_id, study_id))
    document['metadata']['resource_configuration'] = f'{workers}_workers_{envs}_envs_{epochs}_epochs_{splits}_splits_batch2048_{concurrency}_slots'
    document['metadata']['resource_rationale'] = 'Selected by matched four-arm direct throughput search; see search decision artifact.'
    args = document['training']['common_args']
    for key, value in [('num_workers', workers), ('num_envs_per_worker', envs), ('num_epochs', epochs), ('worker_num_splits', splits)]:
        matches = [i for i, item in enumerate(args) if item.startswith('--' + key + '=')]
        if len(matches) != 1:
            raise ValueError(f'Expected one {key} setting')
        args[matches[0]] = f'--{key}={value}'
    atomic_json(destination, document)
    return load_study(destination)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight-template', type=Path, required=True)
    p.add_argument('--production-template', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--panel', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true', help='Explicit continuation after the previous supervisor has been stopped')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=a.resume)
    study = load_study(a.preflight_template)
    names = [r.name for r in study.expand_runs()]
    if len(names) != 4:
        raise ValueError('Expected four preflight arms')
    results = []
    for workers, envs, epochs, splits, concurrency in CANDIDATES:
        if splits is None:
            splits = max((r for r in results if r["eligible"]), key=lambda r: r["aggregate_fps"])["splits"]
        slots = [i % 2 for i in range(concurrency)]
        selected_names = names + names[:concurrency - 4]
        name = f'w{workers}_e{envs}_ep{epochs}' + (f'_splits{splits}' if splits != 2 else '') + (f'_p{concurrency}' if concurrency != 4 else '')
        directory = a.output / name
        atomic_json(a.output / 'status.json', dict(stage='profiling', candidate=name, completed=results))
        command = [sys.executable, str(Path(__file__).with_name('profile_training.py')),
                   str(a.preflight_template), names[0], str(directory), '--workers', str(workers),
                   '--envs-per-worker', str(envs), '--epochs', str(epochs), '--worker-splits', str(splits), '--batch-size', '2048', '--frames', str(PROFILE_FRAMES),
                   '--seconds', '1800', '--gpus', *map(str, slots), '--runs', *selected_names]
        review = subprocess.run(command, check=True, capture_output=True, text=True)
        (a.output / (name + '_review.json')).write_text(review.stdout)
        if directory.exists():
            if not a.resume:
                raise FileExistsError(directory)
            # The interrupted supervisor's profiler may still be completing its
            # owned runs. Reuse its result rather than launching duplicates.
            deadline = time.monotonic() + 2100
            while not (directory / 'summary.json').exists():
                if time.monotonic() > deadline:
                    raise RuntimeError(f'Existing candidate {name} did not finish')
                time.sleep(10)
        else:
            with (a.output / (name + '_profiler.log')).open('x') as log:
                result = subprocess.run([*command, '--execute'], stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Candidate {name} failed; inspect its processes and logs before recovery')
        evidence = score_candidate(directory, concurrency)
        results.append(dict(workers=workers, envs=envs, epochs=epochs, splits=splits, concurrency=concurrency, nominal_optimizer_samples_per_second=evidence["aggregate_fps"] / 8 * epochs, **evidence))
        atomic_json(a.output / 'results.json', results)
    eligible = [r for r in results if r['eligible']]
    if not eligible:
        raise RuntimeError('No candidate passed throughput/resource checks')
    winner = max(eligible, key=lambda r: r['aggregate_fps'])
    atomic_json(a.output / 'decision.json', dict(winner=winner, candidates=results,
                selection='Maximum aggregate completed-frame throughput after warmup, comparing four and six simultaneous runs'))
    workers, envs, epochs, splits = winner['workers'], winner['envs'], winner['epochs'], winner['splits']
    preflight = revised_study(a.preflight_template, a.output/'selected_preflight.study.json',
                'intrmotiv_dg_neighborhood_preflight_gpu_core_20260915', workers, envs, epochs, splits, winner["concurrency"])
    production_path = a.output/'selected_production.study.json'
    revised_study(a.production_template, production_path,
                'intrmotiv_dg_neighborhood_production_gpu_core_20260915', workers, envs, epochs, splits, winner["concurrency"])
    manifest = make_manifest(preflight, a.source, SLOTS)
    atomic_json(a.output/'selected_preflight_review.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    atomic_json(a.output/'status.json', dict(stage='validating_selected_configuration', winner=winner))
    run_queue(manifest, resources)
    atomic_json(a.output/'status.json', dict(stage='scientific_gate_and_production', winner=winner))
    command = [sys.executable, str(Path(__file__).with_name('production_neighborhood.py')),
               '--preflight', str(preflight.output_root), '--production-study', str(production_path),
               '--panel', str(a.panel), '--output', str(a.output/'production_transition'),
               '--resource-decision', str(a.output/'decision.json')]
    subprocess.run(command, check=True)
    atomic_json(a.output/'status.json', dict(stage='production_and_evaluation_completed', winner=winner))


if __name__ == '__main__':
    main()
