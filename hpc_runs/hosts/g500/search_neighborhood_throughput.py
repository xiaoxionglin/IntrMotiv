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

from hpc_runs.intrmotiv_study.direct import atomic_json, make_manifest, run_queue
from hpc_runs.intrmotiv_study.spec import load_study
from profile_training import resources

CANDIDATES = [(32, 8, 1), (48, 8, 1), (32, 16, 1), (32, 8, 2), (48, 8, 2)]
SLOTS = [0, 1, 0, 1]


def score_candidate(directory):
    """Require successful online runs and headroom; rank aggregate steady FPS."""
    summary = json.loads((directory / 'summary.json').read_text())
    samples = [json.loads(s) for s in (directory / 'resources.jsonl').read_text().splitlines()]
    valid = len(summary) == 4 and all(
        x['returncode'] == 0 and not x['has_traceback'] and not x['deadline_signal_sent']
        and x['wandb_online'] and x['frames'] >= 262144 and (x['fps_after_warmup'] or 0) > 0
        for x in summary)
    ram = min(x['available_ram_gib'] for x in samples)
    gpu = min(g['free_mib'] for x in samples for g in x['gpus'])
    return dict(eligible=valid and ram >= 64 and gpu >= 16384,
                aggregate_fps=sum(x['fps_after_warmup'] or 0 for x in summary),
                per_run_fps=[x['fps_after_warmup'] for x in summary],
                minimum_ram_gib=ram, minimum_gpu_free_mib=gpu)


def revised_study(template, destination, study_id, workers, envs, epochs=1):
    """Change only execution geometry and experiment/output identities."""
    document = json.loads(template.read_text())
    old_id = document['study_id']
    document = json.loads(json.dumps(document).replace(old_id, study_id))
    document['metadata']['resource_configuration'] = f'{workers}_workers_{envs}_envs_{epochs}_epochs_batch2048_4_slots'
    document['metadata']['resource_rationale'] = 'Selected by matched four-arm direct throughput search; see search decision artifact.'
    args = document['training']['common_args']
    for key, value in [('num_workers', workers), ('num_envs_per_worker', envs), ('num_epochs', epochs)]:
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
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    study = load_study(a.preflight_template)
    names = [r.name for r in study.expand_runs()]
    if len(names) != 4:
        raise ValueError('Expected four preflight arms')
    results = []
    for workers, envs, epochs in CANDIDATES:
        name = f'w{workers}_e{envs}_ep{epochs}'
        directory = a.output / name
        atomic_json(a.output / 'status.json', dict(stage='profiling', candidate=name, completed=results))
        command = [sys.executable, str(Path(__file__).with_name('profile_training.py')),
                   str(a.preflight_template), names[0], str(directory), '--workers', str(workers),
                   '--envs-per-worker', str(envs), '--epochs', str(epochs), '--batch-size', '2048', '--frames', '262144',
                   '--seconds', '1800', '--gpus', *map(str, SLOTS), '--runs', *names]
        review = subprocess.run(command, check=True, capture_output=True, text=True)
        (a.output / (name + '_review.json')).write_text(review.stdout)
        with (a.output / (name + '_profiler.log')).open('x') as log:
            result = subprocess.run([*command, '--execute'], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            # Never advance into another candidate with unknown surviving children.
            raise RuntimeError(f'Candidate {name} failed; inspect its processes and logs before recovery')
        evidence = score_candidate(directory)
        results.append(dict(workers=workers, envs=envs, epochs=epochs, nominal_optimizer_samples_per_second=evidence["aggregate_fps"] / 8 * epochs, **evidence))
        atomic_json(a.output / 'results.json', results)
    eligible = [r for r in results if r['eligible']]
    if not eligible:
        raise RuntimeError('No candidate passed throughput/resource checks')
    winner = max(eligible, key=lambda r: r['aggregate_fps'])
    atomic_json(a.output / 'decision.json', dict(winner=winner, candidates=results,
                selection='Maximum aggregate completed-frame throughput after warmup, four simultaneous arms'))
    workers, envs, epochs = winner['workers'], winner['envs'], winner['epochs']
    preflight = revised_study(a.preflight_template, a.output/'selected_preflight.study.json',
                'intrmotiv_dg_neighborhood_preflight_20260915_aggressive', workers, envs, epochs)
    production_path = a.output/'selected_production.study.json'
    revised_study(a.production_template, production_path,
                'intrmotiv_dg_neighborhood_production_20260915_aggressive', workers, envs, epochs)
    manifest = make_manifest(preflight, a.source, SLOTS)
    atomic_json(a.output/'selected_preflight_review.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    atomic_json(a.output/'status.json', dict(stage='validating_selected_configuration', winner=winner))
    run_queue(manifest, resources)
    atomic_json(a.output/'status.json', dict(stage='scientific_gate_and_production', winner=winner))
    command = [sys.executable, str(Path(__file__).with_name('production_neighborhood.py')),
               '--preflight', str(preflight.output_root), '--production-study', str(production_path),
               '--panel', str(a.panel), '--output', str(a.output/'production_transition')]
    subprocess.run(command, check=True)
    atomic_json(a.output/'status.json', dict(stage='production_and_evaluation_completed', winner=winner))


if __name__ == '__main__':
    main()
