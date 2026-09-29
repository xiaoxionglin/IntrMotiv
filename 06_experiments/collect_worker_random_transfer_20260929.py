"""Pin unsampled W&B reward histories for the two existing transfer studies.

Run identity and factors come from validated StudySpecs, never parsed run names.
Both arms are fetched from one backend so logging clocks and precision match.
This reads existing runs; it neither starts training nor changes remote data.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault('WANDB_SILENT', 'true')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hpc_runs.intrmotiv_study import load_study
import numpy as np
import pandas as pd
import wandb

DEFAULT_DATA = ROOT/'06_experiments/data/worker_random_transfer_20260929'
DEFAULT_STUDIES = Path('/home/xiaoxiong/SFgit/SF_hipposlam/hpc_runs/studies')
TAG = 'intrmotiv/reward/environment_mean'
CONFIG_KEYS = ('seed', 'study_id', 'study_condition', 'env', 'Hippo_L',
               'encoder_conv_architecture', 'transfer_scope', 'transfer_freeze_dg',
               'transfer_freeze_worker', 'transfer_graph', 'transfer_calibrate_frozen_dg',
               'transfer_model_path', 'hrl_manager_mode', 'hrl_direct_target_selection',
               'fixed_task_conditioning', 'fixed_task_goal_mixture', 'train_for_env_steps')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, default=DEFAULT_STUDIES)
    parser.add_argument('--out', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--entity', default=None)
    parser.add_argument('--refresh', action='store_true', help='Refetch already pinned histories')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'histories').mkdir(exist_ok=True)
    (args.out/'studies').mkdir(exist_ok=True)
    api = wandb.Api(timeout=30)
    entity = args.entity or api.default_entity
    selected = []
    studies = []
    for filename, arm in [('cued_reward5_transfer_20260925.study.json', 'W_WORKER'),
                          ('cued_reward5_frozen_dg_controls_20260925.study.json', 'W_RAND_DG')]:
        spec = load_study(args.study_dir/filename)
        runs = [r for r in spec.expand_runs() if r.factors['arm'] == arm]
        if len(runs) != 6:
            raise ValueError(f'Expected two architectures × three seeds for {arm}')
        project = spec.study_metadata['project']
        remote = list(api.runs(f'{entity}/{project}', filters={
            'config.study_condition': {'$in': sorted({r.condition for r in runs})}}, per_page=30))
        for run_spec in runs:
            matches = [r for r in remote if r.config.get('study_condition') == run_spec.condition
                       and r.config.get('seed') == run_spec.seed]
            if len(matches) != 1:
                raise ValueError(f'Expected one W&B run for {run_spec.name}; got {len(matches)}')
            remote_run = matches[0]
            expected = dict(arg[2:].split('=', 1) for arg in run_spec.args if arg.startswith('--') and '=' in arg)
            for key in CONFIG_KEYS:
                if key in expected:
                    actual = remote_run.config.get(key)
                    if str(actual).lower() != expected[key].lower():
                        raise ValueError(f'Config mismatch {run_spec.name}: {key}: {actual} != {expected[key]}')
            if remote_run.state != 'finished':
                raise ValueError(f'Run is not complete: {remote_run.name}, {remote_run.state}')
            selected.append((run_spec, remote_run, spec))
        shutil.copyfile(args.study_dir/filename, args.out/'studies'/filename)
        studies.append({'study_id':spec.study_id, 'schema':spec.raw['schema'],
                        'workflow_version':spec.declared_workflow_version,
                        'study_sha256':spec.fingerprint, 'selected_arm':arm, 'selected_runs':6,
                        'pinned_study':str(Path('studies')/filename)})

    def collect(item):
        run_spec, remote_run, spec = item
        # scan_history is unsampled. W&B _step is NOT environment progress.
        filename = run_spec.name+'.csv'
        path = args.out/'histories'/filename
        raw = (pd.read_csv(path) if path.exists() and not args.refresh else
               pd.DataFrame(remote_run.scan_history(
                   keys=['_step', 'train/env_steps', TAG], page_size=5000)))
        if raw.empty or raw[['train/env_steps', TAG]].isna().any().any():
            raise ValueError(f'Missing reward/frame history: {run_spec.name}')
        if not np.isfinite(raw[['train/env_steps', TAG]].to_numpy(float)).all():
            raise ValueError(f'Nonfinite reward history: {run_spec.name}')
        raw = raw.sort_values('_step')
        duplicates = int(raw['train/env_steps'].duplicated().sum())
        # Retain last logged value at a repeated frame, matching the TB adapter.
        history = raw.drop_duplicates('train/env_steps', keep='last').sort_values('train/env_steps')
        last = float(history['train/env_steps'].iloc[-1])
        # Completed jobs may stop logging just before their declared budget.
        # Match the canonical scalar adapter's 99.5% coverage check and expose
        # the endpoint carry explicitly in the collection manifest.
        if last < .995 * 75_000_000:
            raise ValueError(f'{run_spec.name} ends before the 75M comparison: {last}')
        raw.to_csv(path, index=False)
        print(f'{run_spec.name}: {len(history)} reward samples; {last:g} frames', flush=True)
        return {'run_name':run_spec.name, 'condition':run_spec.condition,
                'architecture':next(b.label for b in spec.bases if b.name == run_spec.base),
                'arm':run_spec.factors['arm'], 'seed':run_spec.seed,
                'study_id':spec.study_id, 'study_schema':spec.raw['schema'],
                'workflow_version':spec.declared_workflow_version, 'study_sha256':spec.fingerprint,
                'wandb_entity':entity, 'wandb_project':spec.study_metadata['project'],
                'wandb_run_id':remote_run.id, 'wandb_run_name':remote_run.name,
                'wandb_url':remote_run.url, 'wandb_state':remote_run.state,
                'history':str(Path('histories')/filename), 'history_rows':len(raw),
                'duplicate_frame_rows':duplicates, 'last_logged_frames':last,
                'endpoint_carry_frames':max(0., 75_000_000-last),
                'maximum_logging_gap_frames':float(history['train/env_steps'].diff().max()),
                'config':{k:remote_run.config.get(k) for k in CONFIG_KEYS},
                'source_checkpoint_sha256':run_spec.metadata['source_sha256']}

    records = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        for future in as_completed([executor.submit(collect, item) for item in selected]):
            records.append(future.result())
    records.sort(key=lambda r:(r['architecture'], r['arm'], r['seed']))
    manifest = {'schema':'intrmotiv/worker-random-transfer-histories/v1',
                'collected_utc':datetime.now(timezone.utc).isoformat(),
                'backend':'W&B scan_history (unsampled)', 'wandb_version':wandb.__version__,
                'frame_axis':'train/env_steps; W&B numeric precision retained, no checkpoint age inferred',
                'metric':TAG, 'studies':studies, 'runs':records}
    (args.out/'collection_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Pinned {len(records)} completed runs: {args.out}')


if __name__ == '__main__':
    main()
