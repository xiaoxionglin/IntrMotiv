"""Read-only retrieval payload for already submitted evaluation/analysis jobs.

Execute with system Python on NEMO, passing original|stationary and 00..03|analysis.
Only completed jobs expose scientific summaries. Failed jobs expose bounded log
tails and authoritative remote hashes. Never imports models or environments,
submits work, changes remote files, or transfers raw trials/replay/checkpoints.
"""

import argparse
import base64
import datetime
import hashlib
import json
from pathlib import Path
import subprocess


BASE = Path('/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/'
            'train_dir/analysis/first_hit_control_20261127')
COHORTS = {
    'original': ('main_r1', ['8950234', '8950235', '8950236', '8950237'], '8950569'),
    'stationary': ('stationary_projected_r1', ['8951060', '8951061', '8951062', '8951063'], '8951341'),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort', choices=COHORTS)
    parser.add_argument('stage', choices=['00', '01', '02', '03', 'analysis'])
    args = parser.parse_args()
    folder, evaluators, analysis = COHORTS[args.cohort]
    root = BASE / folder
    job_id = analysis if args.stage == 'analysis' else evaluators[int(args.stage)]
    command = ['sacct', '-j', job_id, '--format=JobID,State,ExitCode,Elapsed,MaxRSS',
               '--parsable2', '--noheader']
    accounting = subprocess.run(command, text=True, capture_output=True, check=True)
    records = [line.split('|') for line in accounting.stdout.splitlines()]
    state = next(row for row in records if row[0] == job_id)
    result = dict(schema='intrmotiv/completed-job-light-retrieval/v1',
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        cohort=args.cohort, stage=args.stage, job_id=job_id, remote_root=str(root),
        accounting=dict(command=command, stdout=accounting.stdout, stderr=accounting.stderr,
                        returncode=accounting.returncode),
        state=state[1], exit_code=state[2], complete=state[1:3] == ['COMPLETED', '0:0'],
        raw_arrays_and_trial_traces_transferred=False, files=[], skipped=[])
    total_bytes = 0

    def include(relative, *, required=True):
        nonlocal total_bytes
        path = root / relative
        if not path.is_file():
            if required:
                raise FileNotFoundError(path)
            result['skipped'].append(dict(path=str(path), reason='not present'))
            return
        size = path.stat().st_size
        if size > 3_000_000 or total_bytes + size > 15_000_000:
            result['skipped'].append(dict(path=str(path), bytes=size,
                sha256=sha256(path), reason='exceeds lightweight retrieval bound'))
            return
        payload = path.read_bytes()
        result['files'].append(dict(relative_path=str(relative), remote_path=str(path),
            bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
            base64=base64.b64encode(payload).decode('ascii')))
        total_bytes += len(payload)

    if result['complete']:
        if args.stage == 'analysis':
            for relative in ['aggregate_v1/aggregate.json', 'aggregate_v1/protocol.json',
                             'aggregate_v1/per_goal.csv', 'aggregate_v1/source_target.csv',
                             'aggregate_v1/competing_targets.csv',
                             'aggregate_v1/source_snapshot/manifest.json',
                             'trace_analysis_v1/summary.json',
                             'trace_analysis_v1/source_snapshot/manifest.json',
                             'figures_v1/mapping.json']:
                include(relative)
            mapping = json.loads((root / 'figures_v1/mapping.json').read_text())
            for name in mapping['outputs']:
                if not name.endswith('.png'):
                    continue
                path = root / 'figures_v1' / name
                assert sha256(path) == mapping['outputs'][name]
                include(Path('figures_v1') / name)
            if args.cohort == 'stationary':
                include('analysis_execution_v1/source_before.json')
                include('analysis_execution_v1/source_after.json')
        else:
            directory = Path(f'evaluation_sources_{args.stage}')
            for name in ['summary.json', 'pre_run_manifest.json', 'source_snapshot/manifest.json']:
                include(directory / name)
            if args.cohort == 'stationary':
                include(directory / 'block_journal.json')
            summary = json.loads((root / directory / 'summary.json').read_text())
            result['declared_source_panel'] = summary['source_panel']
            result['missing_preserved'] = summary.get('missing')
            result['trial_file_remote_sha256'] = summary['trials_sha256']
            result['source_audit'] = {key: summary['source_audit'][key]
                for key in ['archive_valid', 'current_sources_unchanged']}
    if args.stage == 'analysis':
        prefix = 'fh_analysis_main' if args.cohort == 'original' else 'stationary_analysis'
    else:
        prefix = f'evaluate_{args.stage}'
    result['bounded_log_tails'] = []
    for extension in ['out', 'err']:
        path = root / 'slurm' / f'{prefix}_{job_id}.{extension}'
        if path.is_file():
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - 12000))
                tail = stream.read().decode(errors='replace')
            result['bounded_log_tails'].append(dict(path=str(path), bytes=path.stat().st_size,
                sha256=sha256(path), tail=tail))
    result['transferred_file_bytes'] = total_bytes
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
