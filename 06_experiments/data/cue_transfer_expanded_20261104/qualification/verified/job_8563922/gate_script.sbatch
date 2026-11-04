#!/usr/bin/env bash
# Replaces intentionally cancelled gate8563876; no training or Slurm submission occurs here.
# Create the qualification log directory before sbatch. No jobs are submitted here.
# Optionally export INTRMOTIV_QUALIFICATION_JOB_IDS with the same colon-separated IDs.
#SBATCH --job-name=cue-expanded-recovery-gate
#SBATCH --partition=h200,l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/cue_transfer_expanded_20261104/qualification/recovery-gate-%j.out
#SBATCH --error=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/cue_transfer_expanded_20261104/qualification/recovery-gate-%j.err
set -euo pipefail
workspace=/work/classic/fr_xl1014-corridor-geometry
release=$workspace/IntrMotiv/source_cue_audit_v1152_20261104
runtime=$workspace/IntrMotiv/SF_hipposlam/runtime
train_root=$workspace/IntrMotiv/SF_hipposlam/train_dir
batch_root=$train_root/cue_transfer_expanded_qualification_20261104
qualification_root=$train_root/analysis/cue_transfer_expanded_20261104/qualification
attempt_dir=$qualification_root/job_${SLURM_JOB_ID:?Slurm compute node required}
python=/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python
export INTRMOTIV_WORKSPACE_ROOT=$workspace INTRMOTIV_RUNTIME_ROOT=$runtime
export INTRMOTIV_TRAIN_ROOT=$train_root INTRMOTIV_RUNTIME_SOURCE=$release
export INTRMOTIV_EVAL_PYTHON=$python SFGIT_PYTHON=$python
export PYTHONPATH=$runtime/torch_cuda_2_9_1:$runtime/controller_terminal_binding_v1:$release
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR=$workspace/tmp/cegate_${SLURM_JOB_ID}
export XDG_CACHE_HOME=$attempt_dir/cache MPLCONFIGDIR=$attempt_dir/matplotlib
export WANDB_DIR=$attempt_dir/wandb WANDB_MODE=disabled
export DMLAB_CACHE_DIR=$attempt_dir/dmlab_cache TORCH_HOME=$runtime/cache/torch
mkdir -p "$qualification_root" "$TMPDIR"
mkdir "$attempt_dir"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$WANDB_DIR" "$DMLAB_CACHE_DIR"
cp "$0" "$attempt_dir/gate_script.sbatch"
cd "$release"
"$python" - "$batch_root" "$attempt_dir" <<'PY'
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import traceback
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.discovery import discover_run_directories

EXPECTED_COMMIT = 'd5fe0218fc012e12f82848f8f55cb5aa5fb1f650'
TRAINING_COMMIT = 'be7fcbab24c22872972696b6b5a38116258acc91'
PRIOR_JOB = '8563876'
EXPECTED_TARGETS = [524288, 1048576]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def verify_manifest(study, directories, manifest):
    with manifest.open(newline='') as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    runs = {(run.condition, str(run.seed)): run for run in study.expand_runs()}
    expected = {(condition, seed, target) for condition, seed in runs for target in EXPECTED_TARGETS}
    observed = [(row['condition'], row['seed'], int(row['target_frames'])) for row in rows]
    assert len(rows) == 46 and len(set(observed)) == 46 and set(observed) == expected, 'incomplete or duplicate 46-row panel'
    ages = {target: set() for target in EXPECTED_TARGETS}
    used = set()
    for row in rows:
        run = runs[(row['condition'], row['seed'])]
        checkpoint = Path(row['checkpoint']).resolve(strict=True)
        run_dir = Path(row['run_dir']).resolve(strict=True)
        assert run_dir == directories[run.name].resolve() and checkpoint.is_relative_to(run_dir), 'manifest run/checkpoint mismatch'
        target, actual = int(row['target_frames']), int(row['checkpoint_frames'])
        assert actual >= target, f'missing completed target {target}: {actual}'
        assert int(checkpoint.stem.rsplit('_', 1)[1]) == actual, 'checkpoint filename/frame mismatch'
        assert (run.name, checkpoint) not in used, 'same checkpoint reused for two targets'
        used.add((run.name, checkpoint))
        ages[target].add(actual)
    assert all(len(values) == 1 for values in ages.values()), f'unmatched actual checkpoint ages: {ages}'
    actual_ages = {str(target): next(iter(values)) for target, values in ages.items()}
    return {'field_rows': len(rows), 'common_age_verified': True,
            'actual_frames_by_requested_target': actual_ages,
            'exact_requested_ages': all(actual_ages[str(target)] == target for target in EXPECTED_TARGETS),
            'manifest': str(manifest), 'manifest_sha256': digest(manifest)}


def verify_certificate(certificate, checkpoint, audit_row, job_id, owner_root):
    assert certificate.get('model_and_buffers_exact') is True and certificate.get('counters_exact') is True
    if certificate.get('protocol') == 'learner-checkpoint-restore-v2':
        assert certificate.get('restore_contract') == 'learner_init_scheduled_lr_v1'
        assert all(certificate.get(key) is True for key in (
            'restoration_verified', 'optimizer_state_exact', 'optimizer_runtime_contract_exact', 'optimizer_lr_reset_verified'))
        assert isinstance(certificate.get('optimizer_lr_reset_applied'), bool)
        raw_equal = certificate['optimizer_lr_before_init'] == certificate['optimizer_lr_after_init']
        assert raw_equal or certificate['optimizer_lr_reset_applied'] is True
        assert certificate.get('optimizer_exact') is raw_equal and certificate.get('exact_restore') is raw_equal
    else:
        assert job_id == PRIOR_JOB and certificate.get('protocol') is None, 'unrecognized certificate protocol'
        assert certificate.get('exact_restore') is True and certificate.get('optimizer_exact') is True
    assert certificate.get('checkpoint_sha256') == digest(checkpoint), 'checkpoint SHA differs from certificate'
    immutable = Path(certificate['checkpoint']).resolve(strict=True)
    assert immutable.is_relative_to(owner_root.resolve()) and immutable.name == checkpoint.name
    assert digest(immutable) == certificate['checkpoint_sha256'], 'immutable reload copy changed'
    assert certificate.get('frames') == audit_row['checkpoint_frames'], 'certificate frame mismatch'
    assert certificate.get('train_step') == int(checkpoint.stem.rsplit('_', 2)[1])
    config = json.loads((Path(audit_row['run_dir']) / 'config.json').read_text())
    assert certificate['run'] == config['experiment'] and certificate['controller'] == config.get('controller_learning', 'ppo')
    assert str(certificate.get('device', '')).startswith('cuda'), 'reload did not use GPU'
    assert str(certificate.get('job_id')) == job_id, 'wrong certificate job'


def verify_bn_cuda_replay(audit_rows, output):
    rows = []
    key = 'encoder.DG_projection.linear.weight'
    for name, row in audit_rows.items():
        if row.get('dg_mode') != 'bn_only':
            continue
        before = torch.load(row['initial_checkpoint'], map_location='cpu', weights_only=False)
        after = torch.load(row['terminal_checkpoint'], map_location='cpu', weights_only=False)
        expected = before['model'][key].cuda().clone()
        with torch.no_grad():
            for _ in range(after['train_step']):
                expected.div_(expected.norm(dim=1, keepdim=True).clamp_min(1e-6))
        actual = after['model'][key]
        rows.append({'run_name': name, 'checkpoint': row['terminal_checkpoint'],
                     'checkpoint_sha256': digest(row['terminal_checkpoint']),
                     'train_step': after['train_step'], 'exact': torch.equal(expected.cpu(), actual),
                     'max_abs_error': float((expected.cpu() - actual).abs().max())})
    report = {'protocol': 'retained-dg-row-normalization-cuda-replay-v1',
              'job_id': os.environ['SLURM_JOB_ID'], 'device': torch.cuda.get_device_name(),
              'torch_version': torch.__version__, 'runs': rows,
              'passed': len(rows) == 2 and all(row['exact'] for row in rows)}
    path = output / 'bn_cuda_replay.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    torch.cuda.empty_cache()
    assert report['passed'], 'exact CUDA row-normalization replay failed'
    return {'bn_cuda_replay': str(path), 'bn_cuda_replay_sha256': digest(path), 'bn_cuda_replay_verified': True}


def main():
    batch, output = map(Path, sys.argv[1:])
    study_path = Path('hpc_runs/studies/cue_transfer_expanded_qualification_20261104.study.json').resolve()
    study = load_study(study_path)
    assert study.study_id == 'cue_transfer_expanded_qualification_20261104' and study.expected_runs == 23
    assert study.telemetry['target_frames'] == EXPECTED_TARGETS and study.seeds == (42,)
    release = json.loads(Path('RELEASE.json').read_text())
    assert release['runtime_commit'] == EXPECTED_COMMIT
    runs = study.expand_runs()
    started = time.monotonic()
    expected_names = {run.name for run in runs}
    summary = {**study.provenance(), 'protocol': 'cue-transfer-expanded-qualification-recovery-v2',
               'job_id': os.environ['SLURM_JOB_ID'], 'runtime_release': release,
               'dependency_training_jobs': [value for value in os.environ.get('INTRMOTIV_QUALIFICATION_JOB_IDS', '').split(':') if value],
               'batch_root': str(batch), 'output_root': str(output),
               'gate_script_sha256': digest(output / 'gate_script.sbatch'),
               'required_frames': EXPECTED_TARGETS[-1], 'required_runs': 23,
               'passed': False, 'stage': 'transfer_audit', 'errors': [], 'reloads': []}
    summary_path = output / 'qualification_summary.json'

    def save():
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')

    def execute(command, log_name):
        print(json.dumps({'command': command, 'log': str(output / log_name)}), flush=True)
        with (output / log_name).open('w') as log:
            return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False).returncode

    save()
    audit_path = output / 'transfer_audit.json'
    summary['transfer_audit_exit_code'] = execute([
        sys.executable, '-m', 'hpc_runs.intrmotiv_study.transfer_audit', str(study_path), str(batch),
        '--required-frames', str(EXPECTED_TARGETS[-1]), '--output', str(audit_path)], 'transfer_audit.log')
    audit_rows = {}
    try:
        audit = json.loads(audit_path.read_text())
        summary['transfer_audit'] = str(audit_path)
        summary['transfer_audit_sha256'] = digest(audit_path)
        summary['audit_global_errors'] = audit.get('errors', [])
        assert audit['study_sha256'] == study.fingerprint, 'audit StudySpec mismatch'
        rows = audit.get('runs', [])
        assert len(rows) == 23 and {row['run_name'] for row in rows} == expected_names, 'audit lacks all 23 runs'
        audit_rows = {row['run_name']: row for row in rows}
        summary['audit_passed_runs'] = sum(row['passed'] for row in rows)
        summary['audit_errors_by_run'] = {row['run_name']: row['errors'] for row in rows if not row['passed']}
        summary['audit_passed'] = audit.get('passed') is True and summary['transfer_audit_exit_code'] == 0
    except Exception:
        summary['audit_passed'] = False
        summary['errors'].append({'stage': 'read_transfer_audit', 'traceback': traceback.format_exc()})
    try:
        directories = discover_run_directories(study, batch)
    except Exception:
        directories = {}
        summary['errors'].append({'stage': 'discover_runs', 'traceback': traceback.format_exc()})
    summary['stage'] = 'checkpoint_reloads'
    save()
    prior_root = output.parent / ('job_' + PRIOR_JOB)
    replacement_path = output.parent / 'gate_replacement_8563876.json'
    replacement = json.loads(replacement_path.read_text())
    assert replacement['prior_job_id'] == PRIOR_JOB and replacement['intentional_replacement'] is True
    assert replacement['accounting_state'].startswith('CANCELLED')
    prior_summary_path = prior_root / 'qualification_summary.json'
    assert digest(prior_summary_path) == replacement['prior_summary_sha256'], 'cancelled ledger changed'
    prior = json.loads(prior_summary_path.read_text())
    assert prior['study_sha256'] == study.fingerprint and prior['job_id'] == PRIOR_JOB
    assert prior['runtime_release']['runtime_commit'] == TRAINING_COMMIT
    assert len({row['run_name'] for row in prior['reloads']}) == len(prior['reloads'])
    assert {row['run_name'] for row in prior['reloads']}.issubset(expected_names)
    previous = {row['run_name']: row for row in prior['reloads']}
    summary.update(prior_gate=str(prior_root), prior_gate_summary_sha256=digest(prior_summary_path),
                   prior_gate_job_id=PRIOR_JOB, replacement_record=str(replacement_path),
                   replacement_record_sha256=digest(replacement_path), reload_concurrency=4,
                   training_runtime_commit=TRAINING_COMMIT)
    saved_prior = output / 'prior_gate'
    saved_prior.mkdir()
    for path in (prior_summary_path, prior_root / 'transfer_audit.json', replacement_path):
        shutil.copy2(path, saved_prior / path.name)
    for path in prior_root.glob('reload_*.log'):
        shutil.copy2(path, saved_prior / path.name)
    try:
        summary.update(verify_bn_cuda_replay(audit_rows, output))
    except Exception:
        summary['bn_cuda_replay_verified'] = False
        summary['errors'].append({'stage': 'bn_cuda_replay', 'traceback': traceback.format_exc()})
    save()

    def qualify_one(run):
        began = time.monotonic()
        row = audit_rows.get(run.name)
        record = {'run_name': run.name, 'verified': False, 'reused': False}
        try:
            assert row is not None and row['passed'] and run.name in directories, 'transfer audit/discovery failed'
            run_dir, checkpoint = directories[run.name].resolve(), Path(row['terminal_checkpoint']).resolve(strict=True)
            assert run_dir == Path(row['run_dir']).resolve() and checkpoint.is_relative_to(run_dir)
            record['checkpoint'] = str(checkpoint)
            old = previous.get(run.name, {})
            if old.get('verified') is True:
                assert old.get('exit_code') == 0 and old.get('checkpoint') == str(checkpoint)
                certificate_path = Path(old['certificate']).resolve(strict=True)
                assert certificate_path.is_relative_to(prior_root.resolve())
                assert digest(certificate_path) == old['certificate_sha256']
                certificate = json.loads(certificate_path.read_text())
                verify_certificate(certificate, checkpoint, row, PRIOR_JOB, prior_root)
                record.update(reused=True, source_job_id=PRIOR_JOB, exit_code=0)
            else:
                destination = output / 'reloads' / run.name
                record['exit_code'] = execute([
                    sys.executable, '-m', 'hpc_runs.intrmotiv_study.checkpoint_reload',
                    '--run-dir', str(run_dir), '--checkpoint', str(checkpoint), '--output-dir', str(destination)],
                    f'reload_{run.name}.log')
                assert record['exit_code'] == 0, 'reload subprocess failed; see per-run log'
                certificate_path = destination / 'certificate.json'
                certificate = json.loads(certificate_path.read_text())
                verify_certificate(certificate, checkpoint, row, summary['job_id'], output)
                record['source_job_id'] = summary['job_id']
            record.update(certificate=str(certificate_path), certificate_sha256=digest(certificate_path), verified=True,
                          raw_optimizer_exact=certificate['optimizer_exact'],
                          certificate_timing_seconds=certificate.get('timing_seconds'))
        except Exception:
            record['traceback'] = traceback.format_exc()
        record['elapsed_seconds'] = time.monotonic() - began
        return record

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(qualify_one, run) for run in runs]
        for future in as_completed(futures):
            record = future.result()
            summary['reloads'].append(record)
            summary['elapsed_seconds'] = time.monotonic() - started
            save()
            print(json.dumps({'completed_run': record['run_name'], 'verified': record['verified'],
                              'reused': record['reused'], 'elapsed_seconds': record['elapsed_seconds']}), flush=True)
    summary['reloads'].sort(key=lambda row: row['run_name'])
    summary['stage'] = 'render_telemetry'
    save()
    summary['telemetry_exit_code'] = execute([
        sys.executable, '-m', 'hpc_runs.intrmotiv_study', 'render-telemetry',
        str(study_path), str(batch), str(output / 'telemetry')], 'render_telemetry.log')
    summary['common_age_verified'] = False
    try:
        assert summary['telemetry_exit_code'] == 0, 'render-telemetry subprocess failed'
        summary.update(verify_manifest(study, directories, output / 'telemetry/analysis_manifest.tsv'))
    except Exception:
        summary['errors'].append({'stage': 'verify_telemetry', 'traceback': traceback.format_exc()})
    summary['verified_reload_runs'] = sum(row['verified'] for row in summary['reloads'])
    summary['passed'] = summary['audit_passed'] and summary['verified_reload_runs'] == 23 and summary['common_age_verified'] and summary.get('bn_cuda_replay_verified') is True and not summary['errors']
    summary['reused_reload_runs'] = sum(row['reused'] for row in summary['reloads'])
    summary['raw_optimizer_exact_runs'] = sum(row.get('raw_optimizer_exact') is True for row in summary['reloads'])
    summary['elapsed_seconds'] = time.monotonic() - started
    summary['stage'] = 'complete'
    save()
    # Export only lightweight evidence, including failures. Immutable checkpoint
    # copies and learner directories stay in this workspace and out of the vault.
    light = output / 'lightweight'
    light.mkdir()
    for path in (summary_path, audit_path, study_path, Path('RELEASE.json'), output / 'gate_script.sbatch',
                 output / 'telemetry/analysis_manifest.tsv', output / 'telemetry/checkpoint_inventory.tsv'):
        if path.is_file():
            shutil.copy2(path, light / path.name)
    shutil.copytree(saved_prior, light / 'prior_gate')
    if (output / 'bn_cuda_replay.json').is_file():
        shutil.copy2(output / 'bn_cuda_replay.json', light / 'bn_cuda_replay.json')
    certificates = light / 'reload_certificates'
    certificates.mkdir()
    for row in summary['reloads']:
        if row.get('certificate'):
            shutil.copy2(row['certificate'], certificates / (row['run_name'] + '.json'))
    with tarfile.open(output / 'lightweight_evidence.tar.gz', 'w:gz') as archive:
        archive.add(light, arcname='lightweight')
    print(json.dumps({'passed': summary['passed'], 'audit_passed_runs': summary.get('audit_passed_runs', 0),
                      'verified_reload_runs': summary['verified_reload_runs'], 'field_rows': summary.get('field_rows', 0),
                      'summary': str(summary_path), 'lightweight_evidence': str(output / 'lightweight_evidence.tar.gz')}, indent=2), flush=True)
    raise SystemExit(0 if summary['passed'] else 1)


if __name__ == '__main__':
    main()
PY
