"""Read-only NumPy reconstruction of the pinned original factual sampler.

Run on NEMO with the existing NumPy environment. No model, tensor library,
environment, refit, or remote writes occur. The original fit did not log stream
hashes: its values below are reconstructed, while the stationary values were
recorded by the completed fit. Usable goal-label masks intentionally differ.
"""

import ast
import base64
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np


WORK = Path('/work/classic/fr_xl1014-corridor-geometry/IntrMotiv')
ANALYSIS = WORK / 'SF_hipposlam/train_dir/analysis/first_hit_control_20261127'
MAIN = ANALYSIS / 'main_r1'
STATIONARY = ANALYSIS / 'stationary_projected_r1'
ORIGINAL_SOURCE = WORK / 'source_first_hit_control_20261127_r3'
STATIONARY_SOURCE = WORK / 'source_first_hit_stationary_20261127_r1'
RELATIVE_SAMPLER = Path('hpc_runs/first_hit_control/replay_fit.py')
ORIGINAL_SHA = '7e47c74830642e22fffeb00f1c10ccb4b2035cfb3bb29afbeb2f6641859f26ab'
STATIONARY_SHA = '18c80fe74a75f42c4ecb75208774975d2afa93b504093e2c044593d9e674505e'
DATASET_SHA = '765738a4a98a6f81a7cdef84208f2a27991a9202051ed36966382cfc134e2b73'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def array_digest(array):
    return hashlib.sha256(np.asarray(array, dtype='<i8').tobytes()).hexdigest()


def main():
    started = time.monotonic()
    old_path = ORIGINAL_SOURCE / RELATIVE_SAMPLER
    new_path = STATIONARY_SOURCE / RELATIVE_SAMPLER
    assert digest(old_path) == ORIGINAL_SHA
    assert digest(new_path) == STATIONARY_SHA
    old = json.loads((MAIN / 'fit/metadata.json').read_text())
    new = json.loads((STATIONARY / 'fit/metadata.json').read_text())
    config = SimpleNamespace(**old['fit']['config'])
    assert old['fit']['config'] == new['fit']['config']
    assert (config.seed, config.updates, config.batch_size, config.horizon,
            config.holdout_fraction) == (99, 30000, 128, 256, 0.1)
    assert 'batch_provenance' not in old['fit']
    data_path = MAIN / 'replay/replay.npz'
    assert data_path.is_file()
    actual_dataset_sha = digest(data_path)
    assert actual_dataset_sha == DATASET_SHA
    assert new['input_files'][str(data_path)] == DATASET_SHA
    assert old['input_files'][str(data_path)] == DATASET_SHA
    with np.load(data_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ('episode_id', 'successor_valid')}

    # Compile only the original split function and the exact four NumPy
    # sampling statements. Nothing else in the training module is imported.
    tree = ast.parse(old_path.read_text())
    split = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == 'episode_split')
    fit = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == 'fit_replay')
    init_strings = {
        '(train, holdout) = episode_split(data[\'episode_id\'], config.holdout_fraction, config.seed)',
        "train_usable = train[data['successor_valid'][train]]",
        'rng = np.random.default_rng(config.seed)',
    }
    init = [node for node in fit.body if isinstance(node, ast.Assign)
            and ast.unparse(node) in init_strings]
    assert {ast.unparse(node) for node in init} == init_strings
    loop = next(node for node in fit.body if isinstance(node, ast.For)
                and ast.unparse(node.target) == 'update')
    draws = loop.body[:2]
    assert [ast.unparse(node) for node in draws] == [
        'indices = rng.choice(train_usable, config.batch_size, replace=True)',
        'remaining = rng.integers(1, config.horizon + 1, config.batch_size)',
    ]
    namespace = {'np': np, 'config': config, 'data': data}
    exec(compile(ast.Module(body=[split, *init], type_ignores=[]), str(old_path), 'exec'), namespace)
    draw_code = compile(ast.Module(body=draws, type_ignores=[]), str(old_path), 'exec')
    row_hash, deadline_hash = hashlib.sha256(), hashlib.sha256()
    boundary = {}
    for update in range(1, config.updates + 1):
        exec(draw_code, namespace)
        rows, deadlines = namespace['indices'], namespace['remaining']
        row_hash.update(np.asarray(rows, dtype='<i8').tobytes())
        deadline_hash.update(np.asarray(deadlines, dtype='<i8').tobytes())
        if update in (1, config.updates):
            boundary[str(update)] = {'first_eight_rows': rows[:8].tolist(),
                                     'first_eight_deadlines': deadlines[:8].tolist()}
    reconstructed = {
        'factual_indices_sha256': row_hash.hexdigest(),
        'consumed_deadlines_sha256': deadline_hash.hexdigest(),
        'train_indices_sha256': array_digest(namespace['train']),
        'holdout_indices_sha256': array_digest(namespace['holdout']),
    }
    observed = {key: new['fit']['batch_provenance'][key] for key in reconstructed}
    matches = {key: reconstructed[key] == observed[key] for key in reconstructed}
    assert all(matches.values())
    assert len(namespace['train']) == old['fit']['training_transitions'] == new['fit']['training_transitions']
    assert len(namespace['holdout']) == old['fit']['holdout_transitions'] == new['fit']['holdout_transitions']
    assert digest(old_path) == ORIGINAL_SHA and digest(new_path) == STATIONARY_SHA
    assert new['source_audit']['archive_valid'] and new['source_audit']['current_sources_unchanged']

    relative_files = [
        'fit/metadata.json', 'fit/protocol.json', 'fit/initialization.json',
        'fit/source_snapshot/manifest.json',
        'main_job_checks/main_fit_source_before.json',
        'main_job_checks/main_fit_source_after.json',
        'main_job_checks/main_fit_qualification_gate.json',
        'slurm/main_fit_8951059.out', 'slurm/main_fit_8951059.err',
    ]
    files = []
    for relative in relative_files:
        path = STATIONARY / relative
        payload = path.read_bytes()
        assert len(payload) < 2_000_000
        files.append({'relative_path': relative, 'remote_path': str(path),
                      'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload),
                      'base64': base64.b64encode(payload).decode('ascii')})
    command = ['sacct', '-j', '8951059', '--format=JobID,State,ExitCode,Elapsed,MaxRSS',
               '--parsable2', '--noheader']
    accounting = subprocess.run(command, text=True, capture_output=True, check=True)
    rows = [line.split('|') for line in accounting.stdout.splitlines()]
    job = next(row for row in rows if row[0] == '8951059')
    assert job[1:3] == ['COMPLETED', '0:0']
    audit = {
        'schema': 'intrmotiv/reconstructed-sampler-audit/v1',
        'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'original_hash_status': 'reconstructed after completion; not observed during original fitting',
        'stationary_hash_status': 'observed by the actual completed stationary fit',
        'original_source_commit': '10cd0e574cb3943e244d8f5847aef7aecff87dfa',
        'stationary_source_commit': 'bb487bff2d1683bceae8173ad5d340b42b39adcb',
        'original_sampler_path': str(old_path), 'original_sampler_sha256': ORIGINAL_SHA,
        'stationary_sampler_path': str(new_path), 'stationary_sampler_sha256': STATIONARY_SHA,
        'dataset_path': str(data_path), 'dataset_sha256': actual_dataset_sha,
        'loaded_members_only': list(data), 'numpy_version': np.__version__,
        'original_fit_metadata_sha256': digest(MAIN / 'fit/metadata.json'),
        'stationary_fit_metadata_sha256': digest(STATIONARY / 'fit/metadata.json'),
        'config': vars(config), 'total_factual_draws': config.updates * config.batch_size,
        'train_rows': len(namespace['train']), 'holdout_rows': len(namespace['holdout']),
        'certified_training_rows': len(namespace['train_usable']),
        'original_reconstructed': reconstructed, 'stationary_observed': observed,
        'all_four_hashes_match': all(matches.values()), 'hash_comparisons': matches,
        'boundary_samples': boundary,
        'index_encoding': 'C-order little-endian int64; concatenated update batches',
        'usable_masks_compared': False,
        'usable_masks_note': 'The stationary objective masks the current winner; goal-label masks are intentionally different.',
        'scope': 'NumPy sampling only; no model, environment, refit, or native execution; no efficacy claim.',
        'source_audit': {key: new['source_audit'][key] for key in ('archive_valid', 'current_sources_unchanged')},
        'accounting': {'command': command, 'stdout': accounting.stdout, 'stderr': accounting.stderr,
                       'returncode': accounting.returncode},
        'seconds': time.monotonic() - started,
    }
    print(json.dumps({'audit': audit, 'files': files}, allow_nan=False))


if __name__ == '__main__':
    main()
