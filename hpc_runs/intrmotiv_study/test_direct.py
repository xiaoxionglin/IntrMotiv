from pathlib import Path
import pytest
from hpc_runs.intrmotiv_study.direct import log_status, source_digest, atomic_json


def test_source_digest_detects_source_not_outputs(tmp_path):
    source = tmp_path / 'sample_factory'; source.mkdir()
    (source/'train.py').write_text('a=1')
    first = source_digest(tmp_path)
    (source/'events.log').write_text('changing log')
    assert first == source_digest(tmp_path)
    (source/'train.py').write_text('a=2')
    assert first != source_digest(tmp_path)


def test_log_failure_even_with_progress(tmp_path):
    path = tmp_path/'train.log'
    path.write_text('Total num frames: 65536\nhttps://wandb.ai/entity/project/runs/abc123\nTraceback (most recent call last)')
    result = log_status(path)
    assert result['frames'] == 65536 and result['traceback']
    assert len(result['wandb_urls']) == 1


def test_atomic_json_replaces(tmp_path):
    path=tmp_path/'state.json'
    atomic_json(path, {'state':'pending'})
    atomic_json(path, {'state':'done'})
    assert 'done' in path.read_text()
    assert not list(tmp_path.glob('*.tmp'))
