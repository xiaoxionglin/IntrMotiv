"""Selection rejects unhealthy runs and study revision preserves science."""
import json
from pathlib import Path
from search_neighborhood_throughput import PROFILE_FRAMES, revised_study, score_candidate


def test_revised_study_preserves_non_resource_arguments(tmp_path):
    source=Path('hpc_runs/studies/dg_neighborhood_production.study.json')
    original=json.loads(source.read_text())
    study=revised_study(source,tmp_path/'selected.json','intrmotiv_dg_neighborhood_production_20260915_tuned',32,8,2)
    new=json.loads((tmp_path/'selected.json').read_text())
    old_id=original['study_id']
    expected=[x.replace(old_id,new['study_id']) for x in original['training']['common_args']]
    expected=[('--num_workers=32' if x.startswith('--num_workers=') else '--num_envs_per_worker=8' if x.startswith('--num_envs_per_worker=') else '--num_epochs=2' if x.startswith('--num_epochs=') else x) for x in expected]
    assert new['training']['common_args']==expected
    assert study.expected_runs==12
    assert original['factors']==new['factors']


def test_score_rejects_resource_pressure_and_incomplete_runs(tmp_path):
    rows=[dict(returncode=0,has_traceback=False,deadline_signal_sent=False,wandb_online=True,frames=PROFILE_FRAMES,fps_after_warmup=500) for _ in range(4)]
    (tmp_path/'summary.json').write_text(json.dumps(rows))
    (tmp_path/'resources.jsonl').write_text(json.dumps(dict(available_ram_gib=100,gpus=[dict(free_mib=20000)]))+'\n')
    assert score_candidate(tmp_path)['aggregate_fps']==2000
    assert score_candidate(tmp_path)['eligible']
    rows[0]['deadline_signal_sent']=True
    (tmp_path/'summary.json').write_text(json.dumps(rows))
    assert not score_candidate(tmp_path)['eligible']
    rows[0]['deadline_signal_sent']=False
    (tmp_path/'summary.json').write_text(json.dumps(rows))
    (tmp_path/'resources.jsonl').write_text(json.dumps(dict(available_ram_gib=63,gpus=[dict(free_mib=20000)]))+'\n')
    assert not score_candidate(tmp_path)['eligible']


def test_six_run_score_requires_six_complete_runs(tmp_path):
    rows=[dict(returncode=0,has_traceback=False,deadline_signal_sent=False,wandb_online=True,frames=PROFILE_FRAMES,fps_after_warmup=300) for _ in range(6)]
    (tmp_path/'summary.json').write_text(json.dumps(rows))
    (tmp_path/'resources.jsonl').write_text(json.dumps(dict(available_ram_gib=100,gpus=[dict(free_mib=20000)]))+'\n')
    assert score_candidate(tmp_path,6)['eligible']
    assert score_candidate(tmp_path,6)['aggregate_fps']==1800
    assert not score_candidate(tmp_path,4)['eligible']


def test_resource_stopped_summary_is_recorded_as_ineligible(tmp_path):
    rows=[dict(returncode=2,has_traceback=False,deadline_signal_sent=True,wandb_online=True,frames=0,fps_after_warmup=None) for _ in range(6)]
    (tmp_path/'summary.json').write_text(json.dumps(rows))
    (tmp_path/'resources.jsonl').write_text(json.dumps(dict(available_ram_gib=43,gpus=[dict(free_mib=80000)]))+'\n')
    result=score_candidate(tmp_path,6)
    assert not result['eligible']
    assert result['aggregate_fps']==0
