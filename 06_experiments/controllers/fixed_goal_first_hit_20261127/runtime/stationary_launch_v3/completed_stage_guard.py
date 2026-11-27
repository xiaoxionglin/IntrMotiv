"""Verify completed test/fit prerequisites and exact native effective-config inputs."""
import hashlib,json,pathlib,sys,xml.etree.ElementTree as ET
from hpc_runs.first_hit_control.dmlab_evaluate import verify_source_config
from hpc_runs.first_hit_control.reused_control import read_reused_shard
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
p=pathlib.Path(sys.argv[1]);assert sha(p)==sys.argv[2];m=json.loads(p.read_text())
for path,expected in m['completed_stage_inputs_sha256'].items():assert sha(path)==expected,path
root=pathlib.Path(m['output_root']);suites=list(ET.parse(root/'qualification_tests_r2/results.xml').iter('testsuite'))
assert sum(int(s.attrib.get('tests',0)) for s in suites)==246
assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ['errors','failures','skipped'])
probe=json.loads((root/'qualification_tests_r2/probe.json').read_text());assert probe['source_first_imports'] and not probe['native_environment_created']
fit=json.loads((root/'qualification_fit/metadata.json').read_text());assert fit['fit']['config']['updates']==100 and fit['fit']['config']['horizon']==64
argv=m['canonical_native_stage']['argv'];value=lambda flag:argv[argv.index(flag)+1]
assets=pathlib.Path(value('--runfiles')).resolve(strict=True)
assert str(assets)==fit['collection_evaluation_overrides']['dmlab_runfiles_path']
assert (assets/'provenance.json').is_file()
overrides=dict(dmlab_use_level_cache=False,dmlab_level_cache_path=str(root/'qualification_evaluation/cache'),dmlab_runfiles_path=str(assets),online_spatial_telemetry=False,with_wandb=False,transfer_freeze_dg=True)
config_sha=verify_source_config(pathlib.Path(value('--run-dir'))/'config.json',fit,overrides)
record=dict(completed_test_count=246,prior_job_id='8950995',prior_job_state='FAILED',prior_successful_stages=['pure tests','source/prefix probe','100-update copied-readout fit'],input_files_unchanged=len(m['completed_stage_inputs_sha256']),source_config_sha256=config_sha,original_runfiles_path=str(assets),new_training_updates=0,new_collection_decisions=0)
if len(sys.argv)>3 and sys.argv[3]=='after':
 shard=read_reused_shard(root/'qualification_evaluation',allow_preflight=True)
 rows=[r for r in shard.rows if r['policy']=='first_hit' and r['commanded']]
 assert len(rows)==2 and all(r['issued'] and not r['missing'] for r in rows)
 signature=shard.summary['matched_start_signatures'];assert len(signature)==1 and signature[0]['prefix_length']==20 and signature[0]['seed']==31003
 record.update(native_completed=True,new_candidate_paths=2,source_control_rerun=False,discovery_decisions=0,native_receipt=shard.receipt)
print(json.dumps(record,indent=2))
