"""Pure import, plan and cached-control integration checks; no engine creation."""
import hashlib,json,pathlib,sys
from hpc_runs.first_hit_control.reused_plan import load_plan,render_commands
from hpc_runs.first_hit_control.aggregate_evaluation import read_shard
from hpc_runs.first_hit_control.reused_control import certify_retained_starts,install_callback,verify_engine_sources
import torch
source=pathlib.Path(sys.argv[1]);plan_path=source/'hpc_runs/studies/stationary_projected_c05_s99_20261127.analysis.json'
plan,validation=load_plan(plan_path)
assert validation['effective_plan_sha256']=='c50738255608e644209c53c6d9e3b93471a830c2a379d200b6ef86f836c9a532'
import sf_working_directories.IntrMotiv.evaluation.place_fields as fields
import sf_working_directories.IntrMotiv.evaluation.target_control_interventions as control
assert callable(fields.load_policy_env) and callable(control.run_landmark_matched_interventions)
assert pathlib.Path(fields.__file__).resolve().is_relative_to(source)
assert pathlib.Path(control.__file__).resolve().is_relative_to(source)
baseline=read_shard(plan['qualification']['source_evaluation'],allow_preflight=True)
panel=torch.load(baseline.directory/'matched_starts.pt',map_location='cpu',weights_only=True)
blocks,starts,receipt=certify_retained_starts(baseline,panel)
issued=[r for r in receipt if blocks[r['source'],r['repeat']]['issued']]
assert len(issued)==1 and issued[0]['seed']==31003 and issued[0]['prefix_length']==20
canonical=source/'sf_working_directories/IntrMotiv/evaluation/target_control_interventions.py'
checked=verify_engine_sources(baseline,source,install_callback(canonical.read_text()))
print(json.dumps({'effective_plan_sha256':validation['effective_plan_sha256'],'exact_native_engine_files':checked,'retained_start_receipt':receipt,'source_first_imports':True,'native_environment_created':False,'rendered_stages':[s['stage_id'] for s in render_commands(plan,validation)['stages']]},indent=2))
