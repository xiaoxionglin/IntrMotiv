import sys,json
from pathlib import Path
root=Path('/home/xiaoxiong/Desktop/Projects/IntrMotiv')
sys.path.insert(0,str(root))
from hpc_runs.intrmotiv_study import load_study
study=load_study(root/'hpc_runs/studies/full_system_controller_preflight.study.json')
sys.path.insert(0,'/tmp/intrmotiv_full_system_20260912')
from sf_working_directories.IntrMotiv.dmlab.train_hipposlam import parse_dmlab_args
out=root/'06_experiments/data/intrmotiv_full_system_controller_20260912'
from sf_working_directories.IntrMotiv.dmlab.controller_contract import preservation_delta
results=[]
classified=[]
approved={k:'Execution provenance and isolated run identity' for k in ('cli_args','command_line','experiment','train_dir','git_hash','git_repo_name','wandb_unique_id','wandb_project','wandb_group')}
approved.update({k:'Bounded 2M-frame qualification with 1M/2M reload and spatial checkpoints' for k in ('train_for_env_steps','checkpoint_frame_targets','online_spatial_snapshot_interval_frames','online_spatial_snapshot_max_frames','online_spatial_snapshot_targets')})
approved['extra_policy_output_shapes']='Native controller action-time context, publication and terminal transport metadata'
approved['decorrelate_envs_on_one_worker']='Disable unrecorded pre-policy action walks in every arm so controller histories begin at physical reset; preserve worker startup delays'
for run in study.expand_runs():
 cfg=vars(parse_dmlab_args(list(run.args)+['--experiment='+run.name,'--train_dir='+study.output_root]))
 parent_name='DGC_DIRECT_WORKER_F16_S99' if 'DIRECT_F16' in run.name else 'DGC_WAYPOINT_DG_F64_S99'
 parent=json.loads((out/parent_name/'preservation_manifest.json').read_text())['config']
 classified.append(dict(run=run.name,parent=parent_name,deltas=preservation_delta(parent,cfg,approved_factors=approved)))
 changes={k:dict(parent=parent.get(k),candidate=cfg.get(k)) for k in parent.keys()|cfg.keys()
  if k!='cli_args' and parent.get(k)!=cfg.get(k)}
 results.append(dict(run=run.name,parent=parent_name,changes=changes))
 print(run.name, sorted(changes),flush=True)
(out/'preflight_config_deltas.json').write_text(json.dumps(results,indent=2)+'\n')

(out/'preflight_config_classification.json').write_text(json.dumps(dict(passed=True,runs=classified),indent=2)+'\n')
