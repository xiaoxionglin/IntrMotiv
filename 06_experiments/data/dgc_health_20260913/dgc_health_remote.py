from pathlib import Path
import json,re,time
from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study import spatial
from hpc_runs.intrmotiv_study.cli import command_collect_spatial
from types import SimpleNamespace
spec=Path('hpc_runs/studies/dg_capacity_goal_conditioning.study.json')
study=load_study(spec)
runs=study.expand_runs()
root=Path('/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir')
logs=root/'_slurm/intrmotiv_dg_capacity_goal_conditioning_20260910/20260910T182743Z/logs'
health=[]
for run in runs:
    files=list(logs.glob('00_'+run.name+'-slurm-*.err'))
    if not files: continue
    p=max(files,key=lambda p:p.stat().st_mtime)
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-200000)); tail=f.read().decode(errors='replace')
    lines=[re.sub(r'\x1b\[[0-9;]*m','',s) for s in tail.splitlines() if 'Fps is' in s]
    health.append({'run_name':run.name,'log':str(p),'age_sec':time.time()-p.stat().st_mtime,'first':lines[0] if lines else None,'latest':lines[-1] if lines else None,'error_tail':[s for s in tail.splitlines() if 'Traceback' in s or 'ERROR' in s][-3:]})
print(json.dumps({'health':health}),flush=True)
# The deployed collector incorrectly restricts snapshots to historical <=100M
# targets. Override target discovery only, using the exact training declaration.
raw=json.loads(spec.read_text())
args=raw['training']['common_args']
target_arg=next(a for a in args if a.startswith('--online_spatial_snapshot_targets='))
targets=tuple(int(x) for x in target_arg.split('=',1)[1].split(','))
spatial.expected_spatial_targets=lambda s: targets
out=root/'analysis/dgc_health_20260913/spatial'
command_collect_spatial(SimpleNamespace(study=spec,output_dir=out,snapshot_root=root/'analysis/online_spatial/intrmotiv_dg_capacity_goal_conditioning_20260910',plot_run=[],plot_target=[],require_complete=False,include_details=False))
print(json.dumps({'collector_target_override':targets,'reason':'historical <=100M collector validation; targets from training common_args'}))
