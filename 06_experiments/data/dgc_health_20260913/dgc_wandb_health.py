import json
import wandb
from hpc_runs.intrmotiv_study import load_study
from pathlib import Path
study=load_study(Path('hpc_runs/studies/dg_capacity_goal_conditioning.study.json'))
names={r.name for r in study.expand_runs()}
api=wandb.Api(timeout=40)
for r in api.runs('xiaoxionglin-bernstein-center-freiburg/SF_IntrMotiv_DGCapacityGoalConditioning'):
    if not any(n in r.name for n in names): continue
    summary=dict(r.summary)
    keep={k:v for k,v in summary.items() if any(t in k.lower() for t in ['coverage','stationary','entropy','density','silent','sensitivity','success_fraction','env_steps','distance','active_unit','_timestamp'])}
    print(json.dumps({'name':r.name,'id':r.id,'state':r.state,'summary':keep}),flush=True)
