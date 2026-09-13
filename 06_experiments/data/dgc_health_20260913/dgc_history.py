import json,wandb
api=wandb.Api(timeout=40)
for run in api.runs('xiaoxionglin-bernstein-center-freiburg/SF_IntrMotiv_DGCapacityGoalConditioning'):
 if '00_DGC_' not in run.name: continue
 rows=run.history(samples=400,pandas=False)
 keys=['train/env_steps','intrmotiv/online/window/target_env_steps','intrmotiv/online/trajectory/stationary_step_fraction','intrmotiv/hrl/option_success_fraction','intrmotiv/hrl/goal_condition/action_sensitivity','policy_stats/avg_z_00_openfield_map2_fixed_loc3_fixedlength_noreward_coverage_auc']
 data=[{k:r[k] for k in keys+['_step','_timestamp'] if k in r} for r in rows]
 print(json.dumps({'name':run.name,'history':data}),flush=True)
