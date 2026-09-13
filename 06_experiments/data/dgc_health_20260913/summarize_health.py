from pathlib import Path
import json,re,csv
root=Path('06_experiments/data/dgc_health_20260913')
health=json.loads((root/'dgc_health_remote.out').read_text().splitlines()[0])['health']
for r in health:
 r['first_frames']=int(re.search(r'Total num frames: (\d+)',r['first'])[1])
 r['latest_frames']=int(re.search(r'Total num frames: (\d+)',r['latest'])[1])
assert len(health)==27 and all(r['latest_frames']>r['first_frames'] for r in health)
(root/'log_health.json').write_text(json.dumps(health,indent=2))
rows=[]
for line in (root/'dgc_wandb_health.jsonl').read_text().splitlines():
 r=json.loads(line);s=r['summary'];row={'run_name':r['name'].split('_2026')[0].removeprefix('00_'),'wandb_id':r['id'],'frames':s['train/env_steps']}
 for label,key in {'stationary':'intrmotiv/online/trajectory/stationary_step_fraction','movement':'intrmotiv/online/trajectory/mean_physical_step_distance','option_success':'intrmotiv/hrl/option_success_fraction','readout_logit_sensitivity':'intrmotiv/hrl/goal_condition/action_sensitivity','silent_fraction':'intrmotiv/dg/silent_unit_fraction','coverage_auc':'policy_stats/avg_z_00_openfield_map2_fixed_loc3_fixedlength_noreward_coverage_auc'}.items(): row[label]=s.get(key)
 rows.append(row)
with (root/'latest_health.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print('All 27 frame counters advanced across bounded tails; latest summaries saved.')
