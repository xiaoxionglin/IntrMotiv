"""W&B lift-history screen and selected full-history audit; no training mutation."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import wandb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hpc_runs.intrmotiv_study import load_study

OUT = ROOT / '06_experiments/results/late_lift_audit_20260908'
ENTITY = 'xiaoxionglin-bernstein-center-freiburg'
STEP = 'train/env_steps'
LIFT = 'intrmotiv/hrl/target_hit_lift'
METRICS = dict(lift=LIFT, target_rate='intrmotiv/hrl/target_hit_rate',
    target_num='intrmotiv/hrl/target_hit_numerator', target_count='intrmotiv/hrl/target_hit_event_count',
    shuffle_num='intrmotiv/hrl/shuffled_hit_numerator', shuffle_count='intrmotiv/hrl/shuffled_hit_event_count',
    success='intrmotiv/hrl/option_success_fraction', sensitivity='intrmotiv/hrl/goal_condition/action_sensitivity',
    action_tv='intrmotiv/hrl/goal_condition/action_probability_tv', density='intrmotiv/dg/density',
    silence='intrmotiv/dg/silent_unit_fraction', entropy='intrmotiv/dg/usage_entropy',
    edges='intrmotiv/hrl/known_edge_fraction', target_valid='intrmotiv/hrl/active_target_fraction',
    coverage='policy_stats/avg_z_00_openfield_map2_fixed_loc3_fixedlength_noreward_coverage_auc',
    recruitment='intrmotiv/dg/recruitment/total')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--detail', nargs='*'); args=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    api=wandb.Api(timeout=60)
    if args.detail is not None:
        inv=json.loads((OUT/'inventory.json').read_text())
        def detail(rec):
            r=api.run(rec['path']); p=OUT/(r.id+'_full.csv.gz')
            if p.exists(): return r.name, 'cached'
            keys=[v for v in METRICS.values() if v in r.summary]
            # Scan all history rows: requesting all keys would silently require
            # their intersection and could discard asynchronously logged metrics.
            rows=[]
            for row in r.scan_history(page_size=1000):
                kept={k:row[k] for k in [STEP,'_step',*keys] if k in row}
                if STEP in kept and len(kept)>2: rows.append(kept)
            df=pd.DataFrame(rows).rename(columns={v:k for k,v in METRICS.items()})
            df.to_csv(p,index=False)
            return r.name,len(df)
        jobs=[r for r in inv if r['id'] in args.detail]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for f in as_completed([pool.submit(detail,r) for r in jobs]): print(f.result(),flush=True)
        return
    provenance=[]
    for p in (ROOT/'hpc_runs/studies').glob('*.study.json'):
        s=load_study(p); provenance.append(s.provenance())
    inv=[]
    for project in api.projects(ENTITY):
        if not project.name.startswith('SF_IntrMotiv_') or any(x in project.name for x in ['Preflight','Smoke','OnlineSpatial']): continue
        for r in api.runs(f'{ENTITY}/{project.name}',per_page=100):
            if LIFT not in r.summary: continue
            inv.append(dict(id=r.id,name=r.name,path=f'{ENTITY}/{project.name}/{r.id}',project=project.name,
                group=r.group,state=r.state,url=r.url,config=dict(r.config),summary=dict(r.summary)))
        print(project.name,len(inv),flush=True)
    (OUT/'inventory.json').write_text(json.dumps(inv,indent=2,default=str))
    (OUT/'provenance.json').write_text(json.dumps(dict(retrieved_utc=datetime.now(timezone.utc).isoformat(),
        source='W&B sampled lift history, 1000 requested points/run; selected runs verified by full scan',studies=provenance),indent=2))
    def screen(rec):
        p=OUT/(rec['id']+'_screen.csv.gz')
        if p.exists(): d=pd.read_csv(p)
        else:
            r=api.run(rec['path']); d=r.history(keys=[STEP,LIFT],samples=1000,pandas=True)
            d.to_csv(p,index=False)
        if STEP not in d or LIFT not in d or len(d)==0: return dict(id=rec['id'],error='missing history')
        d=d.dropna(subset=[STEP,LIFT]).sort_values(STEP); end=d[STEP].max()
        early=d[(d[STEP]>=.1*end)&(d[STEP]<.5*end)][LIFT]; late=d[d[STEP]>=.5*end][LIFT]
        tail=d[d[STEP]>=.85*end][LIFT]
        return dict(id=rec['id'],name=rec['name'],project=rec['project'],end_m=end/1e6,n=len(d),
            early_median=early.median(),late_median=late.median(),late_mean=late.mean(),late_max=late.max(),
            late_p95=late.quantile(.95),tail_median=tail.median(),late_gt2=(late>2).mean(),url=rec['url'])
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,f in enumerate(as_completed([pool.submit(screen,r) for r in inv])):
            try: results.append(f.result())
            except Exception as e: results.append(dict(error=str(e)))
            if i%25==0: print('screen',i,flush=True)
    pd.DataFrame(results).sort_values('late_max',ascending=False).to_csv(OUT/'screen.csv',index=False)

if __name__=='__main__': main()
