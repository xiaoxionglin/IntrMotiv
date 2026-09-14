"""Authorized preflight-to-production transition for the four-arm DG study.

Wait for normal completion; evaluate all terminal models on the shared panel;
run exact checkpoint gates; select 2 or 4 GPU slots from measured throughput;
print and persist the resulting canonical manifest; launch the 12 production
runs. Any failure stops the transition. The desktop heartbeat handles recovery.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from hpc_runs.intrmotiv_study.direct import atomic_json, make_manifest, run_queue
from hpc_runs.intrmotiv_study.spec import load_study
from hpc_runs.hosts.g500.profile_training import resources
from evaluate_neighborhood import evaluate
from gate_neighborhood import qualify


def concurrency_from_samples(path):
    """Compare four-run aggregate speed to the completed two-GPU baseline."""
    samples=[json.loads(line) for line in path.read_text().splitlines()]
    # The direct resource file records hardware; progress is reconstructed from
    # timestamped SF logs so late process shutdown does not dominate throughput.
    import re
    from datetime import datetime
    speed=[]
    for log in sorted(path.parent.glob('DGN_*.log')):
        counters=[]
        for line in log.read_text(errors='replace').splitlines():
            match=re.search(r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+)\].*Total num frames: (\d+)',line)
            if match:
                timestamp=datetime.strptime(match[1],'%Y-%m-%d %H:%M:%S,%f').timestamp()
                frames=int(match[2])
                if frames>0 and (not counters or frames>counters[-1][1]): counters.append((timestamp,frames))
        if len(counters)<3: raise RuntimeError('Insufficient completed updates for concurrency decision')
        speed.append((counters[-1][1]-counters[0][1])/(counters[-1][0]-counters[0][0]))
    if len(speed)!=4: raise RuntimeError('Expected four measured preflights')
    aggregate=sum(speed)
    reference=2*280.1182  # Completed two-GPU baseline, conservative slower arm.
    minimum_ram=min(s['available_ram_gib'] for s in samples)
    minimum_gpu=min(g['free_mib'] for s in samples for g in s['gpus'])
    four=aggregate>=1.4*reference and minimum_ram>=64 and minimum_gpu>=16384
    return ([0,1,0,1] if four else [0,1]), dict(per_run_fps=speed,aggregate_fps=aggregate,
          two_run_reference_fps=reference,required_scaling_efficiency=0.7,
          minimum_available_ram_gib=minimum_ram,minimum_gpu_free_mib=minimum_gpu,
          limitation='Mixed-objective preflights versus existing-objective resource baseline; practical admission decision, not a global optimum.')


def publish_evaluation(report, manifest):
    import wandb
    run=wandb.init(project='SF_IntrMotiv_DGNeighborhood',entity='xiaoxionglin-bernstein-center-freiburg',
                   group=manifest['study_id'],job_type='heldout_evaluation',
                   name=manifest['study_id']+'_qualification',dir='/scratch/lin/IntrMotiv/logs/wandb',
                   config={k:manifest[k] for k in ('study_sha256','source_sha256','manifest_sha256')},mode='online')
    run.summary['qualification_passed']=True
    run.summary['panel_sha256']=report['panel_sha256']
    for item in report['runs']:
        for key in ('population_active_fraction','silent_units','active_only_map_cosine','peak_diversity'):
            value=item['heldout'][key]
            if value is not None: run.summary[item['name']+'/'+key]=value
        units=item['heldout']['per_unit']
        table=wandb.Table(columns=['unit','rms_radius','positive_recall','false_positive_rate','active_fraction','spatial_information'])
        for j in range(len(units['active_fraction'])):
            table.add_data(j,*(units[k][j] for k in ('spatial_rms_radius','positive_recall','false_positive_rate','active_fraction','spatial_information')))
        run.log({item['name']+'/heldout_units':table})
    url=run.url
    run.finish()
    return url


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight',type=Path,required=True)
    p.add_argument('--production-study',type=Path,required=True)
    p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    audit=a.preflight/'direct_execution'
    manifest=json.loads((audit/'manifest.json').read_text())
    atomic_json(a.output/'status.json',dict(stage='waiting_for_preflight',preflight_manifest_sha256=manifest['manifest_sha256']))
    while True:
        states=json.loads((audit/'state.json').read_text())
        if any(s['status'] in ('failed','blocked_by_failure') for s in states):
            raise RuntimeError('Preflight failed; production is blocked')
        if all(s['status']=='completed' for s in states): break
        time.sleep(30)
    if not (a.panel/'metadata.json').is_file(): raise RuntimeError('Shared feature panel is incomplete')
    atomic_json(a.output/'status.json',dict(stage='evaluating_preflights'))
    evaluation_root=a.output/'evaluations'
    for spec in manifest['runs']:
        run=a.preflight/spec['name']
        checkpoint=max(run.glob('checkpoint_p0/checkpoint_*.pth'),key=lambda p:int(p.stem.split('_')[-1]))
        evaluate(run,checkpoint,a.panel,evaluation_root/spec['name'])
    report=qualify(a.preflight,a.panel,evaluation_root)
    url=publish_evaluation(report,manifest)
    slots,evidence=concurrency_from_samples(audit/'resources.jsonl')
    study=load_study(a.production_study)
    if study.expected_runs!=12 or sorted(study.seeds)!=[8,99,123]: raise RuntimeError('Unexpected production matrix')
    reviewed=make_manifest(study,Path(manifest['source_root']),slots)
    if reviewed['source_sha256']!=manifest['source_sha256']: raise RuntimeError('Production source differs from qualified preflight')
    if any(r['target_frames']!=10000000 for r in reviewed['runs']): raise RuntimeError('Unexpected production frame budget')
    atomic_json(a.output/'resource_decision.json',dict(gpu_slots=slots,**evidence))
    atomic_json(a.output/'production_review.json',reviewed)
    print(json.dumps(reviewed,indent=2),flush=True)
    atomic_json(a.output/'status.json',dict(stage='production_running',evaluation_wandb_url=url,
                production_manifest_sha256=reviewed['manifest_sha256'],gpu_slots=slots))
    run_queue(reviewed,resources)
    atomic_json(a.output/'status.json',dict(stage='production_finished_evaluation_pending',evaluation_wandb_url=url))
    # The same shared panel is reused after production. Checkpoint discovery
    # follows declared identities and records the actual terminal frame count.
    for spec in reviewed['runs']:
        run=Path(reviewed['output_root'])/spec['name']
        checkpoint=max(run.glob('checkpoint_p0/checkpoint_*.pth'),key=lambda p:int(p.stem.split('_')[-1]))
        evaluate(run,checkpoint,a.panel,a.output/'production_evaluations'/spec['name'])
    atomic_json(a.output/'status.json',dict(stage='production_and_terminal_evaluation_completed',evaluation_wandb_url=url))


if __name__=='__main__':
    main()
