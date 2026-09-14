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


def publish_evaluation(report, manifest, report_path=None):
    import wandb
    run=wandb.init(project='SF_IntrMotiv_DGNeighborhood',entity='xiaoxionglin-bernstein-center-freiburg',
                   group=manifest['study_id'],job_type='heldout_evaluation',
                   name=manifest['study_id']+'_qualification',dir='/scratch/lin/IntrMotiv/logs/wandb',
                   config={k:manifest[k] for k in ('study_sha256','source_sha256','manifest_sha256')},mode='online')
    run.summary['evaluation_completed']=True
    if report.get('passed') is True: run.summary['qualification_passed']=True
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
    if report_path is not None:
        artifact=wandb.Artifact(manifest['study_id']+'-evaluation',type='evaluation')
        artifact.add_file(str(report_path))
        run.log_artifact(artifact)
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
    from hpc_runs.intrmotiv_study.telemetry import CheckpointRecord, build_place_field_manifests, write_manifest
    from sf_working_directories.IntrMotiv.evaluation.build_place_field_sweep import select_checkpoints, checkpoint_frames
    preflight_study=load_study(manifest['study_spec'])
    if preflight_study.fingerprint!=manifest['study_sha256']: raise RuntimeError('Preflight StudySpec changed')
    preflight_inventory=[]
    for spec in manifest['runs']:
        run=a.preflight/spec['name']
        checkpoint=max(run.glob('checkpoint_p0/checkpoint_*.pth'),key=lambda p:int(p.stem.split('_')[-1]))
        evaluate(run,checkpoint,a.panel,evaluation_root/spec['name'])
        preflight_inventory.append(CheckpointRecord(spec['name'],spec['target_frames'],checkpoint_frames(checkpoint),checkpoint,run))
        earlier=[t for t in preflight_study.telemetry['target_frames'] if t<spec['target_frames']]
        for target,milestone in select_checkpoints(run,target_frames=earlier):
            evaluate(run,milestone,a.panel,evaluation_root/(spec['name']+'__target_'+str(target)))
            preflight_inventory.append(CheckpointRecord(spec['name'],target,checkpoint_frames(milestone),milestone,run))
    preflight_rows,_=build_place_field_manifests(preflight_study,preflight_inventory)
    write_manifest(a.output/'preflight_evaluation_manifest.tsv',preflight_rows)
    report=qualify(a.preflight,a.panel,evaluation_root)
    url=publish_evaluation(report,manifest,evaluation_root/'qualification.json')
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
    # Use the canonical manifest selector: all declared checkpoints for seed99,
    # terminal checkpoints for replication seeds. No condition-name parsing.
    from hpc_runs.intrmotiv_study.telemetry import discover_nemo_checkpoints, build_place_field_manifests, write_manifest
    inventory=discover_nemo_checkpoints(study,Path(reviewed['output_root']))
    rows,_=build_place_field_manifests(study,inventory)
    write_manifest(a.output/'production_evaluation_manifest.tsv',rows)
    production_results=[]
    for row in rows:
        destination=a.output/'production_evaluations'/row['label_suffix']
        evaluate(Path(row['run_dir']),Path(row['checkpoint']),a.panel,destination)
        summary=json.loads((destination/'summary.json').read_text())
        production_results.append(dict(name=row['label_suffix'],checkpoint=row['checkpoint'],
                                       target_frames=int(row['target_frames']),heldout=summary['heldout']))
    production_report=dict(panel_sha256=report['panel_sha256'],study_sha256=reviewed['study_sha256'],
                           source_sha256=reviewed['source_sha256'],runs=production_results)
    atomic_json(a.output/'production_evaluation_summary.json',production_report)
    production_url=publish_evaluation(production_report,reviewed,a.output/'production_evaluation_summary.json')
    atomic_json(a.output/'status.json',dict(stage='production_and_declared_evaluation_completed',
                evaluation_wandb_url=url,production_evaluation_wandb_url=production_url))


if __name__=='__main__':
    main()
