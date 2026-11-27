"""Bounded read-only status from exact submitted jobs and canonical stage paths."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

root=Path('/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1')
submission=json.loads((root/'launch_v3/main_submission_verified_dependencies.json').read_text())
bundle=json.loads((root/'launch_v3/main_launch_bundle.json').read_text())
stages={job['logical_job']:job for job in bundle['jobs']}
ids=[job['job_id'] for job in submission['jobs']]
accounting=subprocess.run(['sacct','-j',','.join(ids),'--format=JobID,State,ExitCode,Elapsed,MaxRSS','--parsable2','--noheader'],text=True,capture_output=True)
account={parts[0]:parts for line in accounting.stdout.splitlines() if len(parts:=line.split('|'))>=4}

def status(jid):
    result=subprocess.run(['scontrol','show','job',jid,'-o'],text=True,capture_output=True)
    fields=dict(item.split('=',1) for item in result.stdout.split() if '=' in item)
    if not fields.get('JobState') and jid in account:
        parts=account[jid];fields.update(JobState=parts[1],ExitCode=parts[2],RunTime=parts[3],StateSource='sacct')
    if not fields.get('JobState'):
        fallback=subprocess.run(['sacct','-j',jid,'--format=JobID,State,ExitCode,Elapsed','--parsable2','--noheader'],text=True,capture_output=True)
        for line in fallback.stdout.splitlines():
            parts=line.split('|')
            if parts[0]==jid and len(parts)>=4:
                fields.update(JobState=parts[1],ExitCode=parts[2],RunTime=parts[3],StateSource='sacct');break
    return {key:fields.get(key) for key in ['JobState','Dependency','RunTime','ExitCode','NodeList','StateSource']}

result=dict(schema='intrmotiv/stationary-progress/v1',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            source_release=bundle['source_release'],effective_plan_sha256=bundle['effective_plan_sha256'],
            jobs=[],stages={},partial_journals_are_diagnostic_only=True)
for job in submission['jobs']:
    jid,name=job['job_id'],job['logical_job']
    row=dict(job_id=jid,logical_job=name,scheduler=status(jid))
    for suffix in ['out','err']:
        path=root/'slurm'/f'{name}_{jid}.{suffix}'
        if not path.exists():continue
        text=path.read_text(errors='replace');row[suffix+'_bytes']=path.stat().st_size
        if suffix=='out':
            updates=[]
            for line in text.splitlines():
                if line.startswith('{"update":'):
                    try:updates.append(json.loads(line))
                    except json.JSONDecodeError:pass
            if updates:row['fit_progress']=updates[-1]
            row['engine_initialization_lines']=text.count('Map number:')
        row[suffix+'_tail']=text.splitlines()[-2:]
        if 'Traceback (most recent call last)' in text:
            row['exception_tail']=text[text.rindex('Traceback (most recent call last)'):][-2500:]
    result['jobs'].append(row)
    argv=stages[name]['argv'];destination=Path(argv[argv.index('--output')+1])
    if (destination/'block_journal.json').is_file():
        journal=json.loads((destination/'block_journal.json').read_text())
        row['block_journal']={key:journal[key] for key in ['status','aggregatable','completed_blocks','expected_blocks','rows']}
    meta=destination/('metadata.json' if name=='main_fit' else 'summary.json')
    if not meta.is_file():continue
    try:value=json.loads(meta.read_text())
    except json.JSONDecodeError:
        result['stages'][name]=dict(path=str(meta),status='present_but_incomplete_json');continue
    stage=dict(path=str(meta),sha256=hashlib.sha256(meta.read_bytes()).hexdigest(),schema=value['schema'])
    if name=='main_fit':
        stage.update(checkpoint_sha256=value['checkpoint_sha256'],readout_sha256=value['fit']['final_readout_sha256'],
                     fit_config=value['fit']['config'],seconds=value['fit']['seconds'],
                     source_audit={key:value['source_audit'][key] for key in ['archive_valid','current_sources_unchanged']})
    else:
        stage.update(decisions=value['decisions'],source_panel=value['source_panel'],source_control_rerun=value['source_control_rerun'],
                     discovery_decisions=value['discovery_decisions'],trials_sha256=value['trials_sha256'],
                     frozen_checks={key:value[key] for key in ['policies_frozen','graph_frozen','readout_frozen','action_head_frozen','checkpoint_unchanged','source_configuration_unchanged','fitted_checkpoint_unchanged']})
    result['stages'][name]=stage
analysis=root/'analysis_launch_v1/analysis_submission.json'
if analysis.is_file():
    jid=json.loads(analysis.read_text())['job_id'];result['analysis_job']=dict(job_id=jid,scheduler=status(jid))
print(json.dumps(result,indent=2))
