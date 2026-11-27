"""Twelve bounded five-minute read-only checks; no submission, retries or source changes."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

base=Path('/tmp/intrmotiv-first-hit-stationary-20261127')
report=Path('/home/xiaoxiong/Desktop/Projects/IntrMotiv/06_experiments/controllers/fixed_goal_first_hit_20261127/runtime/stationary_launch_v3')
ssh=['ssh','-F','/dev/null','-o','IdentityAgent=none','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-i','/home/xiaoxiong/.ssh/id_ed25519_NEMO2_v2','fr_xl1014@nemo2-login.nemo.uni-freiburg.de','python3 -']
stop=base/'stop_supervision_stationary'
previous=report/'progress_latest.json'
if not previous.is_file():
    previous=report/'main_startup.json'
due=datetime.datetime.fromisoformat(json.loads(previous.read_text())['checked_utc']).timestamp()+300
for index in range(12):
    while time.time()<due:
        if stop.exists():print('STOP requested locally',flush=True);raise SystemExit(0)
        time.sleep(min(30,max(0,due-time.time())))
    if stop.exists():break
    try:completed=subprocess.run(ssh,input=(base/'monitor_stationary.py').read_text(),text=True,capture_output=True,timeout=60)
    except Exception as exc:print('STOP transport exception: '+repr(exc),flush=True);raise SystemExit(1)
    (base/'supervision_last_stdout.json').write_text(completed.stdout)
    (base/'supervision_last_stderr.txt').write_text(completed.stderr)
    if completed.returncode:print('STOP transport/remote nonzero; no retry: '+completed.stderr[-2000:],flush=True);raise SystemExit(1)
    try:data=json.loads(completed.stdout)
    except json.JSONDecodeError:print('STOP invalid monitor response; inspect retained output',flush=True);raise SystemExit(1)
    stamp=data['checked_utc'].replace(':','')
    (base/('progress_'+stamp+'.json')).write_text(json.dumps(data,indent=2)+'\n')
    temporary=report/'progress_latest.json.tmp';temporary.write_text(json.dumps(data,indent=2)+'\n');os.replace(temporary,report/'progress_latest.json')
    print(json.dumps(dict(checked_utc=data['checked_utc'],check=index+1,jobs=[dict(id=j['job_id'],state=j['scheduler']['JobState'],update=j.get('fit_progress',{}).get('update'),blocks=len(j.get('block_journal',{}).get('completed_blocks',[]))) for j in data['jobs']],analysis=data.get('analysis_job'))),flush=True)
    bad={'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'}
    if any(j.get('exception_tail') or j['scheduler']['JobState'] in bad for j in data['jobs']) or data.get('analysis_job',{}).get('scheduler',{}).get('JobState') in bad:
        (report/'monitor_failure.json').write_text(json.dumps(data,indent=2)+'\n');print('STOP observed failure; no retries',flush=True);raise SystemExit(1)
    if data.get('analysis_job',{}).get('scheduler',{}).get('JobState')=='COMPLETED':
        print('STOP final analysis completed; retrieve artifacts separately',flush=True);break
    due=time.time()+300
print('Bounded stationary supervision finished',flush=True)
