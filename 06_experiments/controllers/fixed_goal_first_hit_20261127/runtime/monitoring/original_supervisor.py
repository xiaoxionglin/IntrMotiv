"""Twelve bounded five-minute read-only checks; stop on failure, never retry SSH."""
import datetime,json,os,pathlib,subprocess,time
base=pathlib.Path('/tmp/intrmotiv-first-hit-qualification-20261127')
runtime=pathlib.Path('/home/xiaoxiong/Desktop/Projects/IntrMotiv/06_experiments/controllers/fixed_goal_first_hit_20261127/runtime')
stop=base/'stop_supervision'
ssh=['ssh','-F','/dev/null','-o','IdentityAgent=none','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-i','/home/xiaoxiong/.ssh/id_ed25519_NEMO2_v2','fr_xl1014@nemo2-login.nemo.uni-freiburg.de','python3 -']
previous=json.loads((runtime/'main_progress_latest.json').read_text())
due=datetime.datetime.fromisoformat(previous['checked_utc']).timestamp()+300
for index in range(12):
    while time.time()<due:
        if stop.exists():
            print('STOP requested by local marker',flush=True);raise SystemExit(0)
        time.sleep(min(30,max(0,due-time.time())))
    if stop.exists():break
    try:
        result=subprocess.run(ssh,input=(base/'monitor_main_jobs.py').read_text(),text=True,capture_output=True,timeout=60)
    except Exception as exc:
        print('STOP monitor transport exception: '+repr(exc),flush=True);raise SystemExit(1)
    (base/'supervision_last_stdout.json').write_text(result.stdout)
    (base/'supervision_last_stderr.txt').write_text(result.stderr)
    if result.returncode:
        print('STOP SSH/remote nonzero; no retry: '+result.stderr[-2000:],flush=True);raise SystemExit(1)
    try:data=json.loads(result.stdout)
    except json.JSONDecodeError:
        print('STOP invalid monitor JSON; inspect saved stdout',flush=True);raise SystemExit(1)
    stamp=data['checked_utc'].replace(':','').replace('+0000','Z')
    (base/('progress_'+stamp+'.json')).write_text(json.dumps(data,indent=2)+'\n')
    temporary=runtime/'main_progress_latest.json.tmp';temporary.write_text(json.dumps(data,indent=2)+'\n');os.replace(temporary,runtime/'main_progress_latest.json')
    summary={'checked_utc':data['checked_utc'],'check':index+1,'jobs':[{'id':j['job_id'],'state':j['scheduler']['JobState'],'engines':j.get('engine_initialization_lines'),'resources':j.get('live_resource_statistics')} for j in data['jobs']],'analysis':data.get('analysis_job')}
    print(json.dumps(summary),flush=True)
    failures=[j for j in data['jobs'] if j.get('exception_tail') or j['scheduler']['JobState'] in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL']]
    analysis=data.get('analysis_job',{}).get('scheduler',{})
    if failures or analysis.get('JobState') in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL']:
        (runtime/'main_monitor_failure.json').write_text(json.dumps(data,indent=2)+'\n');print('STOP observed job failure; no submissions or retries',flush=True);raise SystemExit(1)
    if analysis.get('JobState')=='COMPLETED':
        print('STOP final analysis completed; retrieve final artifacts separately',flush=True);break
    due=time.time()+300
print('Bounded supervision finished',flush=True)
