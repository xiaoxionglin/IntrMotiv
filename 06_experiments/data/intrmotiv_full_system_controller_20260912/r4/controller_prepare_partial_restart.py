import csv,json,shlex,sys,subprocess
from pathlib import Path
from sf_working_directories.IntrMotiv.launcher.resume_slurm_submission import write_manifest
workdir=Path(sys.argv[1]);oldpath=Path(sys.argv[2])
old={r['experiment']:r for r in csv.DictReader(oldpath.open(),delimiter='\t')}
rows=list(csv.DictReader((workdir/'jobs.tsv').open(),delimiter='\t'));preserved=[]
for i,row in enumerate(rows):
 prior=old[row['experiment']]
 if '--controller_learning=ppo' in shlex.split(row['command']):
  rows[i]=prior;preserved.append(row['experiment']);continue
 status=subprocess.check_output(['sacct','-j',prior['job_id'],'-X','-n','-o','State'],text=True).strip().split()[0]
 if status not in ('FAILED','COMPLETED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'):
  raise RuntimeError('Cannot restart active job '+prior['job_id']+': '+status)
 row['job_id']='';row['status']='pending_submission'
submission=json.loads((workdir/'submission.json').read_text())
submission['resume_from_jobs_tsv']=str(oldpath);submission['preserved_existing_experiments']=preserved
write_manifest(workdir,rows,submission)
print(json.dumps(dict(preserved=preserved,pending=[r['experiment'] for r in rows if r['status']=='pending_submission'])))
