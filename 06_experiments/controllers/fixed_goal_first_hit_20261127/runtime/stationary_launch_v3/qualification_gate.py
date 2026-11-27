"""Composite gate: prior failed job has passed tests/fit; separate native job must complete."""
import json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);here=pathlib.Path(__file__).resolve().parent
checks=root/'qualification_native_r3'
before=json.loads((checks/'source_before.json').read_text());after=json.loads((checks/'source_after.json').read_text())
assert before==after and after['source_files_unchanged']==1437
completed=json.loads((checks/'completion.json').read_text())
assert completed['native_completed'] and completed['new_candidate_paths']==2 and completed['completed_test_count']==246
assert completed['prior_job_id']=='8950995' and completed['prior_job_state']=='FAILED'
result=subprocess.run([sys.executable,str(here/'completed_stage_guard.py'),str(here/'launch_manifest.json'),'2e91bdf82ab9976cdb4e38bf2f9d9970b1a398e06f10aa4a33af9ad945f798b9','after'],text=True,capture_output=True,check=True)
current=json.loads(result.stdout);assert current==completed
print(json.dumps(dict(qualified=True,scope='plumbing only; no efficacy decision',prior_stage_job_id='8950995',prior_stage_job_state='FAILED',native_continuation_job_id='8951028',prior_tests=246,new_candidate_paths=2,retained_prefix_length=20,source_checks=after,native_receipt=current['native_receipt']),indent=2))
