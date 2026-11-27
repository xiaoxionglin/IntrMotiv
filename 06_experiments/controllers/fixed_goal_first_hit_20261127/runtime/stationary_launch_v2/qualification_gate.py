"""Validate the completed plumbing qualification without retraining or native execution."""
import json,pathlib,sys,xml.etree.ElementTree as ET
from hpc_runs.first_hit_control.aggregate_evaluation import read_shard
output=pathlib.Path(sys.argv[1]);tests=output/'qualification_tests_r2'
xml=ET.parse(tests/'results.xml').getroot()
suites=list(xml.iter('testsuite'))
assert suites and sum(int(s.attrib.get('tests',0)) for s in suites)==246
assert all(int(s.attrib.get('errors',0))==0 and int(s.attrib.get('failures',0))==0 and int(s.attrib.get('skipped',0))==0 for s in suites)
before=json.loads((tests/'source_before.json').read_text());after=json.loads((tests/'source_after.json').read_text())
assert before==after and after['source_files_unchanged']==1437
probe=json.loads((tests/'probe.json').read_text());assert probe['source_first_imports'] and not probe['native_environment_created']
shard=read_shard(output/'qualification_evaluation',allow_preflight=True)
summary=shard.summary
assert summary['source_panel']==[2] and summary['eligible_targets']==[0,1] and summary['config']['horizon']==64
assert summary['discovery_decisions']==0 and not summary['source_control_rerun']
candidate=[row for row in shard.rows if row['policy']=='first_hit' and row['commanded']]
assert len(candidate)==2 and all(row['issued'] and not row['missing'] for row in candidate)
signatures=summary['matched_start_signatures']
assert len(signatures)==1 and signatures[0]['seed']==31003 and signatures[0]['prefix_length']==20
fit=json.loads((output/'qualification_fit/metadata.json').read_text())
assert fit['fit']['config']['updates']==100 and fit['fit']['config']['horizon']==64
print(json.dumps({'qualified':True,'scope':'plumbing only; no efficacy decision','candidate_paths':2,'retained_prefix_length':20,'qualification_replay_pooled':False,'qualification_weights_reused':False,'qualification_receipt':shard.receipt,'test_count':sum(int(s.attrib.get('tests',0)) for s in suites)},indent=2))
