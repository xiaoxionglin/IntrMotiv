"""Pure analysis qualification against existing immutable native artifacts."""
import argparse,json,sys
from pathlib import Path
from hpc_runs.first_hit_control.aggregate_evaluation import file_hash
from hpc_runs.first_hit_control.trace_analysis import load_field,summarize_trace
from hpc_runs.first_hit_control.plot_evaluation import _runtime
p=argparse.ArgumentParser();p.add_argument('--field-audit',required=True);p.add_argument('--field-npz',required=True);p.add_argument('--native-qualification',required=True);p.add_argument('--output',required=True);a=p.parse_args()
output=Path(a.output);output.mkdir()
field=load_field(a.field_audit,a.field_npz,16)
records={}
for panel in ('evaluation','evaluation_nonempty_prefix'):
 directory=Path(a.native_qualification)/panel
 summary=json.loads((directory/'summary.json').read_text())
 path=directory/'trials.jsonl';assert file_hash(path)==summary['trials_sha256']
 rows=[json.loads(line) for line in path.read_text().splitlines()]
 own=[row for row in rows if row['issued'] and row['commanded']]
 descriptions=[summarize_trace(row,16,field)[0] for row in own]
 assert len(descriptions)==4
 records[panel]={'trials_sha256':file_hash(path),'issued_own_paths':len(own),'observed_samples':[r['observed_samples'] for r in descriptions]}
plt,font=_runtime();fig,axis=plt.subplots(figsize=(10,3),dpi=100)
axis.text(.5,.5,'Analysis font qualification',ha='center',va='center',fontsize=30);axis.set_axis_off();fig.savefig(output/'font_probe.png');plt.close(fig)
assert not any(name=='deepmind_lab' or name.startswith('deepmind_lab.') for name in sys.modules)
receipt={'schema':'intrmotiv/pure-analysis-native-artifact-qualification/v1','native_execution':False,'training':False,'field_sha256':file_hash(a.field_npz),'panels':records,'font':{'path':str(font),'sha256':file_hash(font),'minimum_points':30},'font_probe_sha256':file_hash(output/'font_probe.png')}
(output/'qualification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
