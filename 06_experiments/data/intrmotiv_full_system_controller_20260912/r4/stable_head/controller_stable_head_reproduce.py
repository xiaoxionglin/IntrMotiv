import json,sys,os,tempfile,collections
from pathlib import Path
import torch
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.model_sharing import ParameterServer
from sf_working_directories.IntrMotiv.dmlab.train_hipposlam import register_dmlab_components
from sf_working_directories.IntrMotiv.dmlab.controller_learner import ControllerLearner
from sf_working_directories.IntrMotiv.dmlab.controller_transition import ReplayRejected
root=Path('/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir')
batch=root/'intrmotiv_full_system_controller_preflight_20260912_r4'
source=sorted((batch/'FSCP_WAYPOINT_F64_DDQN_HER_S99_'/'00_FSCP_WAYPOINT_F64_DDQN_HER_S99'/'checkpoint_p0').glob('checkpoint_*.pth'))[-1]
record={'baseline':str(source)}
cfg=AttrDict(json.loads((source.parent.parent/'config.json').read_text()));cfg.device='gpu';cfg.cli_args={}
cfg.train_dir=tempfile.mkdtemp(prefix='eligibility_',dir=root/'analysis')
target=Path(cfg.train_dir)/cfg.experiment/'checkpoint_p0';target.mkdir(parents=True);os.link(record['baseline'],target/source.name)
register_dmlab_components();torch.set_num_threads(2)
env=make_env_func_batched(cfg,env_config=None);info=extract_env_info(env,cfg);env.close()
v=torch.zeros(1,dtype=torch.int32);learner=ControllerLearner(cfg,info,v,0,ParameterServer(0,v,False));learner.init()

import numpy as np, collections, time
from sf_working_directories.IntrMotiv.dmlab.controller_snapshot import evaluate_replay
from sf_working_directories.IntrMotiv.dmlab.controller_transition import transition_values_batch
keys=learner.replay.candidate_order();pending=[];counts=collections.Counter();checked=0
for start in range(0,min(len(keys),65536),256):
 group=[]
 for key in keys[start:start+256]:
  try:group.append(learner._example(key))
  except (ReplayRejected,ValueError) as e:counts[str(e)]+=1
 with torch.no_grad():searched=learner._evaluate_pairs(group)
 for example,result in zip(group,searched):
  if isinstance(result,str):counts[result]+=1
  else:pending.append(example)
 while len(pending)>=256:
  examples=pending[:256];del pending[:256]
  full=learner._evaluate_pairs(examples);checked+=len(examples)
  bad=[(e.rows[e.burn_in].key,r) for e,r in zip(examples,full) if isinstance(r,str)]
  if bad:
   print(json.dumps(dict(inconsistencies=bad,checked=checked,counts=counts)),flush=True)
   torch.save(dict(examples=examples,bad=bad),root/'analysis/controller_eligibility_inconsistency_examples.pth')
   raise RuntimeError('Fixed head still has eligibility batch dependence')
  del full
 if start%4096==0:print(json.dumps(dict(searched=start+len(group),checked=checked,counts=counts)),flush=True)
print(json.dumps(dict(recheck_consistent=True,checked=checked,counts=counts)),flush=True)
