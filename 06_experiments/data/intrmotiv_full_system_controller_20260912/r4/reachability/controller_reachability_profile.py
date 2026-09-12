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
record=json.loads((root/'analysis/restart_baselines/intrmotiv_full_system_controller_preflight_20260912_r4_compatibility/baselines.json').read_text())[5]
source=Path(record['source'])
cfg=AttrDict(json.loads((source.parent.parent/'config.json').read_text()));cfg.device='gpu';cfg.cli_args={}
cfg.train_dir=tempfile.mkdtemp(prefix='eligibility_',dir=root/'analysis')
target=Path(cfg.train_dir)/cfg.experiment/'checkpoint_p0';target.mkdir(parents=True);os.link(record['baseline'],target/source.name)
register_dmlab_components();torch.set_num_threads(2)
env=make_env_func_batched(cfg,env_config=None);info=extract_env_info(env,cfg);env.close()
v=torch.zeros(1,dtype=torch.int32);learner=ControllerLearner(cfg,info,v,0,ParameterServer(0,v,False));learner.init()
import importlib.util
import sf_working_directories.IntrMotiv.dmlab.topological_frontier as topology
old_spec=importlib.util.spec_from_file_location('sf_working_directories.IntrMotiv.dmlab._old_topological',
 '/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_controller_screened_20260912/sf_working_directories/IntrMotiv/dmlab/topological_frontier.py')
old_topology=importlib.util.module_from_spec(old_spec);sys.modules[old_spec.name]=old_topology;old_spec.loader.exec_module(old_topology)
new_select=topology.select_connectivity_probe
from functools import partial
import numpy as np,time
from sf_working_directories.IntrMotiv.dmlab.controller_snapshot import evaluate_replay
from sf_working_directories.IntrMotiv.dmlab.controller_transition import transition_values_batch
keys=list(learner.replay.rows);rng=np.random.default_rng(42)
selected=[keys[int(i)] for i in rng.integers(len(keys),size=8192)]
counts=collections.Counter();seconds={'full':0.,'screened':0.};max_q_error=0.
for begin in range(0,len(selected),256):
 examples=[]
 for key in selected[begin:begin+256]:
  try:examples.append(learner._example(key))
  except (ReplayRejected,ValueError) as exc:counts[str(exc)]+=1
 for snapshot in (learner.online_snapshot,learner.target_snapshot):
  pairs=[]
  for name,flag in (('full',False),('screened',True)):
   topology.select_connectivity_probe=new_select if flag else old_topology.select_connectivity_probe
   torch.cuda.synchronize();start=time.perf_counter()
   pairs.append(evaluate_replay(snapshot,partial(transition_values_batch,screen_current=flag),examples))
   torch.cuda.synchronize();seconds[name]+=time.perf_counter()-start
  for a,b in zip(*pairs):
   if isinstance(a,str):assert a==b,(a,b);counts[a]+=1
   else:
    assert not isinstance(b,str),b
    counts['eligible']+=1
    assert a['done']==b['done'] and torch.equal(a['signature'],b['signature'])
    # Search uses only exact rejection/event/boundary labels; values
    # are recomputed with the original full batch geometry for gradients.
    max_q_error=max(max_q_error,float((a['q']-b['q']).abs().max()))
print(json.dumps(dict(paired_candidates=8192,network_evaluations=16384,counts=counts,seconds=seconds,max_q_error=max_q_error)),flush=True)
import copy
import sf_working_directories.IntrMotiv.dmlab.controller_learner as implementation
original_operation=implementation.transition_values_batch
initial_model=copy.deepcopy(learner.actor_critic.state_dict())
initial_optimizer=copy.deepcopy(learner.optimizer.state_dict())
initial_clock=copy.deepcopy(learner.clock.state_dict())
initial_main_rng=copy.deepcopy(learner.replay.rng.bit_generator.state)
initial_her_rng=copy.deepcopy(learner.her_rng.bit_generator.state)
initial_rejected=copy.deepcopy(learner.replay.rejected)
initial_version=learner.controller_version
initial_online_version=learner.online_snapshot.version
learner.clock.due=lambda accepted:3

def run_updates(screen):
 implementation.transition_values_batch=original_operation
 topology.select_connectivity_probe=new_select if screen else old_topology.select_connectivity_probe
 torch.cuda.synchronize();started=time.perf_counter()
 learner._controller_updates();torch.cuda.synchronize()
 return time.perf_counter()-started

full_seconds=run_updates(False)
expected_model=copy.deepcopy(learner.actor_critic.state_dict())
expected_optimizer=copy.deepcopy(learner.optimizer.state_dict())
expected_clock=copy.deepcopy(learner.clock.state_dict())
expected_main_rng=copy.deepcopy(learner.replay.rng.bit_generator.state)
expected_her_rng=copy.deepcopy(learner.her_rng.bit_generator.state)
learner.actor_critic.load_state_dict(initial_model)
learner.optimizer.load_state_dict(initial_optimizer)
learner.clock.load_state_dict(initial_clock)
learner.replay.rng.bit_generator.state=initial_main_rng
learner.her_rng.bit_generator.state=initial_her_rng
learner.replay.rejected=initial_rejected
learner.controller_version=initial_version
learner.online_snapshot.version=initial_online_version
# Three updates from 1277 do not cross a target refresh.
assert learner.clock.target_at==initial_clock['target_at']
screened_seconds=run_updates(True)

def exact(a,b):
 if torch.is_tensor(a):assert torch.equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:exact(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):exact(x,y)
 else:assert a==b
exact(expected_model,learner.actor_critic.state_dict())
exact(expected_optimizer,learner.optimizer.state_dict())
exact(expected_clock,learner.clock.state_dict())
exact(expected_main_rng,learner.replay.rng.bit_generator.state)
exact(expected_her_rng,learner.her_rng.bit_generator.state)
print(json.dumps(dict(checkpoint=str(source),optimizer_transactions_exact=True,
 baseline_seconds=full_seconds,optimized_seconds=screened_seconds,speedup=full_seconds/screened_seconds,
 before=initial_clock,after=learner.clock.state_dict(),stats=learner.controller_stats)),flush=True)
