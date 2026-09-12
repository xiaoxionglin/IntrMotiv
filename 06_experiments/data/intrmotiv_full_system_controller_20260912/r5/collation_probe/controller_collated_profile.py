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
batch=root/'intrmotiv_full_system_controller_preflight_20260912_r5'
source=sorted((batch/'FSCP_WAYPOINT_F64_DDQN_HER_S99_'/'00_FSCP_WAYPOINT_F64_DDQN_HER_S99'/'checkpoint_p0').glob('checkpoint_*.pth'))[-1]
record={'baseline':str(source)}
cfg=AttrDict(json.loads((source.parent.parent/'config.json').read_text()));cfg.device='gpu';cfg.cli_args={}
cfg.train_dir=tempfile.mkdtemp(prefix='eligibility_',dir=root/'analysis')
target=Path(cfg.train_dir)/cfg.experiment/'checkpoint_p0';target.mkdir(parents=True);os.link(record['baseline'],target/source.name)
register_dmlab_components();torch.set_num_threads(2)
env=make_env_func_batched(cfg,env_config=None);info=extract_env_info(env,cfg);env.close()
v=torch.zeros(1,dtype=torch.int32);learner=ControllerLearner(cfg,info,v,0,ParameterServer(0,v,False));learner.init()




import copy,time,importlib.util
import sf_working_directories.IntrMotiv.dmlab.controller_learner as implementation
spec=importlib.util.spec_from_file_location('sf_working_directories.IntrMotiv.dmlab._uncollated_transition',
 '/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_controller_stable_head_20260912/sf_working_directories/IntrMotiv/dmlab/controller_transition.py')
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
new_operation=implementation.transition_values_batch
initial=dict(model=copy.deepcopy(learner.actor_critic.state_dict()),optimizer=copy.deepcopy(learner.optimizer.state_dict()),
 target=copy.deepcopy(learner.target_snapshot.model.state_dict()),target_version=learner.target_snapshot.version,
 clock=copy.deepcopy(learner.clock.state_dict()),main_rng=copy.deepcopy(learner.replay.rng.bit_generator.state),
 her_rng=copy.deepcopy(learner.her_rng.bit_generator.state),rejected=copy.deepcopy(learner.replay.rejected),
 version=learner.controller_version,online_version=learner.online_snapshot.version)
learner.clock.due=lambda accepted:10

def snapshot():
 return copy.deepcopy(dict(model=learner.actor_critic.state_dict(),optimizer=learner.optimizer.state_dict(),
 target=learner.target_snapshot.model.state_dict(),target_version=learner.target_snapshot.version,
 clock=learner.clock.state_dict(),main_rng=learner.replay.rng.bit_generator.state,her_rng=learner.her_rng.bit_generator.state,
 rejected=learner.replay.rejected))
def run(operation):
 implementation.transition_values_batch=operation
 torch.cuda.synchronize();started=time.perf_counter();learner._controller_updates();torch.cuda.synchronize()
 return time.perf_counter()-started
baseline_seconds=run(old.transition_values_batch);expected=snapshot()
learner.actor_critic.load_state_dict(initial['model']);learner.optimizer.load_state_dict(initial['optimizer'])
learner.target_snapshot.model.load_state_dict(initial['target']);learner.target_snapshot.version=initial['target_version']
learner.clock.load_state_dict(initial['clock']);learner.replay.rng.bit_generator.state=initial['main_rng']
learner.her_rng.bit_generator.state=initial['her_rng'];learner.replay.rejected=initial['rejected']
learner.controller_version=initial['version'];learner.online_snapshot.version=initial['online_version']
optimized_seconds=run(new_operation)
def exact(a,b,path=''):
 if torch.is_tensor(a):assert torch.equal(a,b),path
 elif isinstance(a,dict):
  assert a.keys()==b.keys(),path
  for k in a:exact(a[k],b[k],path+'/'+str(k))
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b)):exact(x,y,path+'/'+str(i))
 else:assert a==b,path
exact(expected,snapshot())
print(json.dumps(dict(checkpoint=str(source),full_updates=10,optimizer_transactions_exact=True,
 baseline_seconds=baseline_seconds,optimized_seconds=optimized_seconds,speedup=baseline_seconds/optimized_seconds,
 before=initial['clock'],after=learner.clock.state_dict(),stats=learner.controller_stats)),flush=True)
