import json,sys,os,tempfile,random
from pathlib import Path
import numpy as np,torch
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.model_sharing import ParameterServer
from sf_working_directories.IntrMotiv.dmlab.train_hipposlam import register_dmlab_components
from sf_working_directories.IntrMotiv.dmlab.controller_learner import ControllerLearner
from sf_working_directories.IntrMotiv.dmlab.custom_learner import DistanceLearnerReward
from sf_working_directories.IntrMotiv.evaluation.place_fields import load_checkpoint_dict
root=Path('/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir')
base=root/'analysis/restart_baselines/intrmotiv_full_system_controller_preflight_20260912_r3'
record=json.loads((base/'baselines.json').read_text())[int(sys.argv[1])]
source=Path(record['source']);cfg=AttrDict(json.loads((source.parent.parent/'config.json').read_text()))
cfg.train_dir=tempfile.mkdtemp(prefix='exact_restore_',dir=root/'analysis');cfg.cli_args={}
target=Path(cfg.train_dir)/cfg.experiment/'checkpoint_p0';target.mkdir(parents=True)
os.link(record['baseline'],target/source.name)
register_dmlab_components();torch.set_num_threads(1)
env=make_env_func_batched(cfg,env_config=None);info=extract_env_info(env,cfg);env.close()
versions=torch.zeros(1,dtype=torch.int32);server=ParameterServer(0,versions,False)
cls=ControllerLearner if cfg.controller_learning=='ddqn' else DistanceLearnerReward
learner=cls(cfg,info,versions,0,server);learner.init()
ck=load_checkpoint_dict(record['baseline'],torch.device('cpu'))
def equal(a,b):
 if torch.is_tensor(a):torch.testing.assert_close(a,b,rtol=0,atol=0)
 elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b,(a,b)
equal(ck['model'],learner.actor_critic.state_dict());equal(ck['optimizer'],learner.optimizer.state_dict())
assert learner.env_steps==ck['env_steps'] and learner.train_step==ck['train_step']
if 'controller' in ck:
 c=ck['controller'];equal(c['target'],learner.target_snapshot.model.state_dict());equal(c['clock'],learner.clock.state_dict())
 equal(c['replay']['rng'],learner.replay.rng.bit_generator.state);equal(c['her_rng'],learner.her_rng.bit_generator.state)
 equal(c['torch_rng'],torch.get_rng_state());equal(c['python_rng'],random.getstate())
 equal((c['numpy_rng'][0],c['numpy_rng'][1].numpy().astype(np.uint32),*c['numpy_rng'][2:]),np.random.get_state())
 assert learner.replay.session==c['replay']['session']+1
 assert learner.publication==c['publication'] and learner.fresh_dg_steps==c['fresh_dg_steps']
 assert learner.replay.ingress.pending==0
 assert len(learner.replay.rows)==len(c['replay']['rows'])
print(json.dumps(dict(run=record['run'],exact_restore=True,frames=learner.env_steps,controller=cfg.controller_learning)),flush=True)
