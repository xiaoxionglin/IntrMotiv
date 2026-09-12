import json,sys
from pathlib import Path
sys.path.insert(0,'/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_controller_terminal_20260912')
from sample_factory.utils.attr_dict import AttrDict
from sf_working_directories.IntrMotiv.dmlab.dmlab_env import make_dmlab_env
root=Path('/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir')
p=next((root/'intrmotiv_full_system_controller_preflight_20260912_r2').glob('FSCP_DIRECT_F16_DDQN_S99_/00_*/config.json'))
cfg=AttrDict(json.loads(p.read_text()))
env=make_dmlab_env(cfg.env,cfg,AttrDict(worker_index=0,vector_index=0,env_id=0))
obs,info=env.reset(seed=99)
for decision in range(10000):
 obs,reward,terminated,truncated,info=env.step(0)
 if terminated or truncated:
  print(json.dumps(dict(decisions=decision+1,terminated=terminated,truncated=truncated,certified=info.get('controller_final_observation_valid'),keys=list(info))),flush=True)
  assert info.get('controller_final_observation_valid'), 'Real DMLab terminal observation not certified'
  break
else:raise RuntimeError('No physical episode end within test bound')
env.close()
