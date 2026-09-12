import json,sys,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_controller_20260912')
from sample_factory.utils.attr_dict import AttrDict
from sf_working_directories.IntrMotiv.dmlab.dmlab_env import make_dmlab_env
root=Path('/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir')
p=next((root/'intrmotiv_full_system_controller_preflight_20260912_r2').glob('FSCP_DIRECT_F16_PPO_S99_/00_*/config.json'))
cfg=AttrDict(json.loads(p.read_text()));env=make_dmlab_env(cfg.env,cfg,AttrDict(worker_index=0,vector_index=0,env_id=0))
obs,info=env.reset(seed=99);digest=hashlib.sha256()
for decision in range(2000):
 obs,reward,terminated,truncated,info=env.step((decision//20)%8)
 for key in sorted(obs):digest.update(np.asarray(obs[key]).tobytes())
 digest.update(np.asarray([reward,float(terminated),float(truncated)],dtype=np.float64).tobytes())
 if terminated or truncated:break
else:raise RuntimeError('No episode boundary')
result=dict(decisions=decision+1,sha256=digest.hexdigest(),terminal=terminated)
Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);env.close()
