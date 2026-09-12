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


import copy,time,math
before=copy.deepcopy(learner.clock.state_dict())
buffers={k:v.detach().clone() for k,v in learner.actor_critic.named_buffers()}
dg={k:v.detach().clone() for k,v in learner.actor_critic.named_parameters() if k.startswith('encoder.DG_projection.')}
fresh=(learner.fresh_dg_steps,learner.fresh_graph_batches)
learner.clock.due=lambda accepted:101
learner.set_progress_callback(lambda:print(json.dumps(dict(completed=learner.clock.completed)),flush=True))
started=time.perf_counter();learner._controller_updates();torch.cuda.synchronize()
assert learner.clock.completed-before['completed']==101
assert learner.clock.main_positions-before['main_positions']==101*256
assert learner.clock.auxiliary_positions>before['auxiliary_positions']
assert learner.clock.target_at>before['target_at']
assert fresh==(learner.fresh_dg_steps,learner.fresh_graph_batches)
for k,v in learner.actor_critic.named_buffers():assert torch.equal(v,buffers[k]),k
for k,v in learner.actor_critic.named_parameters():
 if k in dg:assert torch.equal(v,dg[k]),k
assert all(math.isfinite(v) for v in learner.controller_stats.values())
print(json.dumps(dict(full_updates_passed=True,before=before,after=learner.clock.state_dict(),
 unchanged_online_buffers=True,unchanged_stop_dg_parameters=True,
 seconds=time.perf_counter()-started,stats=learner.controller_stats)),flush=True)
