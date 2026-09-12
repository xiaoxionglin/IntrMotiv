import argparse,importlib.util,json,sys
parser=argparse.ArgumentParser()
parser.add_argument("--source-root",required=True)
parser.add_argument("--baseline-actor",required=True)
parser.add_argument("--manifests",required=True)
parser.add_argument("--output",required=True)
args=parser.parse_args()
sys.path.insert(0,args.source_root)
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from sample_factory.model.model_factory import ModelFactory
from sf_working_directories.IntrMotiv.dmlab.dmlab_gym import ACTION_SET, REDUCED_ACTION_SET, NAVIGATION_ACTION_SET, EXTENDED_ACTION_SET
from sample_factory.utils.attr_dict import AttrDict
from sf_working_directories.IntrMotiv.dmlab.custom_actor_critic import IntrMotivActorCriticSharedWeights as Current
from sf_working_directories.IntrMotiv.dmlab.custom_core import make_hipposlam_core
from sf_working_directories.IntrMotiv.dmlab.custom_encoder import make_hipposlam_encoder
from sf_working_directories.IntrMotiv.dmlab.custom_decoder import make_hipposlam_decoder
from sf_working_directories.IntrMotiv.dmlab.controller_snapshot import ControllerSnapshot, differentiable_replay
from sf_working_directories.IntrMotiv.dmlab.controller_history import ControllerHistory, reconstruct_controller_history
spec=importlib.util.spec_from_file_location('baseline_actor',args.baseline_actor)
baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
factory=ModelFactory()
factory.make_model_encoder_func=make_hipposlam_encoder
factory.make_model_core_func=make_hipposlam_core
factory.make_model_decoder_func=make_hipposlam_decoder
torch.set_num_threads(2)
root=Path(args.manifests)
results=[]
for path in sorted(root.glob('*/preservation_manifest.json')):
 p=json.loads(path.read_text());cfg=AttrDict(p['config']); cfg.device='cpu'
 actions = EXTENDED_ACTION_SET if cfg.dmlab_extended_action_set else ACTION_SET
 if cfg.dmlab_reduced_action_set:actions = REDUCED_ACTION_SET
 if cfg.dmlab_navigation_action_set:actions = NAVIGATION_ACTION_SET
 n_actions=len(actions)
 obs_space=gym.spaces.Dict({'obs':gym.spaces.Box(0,255,(4,cfg.res_h,cfg.res_w),np.uint8),'INSTR':gym.spaces.Box(0,3,(1,),np.int64),'prev_action':gym.spaces.Discrete(n_actions)})
 action_space=gym.spaces.Discrete(n_actions)
 torch.manual_seed(991)
 old=baseline.IntrMotivActorCriticSharedWeights(factory,obs_space,action_space,cfg).eval()
 oldrng=torch.get_rng_state().clone()
 torch.manual_seed(991)
 current=Current(factory,obs_space,action_space,cfg).eval()
 assert torch.equal(oldrng,torch.get_rng_state())
 assert old.state_dict().keys()==current.state_dict().keys()
 for k,v in old.state_dict().items():assert torch.equal(v,current.state_dict()[k]),k
 shadowcfg=AttrDict(dict(cfg,controller_learning='shadow',controller_her=True))
 torch.manual_seed(991)
 shadow=Current(factory,obs_space,action_space,shadowcfg).eval()
 assert torch.equal(oldrng,torch.get_rng_state())
 for k,v in old.state_dict().items():assert torch.equal(v,shadow.state_dict()[k]),k
 states=[torch.zeros(2,old.core.total_state_size) for _ in range(3)]
 for step in range(5):
  torch.manual_seed(550+step)
  obs={'obs':torch.rand(2,4,cfg.res_h,cfg.res_w),'INSTR':torch.ones(2,1,dtype=torch.long),'prev_action':torch.full((2,),step%n_actions,dtype=torch.long)}
  outputs=[]
  for i,actor in enumerate((old,current,shadow)):
   torch.manual_seed(999+step)
   with torch.no_grad():out=actor(obs,states[i])
   states[i]=out['new_rnn_states'];outputs.append(out)
  for out in outputs[1:]:
   for key in outputs[0]:torch.testing.assert_close(out[key],outputs[0][key],rtol=0,atol=0)
 from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
 from sf_working_directories.IntrMotiv.dmlab.controller_transport import cached_observation
 raw={'obs':torch.randint(0,256,(4,4,cfg.res_h,cfg.res_w),dtype=torch.uint8), 'INSTR':torch.ones(4,1,dtype=torch.long),'prev_action':torch.zeros(4,dtype=torch.long)}
 norm=prepare_and_normalize_obs(shadow,dict(raw))
 with torch.no_grad():
  original_head=shadow.forward_head(norm)
  cache=shadow.encoder._controller_visual.clone()
  cached={'obs':raw['obs'][:,-1:].clone(),'INSTR':raw['INSTR'],'prev_action':raw['prev_action'],'controller_visual':cache}
  cached_head=shadow.forward_head(prepare_and_normalize_obs(shadow,cached))
  torch.testing.assert_close(original_head,cached_head,rtol=0,atol=0)
 from sf_working_directories.IntrMotiv.dmlab.controller_replay import PhysicalDecision
 from sf_working_directories.IntrMotiv.dmlab.controller_transition import transition_values,TransitionInput
 physical=[]; physical_states=[torch.zeros(1,shadow.core.total_state_size)]
 for t in range(4):
  observation={k:v[t:t+1] for k,v in raw.items()}
  with torch.no_grad():
   result=shadow(prepare_and_normalize_obs(shadow,dict(observation)),physical_states[-1])
  physical_states.append(result['new_rnn_states'])
  physical.append(PhysicalDecision((0,0),0,t,t,{k:v[0].numpy() for k,v in observation.items()},0,
      result['controller_condition'][0].numpy(),result['controller_context'][0].numpy(),0,0,False,False,4))
 for t in range(3):
  value=transition_values(shadow,TransitionInput(tuple(physical[:t+2]),t))
  assert torch.isfinite(value['reward'])
 before={k:v.clone() for k,v in current.state_dict().items()}
 snapshot=ControllerSnapshot(lambda:Current(factory,obs_space,action_space,cfg),current,0)
 history=ControllerHistory(
  {'obs':torch.randint(0,256,(5,4,cfg.res_h,cfg.res_w),dtype=torch.uint8),
   'INSTR':torch.ones(5,1,dtype=torch.long),'prev_action':torch.zeros(5,dtype=torch.long)},
  torch.arange(5),torch.zeros(1,current.core.total_state_size),
  torch.nn.functional.one_hot(torch.ones(5,dtype=torch.long),cfg.Hippo_n_feature).float(),True,0)
 replay=differentiable_replay(snapshot,current,reconstruct_controller_history,history,source_version=0)
 replay['hidden'].sum().backward()
 assert sum(v.grad.abs().sum() for v in current.decoder.parameters() if v.grad is not None)>0
 for k,v in current.state_dict().items():assert torch.equal(before[k],v),k
 assert all(v.grad is None for v in snapshot.model.parameters())
 results.append({'parent':p['run'],'state_dict_equal':True,'actual_model_replay_snapshot_isolated':True,'actual_decoder_replay_gradients':True,'initialization_rng_equal':True,'ppo_and_shadow_outputs_equal':True,'forced_observation_steps':5,'environments':2,'action_vectors':actions,'new_shadow_parameters':{k:list(v.shape) for k,v in shadow.named_parameters() if k not in old.state_dict()},'scope':'actual encoder/core/decoder; synthetic observations, no environment rollout'})
 print(p['run']+' passed',flush=True)
Path(args.output).write_text(json.dumps(results,indent=2)+'\n')
