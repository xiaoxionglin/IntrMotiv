import sys,json,torch,numpy as np,gymnasium as gym,tempfile,copy,importlib.util
from pathlib import Path
sys.path.insert(0,'/tmp/intrmotiv_stable_head_benchmark')
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.shared_buffers import alloc_trajectory_tensors,policy_output_shapes,action_info
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sf_working_directories.IntrMotiv.dmlab.train_hipposlam import register_dmlab_components,maybe_overwrite_rnn_size
from sf_working_directories.IntrMotiv.dmlab.custom_learner import DistanceLearnerReward
spec=importlib.util.spec_from_file_location('sf_working_directories.IntrMotiv.dmlab._baseline_learner','/tmp/intrmotiv_full_system_baseline_20260912/dmlab/custom_learner.py')
baseline=importlib.util.module_from_spec(spec);sys.modules[spec.name]=baseline;spec.loader.exec_module(baseline)
import sf_working_directories.IntrMotiv.dmlab.topological_frontier as topology
old_spec=importlib.util.spec_from_file_location('sf_working_directories.IntrMotiv.dmlab._old_topological',
 '/tmp/intrmotiv_screened_benchmark/sf_working_directories/IntrMotiv/dmlab/topological_frontier.py')
old_topology=importlib.util.module_from_spec(old_spec);sys.modules[old_spec.name]=old_topology;old_spec.loader.exec_module(old_topology)
new_select=topology.select_connectivity_probe
register_dmlab_components();torch.set_num_threads(2)
root=Path('/home/xiaoxiong/Desktop/Projects/IntrMotiv/06_experiments/data/intrmotiv_full_system_controller_20260912')
results=[]
for path in sorted(root.glob('*/preservation_manifest.json')):
 cfg=AttrDict(json.loads(path.read_text())['config'])
 cfg.update(device='cpu',train_dir=tempfile.mkdtemp(prefix='ppo-preservation-audit-'),
  controller_learning='ddqn',controller_her=True,controller_epsilon=.1,controller_learning_starts=16384,controller_epsilon_decay_decisions=250000,rollout=4,recurrence=4,batch_size=8,num_batches_per_epoch=1,
  num_epochs=1,online_spatial_telemetry=False,save_initial_checkpoint=False,checkpoint_frame_targets='',cli_args={})
 cfg.update(decorrelate_envs_on_one_worker=False,controller_preflight=True,train_for_env_steps=2000000,
  controller_replay_capacity=200000,controller_td_positions=256,controller_decisions_per_update=64,
  controller_target_updates=100,controller_her_positions=256,controller_her_loss_coeff=1.,controller_cache_visual=False)
 maybe_overwrite_rnn_size(cfg)
 cfg.extra_policy_output_shapes=tuple(pair for pair in cfg.extra_policy_output_shapes if not pair[0].startswith('controller_'))
 space=gym.spaces.Dict({'obs':gym.spaces.Box(0,255,(4,72,96),np.uint8),
  'INSTR':gym.spaces.Box(0,3,(1,),np.int64),'prev_action':gym.spaces.Discrete(8)})
 info=EnvInfo(space,gym.spaces.Discrete(8),1,False,False,[],True,4)
 learners=[]
 for cls in (DistanceLearnerReward,DistanceLearnerReward):
  versions=torch.zeros(1,dtype=torch.int32);server=ParameterServer(0,versions,False)
  learner=cls(cfg,info,versions,0,server);learner.init()
  import types
  from sf_working_directories.IntrMotiv.dmlab.controller_learner import ControllerLearner
  learner._calculate_losses=types.MethodType(ControllerLearner._calculate_losses,learner)
  sum(p.sum() for p in learner.actor_critic.parameters() if p.requires_grad).backward()
  learner.optimizer.step();learner.optimizer.zero_grad(set_to_none=True)
  learners.append(learner)
 batch=alloc_trajectory_tensors(cfg,info,2,4,cfg.rnn_size,torch.device('cpu'),False)
 batch['obs']['obs'].random_(0,256);batch['obs']['INSTR'].fill_(1);batch['obs']['prev_action'].zero_()
 state=torch.zeros(2,cfg.rnn_size);batch['rnn_states'][:,0]=state
 actor=learners[0].actor_critic.eval()
 for j in range(4):
  observation={k:v[:,j] for k,v in batch['obs'].items()}
  with torch.no_grad():result=actor(prepare_and_normalize_obs(actor,observation),state)
  result['policy_version']=torch.zeros(2)
  for key,_ in policy_output_shapes(cfg,*action_info(info)):batch[key][:,j]=result[key].reshape_as(batch[key][:,j])
  state=result['new_rnn_states'];batch['rnn_states'][:,j+1]=state
 batch['policy_id'].zero_();batch['dones'].zero_();batch['time_outs'].zero_();batch['rewards'].zero_()
 from sf_working_directories.IntrMotiv.dmlab.controller_snapshot import fresh_dg_parameter_owner
 before={k:v.detach().clone() for k,v in learners[1].actor_critic.named_parameters()}
 for index,learner in enumerate(learners):
  topology.select_connectivity_probe=new_select
  torch.manual_seed(771);np.random.seed(771)
  if index==0:learner.train(copy.deepcopy(batch))
  else:
   with fresh_dg_parameter_owner(learner.actor_critic):learner.train(copy.deepcopy(batch))
 a,b=learners
 for k,v in a.actor_critic.state_dict().items():
  if k.startswith('encoder.DG_projection.'):
   torch.testing.assert_close(v,b.actor_critic.state_dict()[k],rtol=0,atol=0)
 for (ka,pa),(kb,pb) in zip(a.actor_critic.named_parameters(),b.actor_critic.named_parameters()):
  assert ka==kb
  if ka.startswith('encoder.DG_projection.'):
   for key,value in a.optimizer.state[pa].items():
    torch.testing.assert_close(value,b.optimizer.state[pb][key],rtol=0,atol=0)
 changed=[]
 for k,v in b.actor_critic.named_parameters():
  if not k.startswith('encoder.DG_projection.'):
   assert torch.equal(v,before[k]),k
   if not torch.equal(dict(a.actor_critic.named_parameters())[k],before[k]):changed.append(k)
 assert a.train_step==b.train_step==1
 results.append(dict(parent=path.parent.name,fresh_dg_parameters_and_buffers_exact=True,fresh_dg_optimizer_exact=True,
  non_dg_parameters_unchanged=True,previously_drifting_parameters=changed))
 print('PASS '+path.parent.name,flush=True)
Path('/tmp/controller_fresh_owner_parity.json').write_text(json.dumps(results,indent=2)+'\n')
