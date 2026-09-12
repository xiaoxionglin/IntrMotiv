import sys,json,torch,numpy as np,gymnasium as gym,tempfile
from pathlib import Path
sys.path.insert(0,'/tmp/intrmotiv_full_system_20260912')
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.shared_buffers import alloc_trajectory_tensors,policy_output_shapes,action_info
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sf_working_directories.IntrMotiv.dmlab.train_hipposlam import register_dmlab_components,maybe_overwrite_rnn_size
from sf_working_directories.IntrMotiv.dmlab.controller_learner import ControllerLearner
register_dmlab_components();torch.set_num_threads(2)
root=Path('/home/xiaoxiong/Desktop/Projects/IntrMotiv/06_experiments/data/intrmotiv_full_system_controller_20260912')
for path in sorted(root.glob('*/preservation_manifest.json')):
 cfg=AttrDict(json.loads(path.read_text())['config'])
 cfg.update(device='cpu',train_dir=tempfile.mkdtemp(prefix='native-controller-audit-'),
  controller_learning='ddqn',controller_her=True,controller_learning_starts=1,controller_decisions_per_update=4,
  controller_td_positions=2,controller_her_positions=2,controller_her_loss_coeff=1.,controller_replay_capacity=1000,
  controller_target_updates=2,controller_epsilon=.1,controller_epsilon_decay_decisions=250000,
  controller_preflight=True,controller_cache_visual=True,train_for_env_steps=1000,
  rollout=4,recurrence=4,batch_size=8,num_batches_per_epoch=1,num_epochs=1,online_spatial_telemetry=False,
  save_initial_checkpoint=False,checkpoint_frame_targets='',cli_args={})
 maybe_overwrite_rnn_size(cfg)
 space=gym.spaces.Dict({'obs':gym.spaces.Box(0,255,(4,72,96),np.uint8),
  'INSTR':gym.spaces.Box(0,3,(1,),np.int64),'prev_action':gym.spaces.Discrete(8),
  'controller_identity':gym.spaces.Box(0,np.iinfo(np.int64).max,(4,),np.int64)})
 info=EnvInfo(space,gym.spaces.Discrete(8),1,False,False,[],True,4)
 versions=torch.zeros(1,dtype=torch.int32)
 server=ParameterServer(0,versions,False)
 learner=ControllerLearner(cfg,info,versions,0,server);learner.init()
 state=torch.zeros(2,cfg.rnn_size)
 for packet in range(2):
  batch=alloc_trajectory_tensors(cfg,info,2,4,cfg.rnn_size,torch.device('cpu'),False)
  batch['obs']['obs'].random_(0,256);batch['obs']['INSTR'].fill_(1);batch['obs']['prev_action'].zero_()
  for i in range(2):
   for j in range(5):batch['obs']['controller_identity'][i,j]=torch.tensor([i,0,packet*4+j,packet*4+j])
  batch['rnn_states'][:,0]=state
  for j in range(4):
   observation={k:v[:,j] for k,v in batch['obs'].items()}
   with torch.no_grad():result=learner.published(prepare_and_normalize_obs(learner.published,observation),state)
   result['policy_version']=torch.tensor([learner.publication]*2);result['controller_input_state']=state
   result['controller_memory_stats']=torch.zeros(2,3)
   for key,_ in policy_output_shapes(cfg,*action_info(info)):
    batch[key][:,j]=result[key].reshape_as(batch[key][:,j])
   state=result['new_rnn_states'];batch['rnn_states'][:,j+1]=state
  batch['policy_id'].zero_();batch['dones'].zero_();batch['time_outs'].zero_();batch['rewards'].zero_()
  batch['controller_final_valid'].zero_();batch['controller_terminated'].zero_();batch['controller_frames'].fill_(4)
  stats=learner.train(batch)
  print(path.parent.name,packet,learner.clock,learner.replay.rejected,flush=True)
 assert learner.fresh_dg_steps==2
 checkpoint=learner._get_checkpoint_dict();assert 'controller' in checkpoint
 assert learner._save_impl('checkpoint','',1)
 resumed=ControllerLearner(cfg,info,torch.zeros_like(versions),0,ParameterServer(0,torch.zeros_like(versions),False))
 resumed.init()
 assert resumed.clock.state_dict()==learner.clock.state_dict()
 assert resumed.replay.session==learner.replay.session+1
 assert resumed.publication==learner.publication
 for k,v in learner.target_snapshot.model.state_dict().items():
  torch.testing.assert_close(v,resumed.target_snapshot.model.state_dict()[k],rtol=0,atol=0)
 for a,b in zip(learner.actor_critic.parameters(),learner.published.parameters()):assert a.data_ptr()!=b.data_ptr()
 print('PASS '+path.parent.name,flush=True)
