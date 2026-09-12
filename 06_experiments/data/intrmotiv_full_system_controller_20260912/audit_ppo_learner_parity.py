import sys,json,torch,numpy as np,gymnasium as gym,tempfile,copy,importlib.util
from pathlib import Path
sys.path.insert(0,'/tmp/intrmotiv_full_system_20260912')
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.shared_buffers import alloc_trajectory_tensors,policy_output_shapes,action_info
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sf_working_directories.IntrMotiv.dmlab.train_hipposlam import register_dmlab_components,maybe_overwrite_rnn_size
from sf_working_directories.IntrMotiv.dmlab.custom_learner import DistanceLearnerReward
spec=importlib.util.spec_from_file_location('sf_working_directories.IntrMotiv.dmlab._baseline_learner','/tmp/intrmotiv_full_system_baseline_20260912/dmlab/custom_learner.py')
baseline=importlib.util.module_from_spec(spec);sys.modules[spec.name]=baseline;spec.loader.exec_module(baseline)
register_dmlab_components();torch.set_num_threads(2)
root=Path('/home/xiaoxiong/Desktop/Projects/IntrMotiv/06_experiments/data/intrmotiv_full_system_controller_20260912')
results=[]
for path in sorted(root.glob('*/preservation_manifest.json')):
 cfg=AttrDict(json.loads(path.read_text())['config'])
 cfg.update(device='cpu',train_dir=tempfile.mkdtemp(prefix='ppo-preservation-audit-'),
  controller_learning='ppo',controller_her=False,rollout=4,recurrence=4,batch_size=8,num_batches_per_epoch=1,
  num_epochs=1,online_spatial_telemetry=False,save_initial_checkpoint=False,checkpoint_frame_targets='',cli_args={})
 maybe_overwrite_rnn_size(cfg)
 space=gym.spaces.Dict({'obs':gym.spaces.Box(0,255,(4,72,96),np.uint8),
  'INSTR':gym.spaces.Box(0,3,(1,),np.int64),'prev_action':gym.spaces.Discrete(8)})
 info=EnvInfo(space,gym.spaces.Discrete(8),1,False,False,[],True,4)
 learners=[]
 for cls in (baseline.DistanceLearnerReward,DistanceLearnerReward):
  versions=torch.zeros(1,dtype=torch.int32);server=ParameterServer(0,versions,False)
  learner=cls(cfg,info,versions,0,server);learner.init();learners.append(learner)
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
 for learner in learners:
  torch.manual_seed(771);np.random.seed(771)
  learner.train(copy.deepcopy(batch))
 a,b=learners
 for k,v in a.actor_critic.state_dict().items():torch.testing.assert_close(v,b.actor_critic.state_dict()[k],rtol=0,atol=0)
 for p,state in a.optimizer.state_dict()['state'].items():
  for k,v in state.items():torch.testing.assert_close(v,b.optimizer.state_dict()['state'][p][k],rtol=0,atol=0)
 assert a.train_step==b.train_step==1
 results.append(dict(parent=path.parent.name,model_and_buffers_bitwise_equal=True,optimizer_bitwise_equal=True,fresh_updates=1,decisions=8))
 print('PASS '+path.parent.name,flush=True)
Path('/tmp/ppo_learner_parity.json').write_text(json.dumps(results,indent=2)+'\n')
