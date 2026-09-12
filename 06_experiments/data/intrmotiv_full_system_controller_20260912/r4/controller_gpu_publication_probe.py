from types import SimpleNamespace
from threading import Lock
import torch,json,time,inspect
from sample_factory.algo.utils.model_sharing import ParameterClientAsync
from sample_factory.utils.timing import Timing
client=object.__new__(ParameterClientAsync)
client.cfg=SimpleNamespace(device='gpu',controller_learning='ddqn',serial_mode=False);client.policy_id=0;client.policy_versions=torch.tensor([1])
client.latest_policy_version=0;client.num_policy_updates=0;client._policy_lock=Lock();client.timing=Timing()
client._actor_critic=torch.nn.Linear(1024,1024).cuda()
client._shared_model_weights={k:torch.ones_like(v) for k,v in client._actor_critic.state_dict().items()}
event=torch.cuda.Event(enable_timing=True);original=client._actor_critic.load_state_dict
def tracked(weights):
    result=original(weights);event.record();return result
client._actor_critic.load_state_dict=tracked
torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);start.record();wall=time.perf_counter();torch.cuda._sleep(100000000)
client.ensure_weights_updated();host_seconds=time.perf_counter()-wall;completed=event.query()
torch.cuda.synchronize()
print(json.dumps(dict(source=inspect.getfile(ParameterClientAsync),version_visible=client.latest_policy_version,
    lock_released=not client._policy_lock.locked(),copy_marker_completed_at_return=completed,
    host_seconds=host_seconds,gpu_seconds=start.elapsed_time(event)/1000)),flush=True)
