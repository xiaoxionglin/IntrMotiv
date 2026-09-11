"""Batch independent replay histories without sharing online/target memories."""
import torch
import time


def prefix_memory(worker, prefix, device):
    if not prefix:
        return worker.initial(1,device)
    pre = torch.stack([r.observation.preactivation for r in prefix]).to(device)[:,None]
    bypass = torch.stack([r.observation.bypass for r in prefix]).to(device)[:,None]
    goals = torch.tensor([r.goal for r in prefix],device=device)[:,None]
    budgets = torch.tensor([r.budget for r in prefix],device=device)[:,None]
    return worker.rebuild(pre,bypass,goals,budgets,episode_start=prefix[0].index == 0)


def learn_batch(learner, samples, device):
    started = time.monotonic()
    length = max(len(s['segment']) for s in samples)
    pre, bypass, goals, budgets, clocks, actions, rewards, dones, masks = [],[],[],[],[],[],[],[],[]
    online_memory, target_memory = [],[]
    for s in samples:
        online_memory.append(prefix_memory(learner.online,s['prefix'],device))
        target_memory.append(prefix_memory(learner.target,s['prefix'],device))
        rows,obs = s['segment'],s['observations']
        t = len(rows)
        padded = obs + [obs[-1]]*(length-t)
        pre.append(torch.stack([o.preactivation for o in padded]))
        bypass.append(torch.stack([o.bypass for o in padded]))
        goals.append(torch.full((length+1,),s['goal'],dtype=torch.long))
        budgets.append(torch.arange(s['budget'],s['budget']-length-1,-1).clamp_min(0))
        clocks.append(torch.arange(rows[0].index,rows[0].index+length+1))
        actions.append(torch.tensor([r.action for r in rows]+[0]*(length-t)))
        rewards.append(torch.nn.functional.pad(s['reward'],(0,length-t)))
        dones.append(torch.nn.functional.pad(s['done'],(0,length-t),value=True))
        masks.append(torch.nn.functional.pad(s['mask'],(0,length-t),value=False))
    tensors = [torch.stack(x,1).to(device) for x in (pre,bypass,goals,budgets,actions,rewards,dones,masks)]
    prepared = time.monotonic()
    result = learner.update(*tensors,torch.cat(online_memory),torch.cat(target_memory),
                            episode_decisions=torch.stack(clocks,1).to(device))
    result['prefix_and_batch_seconds'] = prepared - started
    result['learner_update_seconds'] = time.monotonic() - prepared
    return result
