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
    if any(s['relabeled'] for s in samples) and learner.online.write_modulation is not None:
        raise ValueError('write-conditioned virtual prefix reconstruction is not qualified')
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


def slice_attempt(sample, start, stop, width):
    """Slice losses without restarting the virtual deadline or physical memory."""
    result = dict(sample)
    result['prefix'] = (sample['prefix'] + sample['segment'][:start])[-width:]
    result['segment'] = sample['segment'][start:stop]
    result['observations'] = sample['observations'][start:stop+1]
    result['budget'] = sample['budget'] - start
    for key in ('reward', 'done', 'mask'):
        result[key] = sample[key][start:stop]
    return result


class PositionBatcher:
    """Consume every sampled loss once, with an exact TD-position/update budget.

    A pending suffix carries its original deadline into the next update. No
    reward is dropped to make a batch fit; no extra HER gradient budget is hidden.
    """
    def __init__(self):
        self.pending = None
        self.backlog = []

    def sample(self, replay, registry, her_fraction, positions):
        samples, remaining = [], positions
        while remaining:
            if self.pending is None:
                try:
                    self.pending = self.backlog.pop(0) if self.backlog else replay.sample(registry, her_fraction)
                except ValueError:
                    self.backlog = samples + self.backlog
                    raise
            n = min(remaining, len(self.pending['segment']))
            samples.append(slice_attempt(self.pending, 0, n, replay.width))
            self.pending = (slice_attempt(self.pending, n, len(self.pending['segment']), replay.width)
                            if n < len(self.pending['segment']) else None)
            remaining -= n
        return samples
