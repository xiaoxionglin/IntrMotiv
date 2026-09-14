"""Shared frozen-feature DG evaluation, reusing the observation-panel and map contracts.

Collect once using a seeded persistent-random policy. Split only at whole-episode
boundaries. Anchors are selected on calibration observations, never held-out data.
This adapter runs outside the immutable training checkout.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from hpc_runs.intrmotiv_study.spatial_contract import calculate_place_field_details, SpatialBounds
from sf_working_directories.IntrMotiv.evaluation.place_fields import load_policy_env, load_checkpoint_dict
from sf_working_directories.IntrMotiv.evaluation.observation_panel import record_observation, previous_actions_after_step
from sf_working_directories.IntrMotiv.dmlab.dg_neighborhood import enclosure_masks, NeighborhoodConfig
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs


def tensor_digest(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode() + b'\0' + value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [finite_json(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def collect(run, checkpoint, output, decisions, seed, batch_size=256):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    cfg, env, info, actor, checkpoint, _ = load_policy_env(run, decisions, False, 0, checkpoint)
    if not hasattr(env.unwrapped, 'seed'):
        raise RuntimeError('Cannot set DMLab trajectory seed')
    env.unwrapped.seed(seed)
    rng = np.random.default_rng(seed)
    records = defaultdict(list)
    obs, _ = env.reset()
    previous = np.full((env.num_agents, 1), actor.action_space.n, dtype=np.int32)
    if env.num_agents != 1:
        raise ValueError('Shared panel collector expects one environment')
    ends = []
    for t in range(decisions * 3):
        record_observation(records, obs, previous)
        if t % 8 == 0:
            action = int(rng.integers(actor.action_space.n))
        actions = torch.tensor([[action]], dtype=torch.int64)
        records['actions'].append(actions.numpy().copy())
        obs, _, terminated, truncated, _ = env.step(preprocess_actions(info, actions))
        dones = make_dones(terminated, truncated).cpu().numpy().astype(bool)
        records['dones'].append(dones)
        previous = previous_actions_after_step(actions.numpy(), dones, actor.action_space.n)
        if dones.any():
            ends.append(t+1)
            if t+1 >= decisions and len(ends) >= 2:
                break
    env.close()
    if len(ends) < 2 or ends[-1] != len(records['dones']):
        raise RuntimeError('Could not collect two complete episodes within the bound')
    data = {k: np.stack(v) for k, v in records.items()}
    panel = output/'observations.npz'
    # Same arrays as observation_panel.save_panel; its legacy NEMO-only path
    # guard does not apply to this explicit scratch-root workstation adapter.
    np.savez_compressed(panel, **data)
    split = min(ends[:-1], key=lambda x: abs(x-len(data['dones'])/2))
    features = []
    actor.model_to_device(torch.device('cuda'))
    actor.eval()
    trunk_hash = tensor_digest(actor.encoder.basic_encoder.state_dict())
    with torch.no_grad():
        for start in range(0, len(data['dones']), batch_size):
            obs = {k[4:]: torch.as_tensor(v[start:start+batch_size, 0], device='cuda')
                   for k, v in data.items() if k.startswith('obs_')}
            normalized = prepare_and_normalize_obs(actor, obs)
            features.append(actor.encoder.projection_input(normalized).cpu().numpy())
    if trunk_hash != tensor_digest(actor.encoder.basic_encoder.state_dict()):
        raise RuntimeError('Frozen trunk changed during panel encoding')
    pose = data['obs_telemetry_pose'][:, 0].astype(np.float32)
    dones = data['dones'][:, 0]
    segment = np.r_[0, np.cumsum(dones[:-1])].astype(np.int32)
    np.savez_compressed(output/'features.npz', features=np.concatenate(features), pose=pose,
                        dones=dones, segment_id=segment, split=np.asarray(split))
    metadata = dict(protocol='dg-shared-observation-panel-v1', policy='uniform_random_hold_8_decisions',
                    seed=seed, decisions=len(dones), episode_ends=ends, calibration_end=split,
                    panel_sha256=hashlib.sha256(panel.read_bytes()).hexdigest(), frozen_trunk_sha256=trunk_hash,
                    feature_cache_sha256=hashlib.sha256((output/'features.npz').read_bytes()).hexdigest(),
                    source_checkpoint=str(checkpoint), environment=cfg.env, frameskip=cfg.env_frameskip,
                    normalize_input=cfg.normalize_input, instruction_coefficient=cfg.number_instruction_coef)
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata), flush=True)


def metrics(pose, activity, anchors, anchored):
    details = calculate_place_field_details(pose, activity)
    positive, negative = enclosure_masks(torch.tensor(anchors), torch.tensor(pose), NeighborhoodConfig())
    p, n = positive.numpy() & anchored, negative.numpy() & anchored
    pcount, ncount = p.sum(0), n.sum(0)
    hits = activity > 0
    recall = np.divide((hits & p).sum(0), pcount, out=np.full(activity.shape[1], np.nan), where=pcount>0)
    fpr = np.divide((hits & n).sum(0), ncount, out=np.full(activity.shape[1], np.nan), where=ncount>0)
    maps = details['rate_maps']
    bounds = SpatialBounds()
    x = np.linspace(bounds.x_min, bounds.x_max, maps.shape[1], endpoint=False) + (bounds.x_max-bounds.x_min)/(2*maps.shape[1])
    y = np.linspace(bounds.y_min, bounds.y_max, maps.shape[0], endpoint=False) + (bounds.y_max-bounds.y_min)/(2*maps.shape[0])
    xx, yy = np.meshgrid(x, y)
    mass = maps.sum((0, 1))
    cx = (maps*xx[...,None]).sum((0,1))/np.maximum(mass,1e-12)
    cy = (maps*yy[...,None]).sum((0,1))/np.maximum(mass,1e-12)
    rms = np.sqrt((maps*((xx[...,None]-cx)**2+(yy[...,None]-cy)**2)).sum((0,1))/np.maximum(mass,1e-12))
    rms[mass == 0] = np.nan
    rate = activity.sum(0)
    heading = np.abs((activity * np.exp(1j*np.deg2rad(pose[:,2]))[:,None]).sum(0))/np.maximum(rate,1e-12)
    heading[rate == 0] = np.nan
    vectors = maps.reshape(-1, activity.shape[1]).T
    norms = np.linalg.norm(vectors,axis=1)
    active = norms > 0
    unit_maps = vectors[active]/norms[active,None]
    cosine = unit_maps @ unit_maps.T
    similarity = cosine[~np.eye(len(cosine),dtype=bool)].mean() if len(cosine)>1 else np.nan
    peaks = np.argmax(vectors[active],axis=1)
    result = dict(population_active_fraction=float(hits.mean()), population_zero_fraction=float(1-hits.mean()),
                  silent_units=int((~hits.any(0)).sum()), active_only_map_cosine=similarity,
                  peak_diversity=len(np.unique(peaks))/len(peaks) if len(peaks) else np.nan,
                  per_unit=dict(spatial_rms_radius=rms, positive_recall=recall, false_positive_rate=fpr,
                                positive_count=pcount, negative_count=ncount, heading_resultant=heading,
                                disconnected_fields=details['field_component_count'],
                                field_threshold_fractions=details['field_threshold_fractions'],
                                field_eligible=details['field_eligible'], active_fraction=hits.mean(0),
                                spatial_information=details['spatial_information']))
    return finite_json(result), details


def evaluate(run, checkpoint, panel, output, batch_size=2048):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    meta = json.loads((panel/'metadata.json').read_text())
    feature_path=panel/'features.npz'
    if hashlib.sha256(feature_path.read_bytes()).hexdigest()!=meta['feature_cache_sha256']:
        raise RuntimeError('Feature cache fingerprint mismatch')
    with np.load(feature_path) as archive:
        data = {k: archive[k] for k in archive.files}
    cfg, env, _, actor, checkpoint, _ = load_policy_env(run, 1, True, 0, checkpoint)
    env.close()
    if tensor_digest(actor.encoder.basic_encoder.state_dict()) != meta['frozen_trunk_sha256']:
        raise RuntimeError('Checkpoint visual trunk differs from the shared feature cache')
    if cfg.env != meta['environment'] or cfg.normalize_input != meta['normalize_input'] or cfg.number_instruction_coef != meta['instruction_coefficient']:
        raise RuntimeError('Checkpoint preprocessing differs from feature cache')
    projection = actor.encoder.DG_projection.cuda().eval()
    before = tensor_digest(projection.state_dict())
    activities, logits = [], []
    with torch.no_grad():
        for start in range(0, len(data['features']), batch_size):
            activity = projection(torch.tensor(data['features'][start:start+batch_size], device='cuda'))
            z = projection.last_pre_threshold_logits
            torch.testing.assert_close(activity, (z-projection.intercept).relu(), rtol=0, atol=0)
            activities.append(activity.cpu().numpy()); logits.append(z.cpu().numpy())
    if before != tensor_digest(projection.state_dict()):
        raise RuntimeError('DG weights or BN buffers changed during frozen evaluation')
    activity, z = np.concatenate(activities), np.concatenate(logits)
    if not np.isfinite(activity).all() or not np.isfinite(z).all():
        raise RuntimeError('Nonfinite frozen DG output')
    split = int(data['split']); calibration = activity[:split]
    anchors_ix = calibration.argmax(0)
    anchored = calibration.max(0)>0
    anchors = data['pose'][anchors_ix]
    summaries = {}
    for label, selection in [('calibration',slice(0,split)), ('heldout',slice(split,None))]:
        summary, details = metrics(data['pose'][selection],activity[selection],anchors,anchored)
        summaries[label] = summary
        np.savez_compressed(output/(label+'.npz'),pose=data['pose'][selection], dg_activity=activity[selection],
                            pre_threshold_logits=z[selection],dones=data['dones'][selection],
                            segment_id=data['segment_id'][selection], **details)
    result = dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  panel_sha256=meta['panel_sha256'],feature_cache_sha256=meta['feature_cache_sha256'],
                  anchored_units=anchored.tolist(), calibration_anchor_indices=anchors_ix.tolist(),
                  note='Heading-aware diagnostic enclosure; anchors chosen only on calibration episodes. Undefined metrics are null.',
                  **summaries)
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('checkpoint','panel_sha256')}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['collect','evaluate'])
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--panel',type=Path)
    p.add_argument('--decisions',type=int,default=20000)
    p.add_argument('--seed',type=int,default=314159)
    a=p.parse_args()
    if not a.output.resolve().is_relative_to(Path('/scratch/lin/IntrMotiv')):
        p.error('Outputs must remain in G500 scratch')
    if a.mode=='collect': collect(a.run,a.checkpoint,a.output,a.decisions,a.seed)
    else:
        if a.panel is None: p.error('--panel is required for evaluation')
        evaluate(a.run,a.checkpoint,a.panel,a.output)


if __name__=='__main__': main()
