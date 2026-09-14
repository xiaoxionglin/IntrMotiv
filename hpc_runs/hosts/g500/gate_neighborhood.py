"""Fail-closed checkpoint and shared-panel qualification for the DG study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from hpc_runs.intrmotiv_study.direct import source_digest, atomic_json
from hpc_runs.intrmotiv_study.spatial_contract import load_spatial_snapshot
from sf_working_directories.IntrMotiv.evaluation.place_fields import load_checkpoint_dict, load_policy_env


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite_tensors(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tensors(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tensors(v) for v in value)
    return True


def qualify(output, panel, evaluation_root):
    audit = output/'direct_execution'
    manifest = json.loads((audit/'manifest.json').read_text())
    states = json.loads((audit/'state.json').read_text())
    require(len(states)==4 and all(s['status']=='completed' for s in states), 'All four preflights must finish successfully')
    require(source_digest(manifest['source_root']) == manifest['source_sha256'], 'Training source changed')
    reference = None
    results=[]
    panel_meta=json.loads((panel/'metadata.json').read_text())
    for spec,state in zip(manifest['runs'],states):
        require(spec['name']==state['name'], 'Run-state alignment differs')
        run=output/spec['name']
        cfg=json.loads((run/'config.json').read_text())
        for key in ('study_sha256','source_sha256','manifest_sha256'):
            require(cfg.get('intrmotiv_'+key)==manifest[key], f'Missing saved provenance: {key}')
        require(cfg['ppo_dg_gradient']=='stop' and cfg['DG_BN_intercept']==2.43 and cfg['Hippo_n_feature']==16, 'Sparse STOP physiology changed')
        require(cfg['dg_batchnorm_semantics']=='legacy_batch', 'BN semantics changed')
        checkpoints=list(run.glob('checkpoint_p0/*.pth'))
        initial_path=next(p for p in checkpoints if p.name.startswith('initial_'))
        terminal_path=max((p for p in checkpoints if p.name.startswith('checkpoint_')),key=lambda p:int(p.stem.split('_')[-1]))
        initial=load_checkpoint_dict(initial_path,torch.device('cpu'))
        terminal=load_checkpoint_dict(terminal_path,torch.device('cpu'))
        require(terminal['env_steps']>=spec['target_frames'], 'Terminal checkpoint is too early')
        require(initial['env_steps']==0, 'Initial checkpoint is not untrained')
        require(finite_tensors(terminal), 'Nonfinite model or optimizer checkpoint')
        if reference is None:
            reference=initial['model']
        require(reference.keys()==initial['model'].keys() and all(torch.equal(reference[k],v) for k,v in initial['model'].items()), 'Seed-paired initial models differ')
        frozen=[k for k in initial['model'] if k.startswith('encoder.basic_encoder.')]
        require(bool(frozen) and all(torch.equal(initial['model'][k],terminal['model'][k]) for k in frozen), 'Frozen visual trunk changed')
        dg='encoder.DG_projection.linear.weight'
        require(not torch.equal(initial['model'][dg],terminal['model'][dg]), 'DG weights did not learn')
        decoder=[k for k in initial['model'] if k.startswith('decoder.')]
        require(any(not torch.equal(initial['model'][k],terminal['model'][k]) for k in decoder), 'Controller decoder did not learn')
        if cfg['dg_objective']=='neighborhood':
            require(not any(cfg[k] for k in ('extra_encoder_losses','encoder_batch_loss','encoder_multi_activation_loss')), 'Old loss enabled in new arm')
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            diagnostics={}
            for event in run.rglob('events.out.tfevents.*'):
                accumulator=EventAccumulator(str(event),size_guidance={'scalars':0}).Reload()
                for tag in accumulator.Tags()['scalars']:
                    if 'dg_neighborhood_' in tag:
                        diagnostics.setdefault(tag,[]).extend(e.value for e in accumulator.Scalars(tag))
            for key in ('self_loss','positive_count','negative_count'):
                values=diagnostics.get('train/dg_neighborhood_'+key,[])
                require(bool(values) and np.isfinite(values).all() and max(values)>0, 'Missing or invalid actual loss diagnostics: '+key)
            pair_counts=diagnostics.get('train/dg_neighborhood_pair_count',[])
            require(bool(pair_counts), 'Missing actual local-pair diagnostics')
            require(max(pair_counts)>0 if cfg['dg_local_repulsion']!='none' else max(pair_counts)==0, 'Wrong local-pair mode executed')
        # Reconstruct the real model and reload the real Adam state, preserving
        # the parameter order used by BaseLearner.init (all actor parameters).
        actor_cfg,env,_,actor,_,_=load_policy_env(run,1,True,0,terminal_path)
        env.close()
        require(actor_cfg.optimizer=='adam', 'Extend exact reload gate for this optimizer')
        optimizer=torch.optim.Adam(list(actor.parameters()),lr=actor_cfg.learning_rate)
        optimizer.load_state_dict(terminal['optimizer'])
        require(len(optimizer.state)>0 and finite_tensors(optimizer.state_dict()), 'Optimizer reload failed')
        for key,value in actor.state_dict().items():
            require(torch.equal(value,terminal['model'][key]), f'Exact model reload mismatch: {key}')
        spatial=list(Path(cfg['online_spatial_output_root']).glob(f"*/{spec['name']}/policy_*/*.npz"))
        require(len(spatial)>=2,'Missing milestone spatial snapshots')
        for path in spatial:
            snapshot=load_spatial_snapshot(path)
            require(str(np.asarray(snapshot['run_name']).item())==spec['name'], 'Telemetry run identity mismatch')
        evaluation=json.loads((evaluation_root/spec['name']/'summary.json').read_text())
        require(evaluation['panel_sha256']==panel_meta['panel_sha256'], 'Evaluation used a different trajectory')
        require(evaluation['checkpoint']==str(terminal_path), 'Evaluation checkpoint is not terminal')
        for split in ('calibration','heldout'):
            with np.load(evaluation_root/spec['name']/(split+'.npz')) as data:
                require(len(data['pose'])>0 and np.isfinite(data['pose']).all() and np.isfinite(data['dg_activity']).all(), 'Invalid held-out arrays')
        results.append(dict(name=spec['name'],checkpoint=str(terminal_path),env_steps=terminal['env_steps'],
                            train_step=terminal['train_step'],wandb_urls=state['wandb_urls'],
                            heldout=evaluation['heldout']))
        del actor,optimizer,terminal,initial
    report=dict(passed=True,study_sha256=manifest['study_sha256'],source_sha256=manifest['source_sha256'],
                manifest_sha256=manifest['manifest_sha256'],panel_sha256=panel_meta['panel_sha256'],runs=results)
    atomic_json(evaluation_root/'qualification.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path);p.add_argument('panel',type=Path);p.add_argument('evaluation_root',type=Path)
    a=p.parse_args(); print(json.dumps(qualify(a.output,a.panel,a.evaluation_root),indent=2))
