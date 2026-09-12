"""Controller gates layered on the existing full-system DG preflight audit."""
import argparse
import csv
import json
import hashlib
from pathlib import Path
import torch
from hpc_runs.audit_dg_capacity_preflight import audit as audit_parent
from hpc_runs.audit_navigation8_algorithm_screen_preflight import _events
from hpc_runs.intrmotiv_offpolicy.sf_transport import updates_due


def load_runtime_checkpoint(path):
    # Reuse the evaluator's exact allowlist for original NumPy scalar metadata.
    from sf_working_directories.IntrMotiv.evaluation.place_fields import load_checkpoint_dict
    return load_checkpoint_dict(path,torch.device('cpu'))


def reload_certificate_errors(baseline, certificate, mode, device):
    """Bind exact learner-init checks to the immutable checkpoint that was tested."""
    if certificate is None:return ['missing exact checkpoint reload certificate']
    errors=[]
    for key in ('exact_restore','model_and_buffers_exact','optimizer_exact','counters_exact'):
        if certificate.get(key) is not True:errors.append('checkpoint reload did not verify '+key)
    expected=dict(run=baseline['run'],checkpoint=baseline['baseline'],frames=baseline['env_steps'],
                  train_step=baseline['train_step'],controller=mode)
    for key,value in expected.items():
        if certificate.get(key)!=value:errors.append('reload certificate mismatch '+key)
    if not str(certificate.get('device','')).startswith('cuda' if device=='gpu' else 'cpu'):
        errors.append('reload certificate device mismatch')
    digest=hashlib.sha256()
    with Path(baseline['baseline']).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    if certificate.get('checkpoint_sha256')!=digest.hexdigest():
        errors.append('reload certificate checkpoint digest mismatch')
    return errors


def restart_errors(final, baseline, completed_reload=False):
    errors=[]
    for key in ('env_steps','train_step'):
        if final[key]<baseline[key] or (final[key]==baseline[key] and not completed_reload):
            errors.append('restart did not advance '+key)
    if 'session' not in baseline:return errors
    state=final['controller'];replay=state['replay']
    if replay['session']!=baseline['session']+1:
        errors.append('restart physical session did not advance exactly once')
    if not any(item['stream'][0]==replay['session'] for item in replay['rows']):
        errors.append('no replay collected in restarted physical session')
    for key in ('accepted','received'):
        if replay[key]<=baseline[key]:errors.append('restart did not preserve/advance '+key)
    for key in ('publication','fresh_dg_steps','fresh_graph_batches'):
        if state[key]<=baseline[key]:errors.append('restart did not preserve/advance '+key)
    for key in ('completed','main_positions','auxiliary_positions','target_at'):
        if state['clock'][key]<baseline['clock'][key]:errors.append('restart regressed '+key)
    if baseline['pending'] and replay['rejected'].get('restart_pending_tail')!=baseline['pending']:
        errors.append('restart did not discard incomplete physical tails')
    return errors


def audit(study, jobs, root, required_frames=2000000, restart_baselines=None, reload_certificate=None):
    result=audit_parent(study,jobs,root,required_frames)
    by_name={r['run']:r for r in result['runs']}
    with Path(jobs).open() as stream:job_rows=list(csv.DictReader(stream,delimiter='\t'))
    baselines={r['run']:r for r in json.loads(Path(restart_baselines).read_text())} if restart_baselines else None
    certificates=None
    if reload_certificate:
        if baselines is None:raise ValueError('Reload certificates require immutable restart baselines')
        document=json.loads(Path(reload_certificate).read_text())
        if document.get('schema')!='intrmotiv/checkpoint-reload/v1':raise ValueError('Unsupported reload certificate')
        certificates={r['run']:r for r in document['runs']}
    for job in job_rows:
        row=by_name[job['experiment'].removeprefix('00_')]
        directory=Path(root)/job['train_root']/job['experiment']
        cfg=json.loads((directory/'config.json').read_text())
        final=None
        if baselines is not None:
            checkpoints=sorted((directory/'checkpoint_p0').glob('checkpoint_*.pth'))
            if job['experiment'] not in baselines:row['errors'].append('missing restart baseline')
            elif checkpoints:
                final=load_runtime_checkpoint(checkpoints[-1])
                baseline=baselines[job['experiment']]
                reload_verified=False
                if certificates is not None:
                    certificate=certificates.get(job['experiment'])
                    errors=reload_certificate_errors(baseline,certificate,cfg.get('controller_learning','ppo'),cfg['device'])
                    row['errors'].extend(errors);reload_verified=not errors
                    row['exact_checkpoint_reload']=dict(verified=reload_verified,certificate=certificate)
                # A completed PPO preflight needs an exact reload, not extra
                # training beyond its bounded horizon. Require checkpoint-bound
                # model/buffer/optimizer/counter evidence for this case. All DG,
                # telemetry, horizon and counter-regression gates remain active.
                completed_reload=(cfg.get('controller_learning','ppo')=='ppo'
                                  and baseline['env_steps']>=required_frames and reload_verified)
                row['errors'].extend(restart_errors(final,baseline,completed_reload))
            row['passed']=not row['errors']
        if cfg.get('controller_learning','ppo')=='ppo':continue
        errors=row['errors']
        checkpoints=sorted((directory/'checkpoint_p0').glob('checkpoint_*.pth'))
        if not checkpoints:continue
        # The ordinary place-field evaluator uses this safe loader too.
        if final is None:final=load_runtime_checkpoint(checkpoints[-1])
        state=final.get('controller')
        if state is None:
            errors.append('missing complete controller checkpoint');row['passed']=False;continue
        clock=state['clock'];replay=state['replay']
        terminals=[item for item in replay.get('rows',[]) if item['terminated'] or item['truncated']]
        if not terminals:errors.append('physical episode end not exercised in retained replay')
        elif not any(item['successor_valid'] and item['successor'] is not None for item in terminals):
            errors.append('no certified terminal successors in real DMLab replay')
        debt=updates_due(replay['accepted'],cfg['controller_learning_starts'],
                         cfg['controller_decisions_per_update'],clock['completed'])
        if debt:errors.append(f'unpaid main update debt: {debt}')
        if clock['main_positions']!=clock['completed']*cfg['controller_td_positions']:
            errors.append('main TD accounting mismatch')
        if clock['target_at']<cfg['controller_target_updates']:
            errors.append('target refresh not exercised')
        if state['fresh_dg_steps']!=final['train_step']:
            errors.append('fresh DG optimizer count diverges from parent clock')
        if state['fresh_graph_batches']>state['fresh_dg_steps']:
            errors.append('graph processing multiplied beyond fresh steps')
        if cfg['controller_her'] and clock['auxiliary_positions']<=0:
            errors.append('no auxiliary HER positions learned')
        if not cfg['controller_her'] and clock['auxiliary_positions']:
            errors.append('HER positions in plain DDQN')
        if replay['received']<replay['accepted']:
            errors.append('physical interactions undercounted')
        if state['publication']!=int(final['model']['controller_q.publication_version']):
            errors.append('publication checkpoint inconsistent')
        values=_events(directory)
        def last(suffix):
            events=[e for tag,es in values.items() if tag.endswith('/'+suffix) for e in es]
            return float(max(events,key=lambda e:e.step).value) if events else None
        required=['main_loss','auxiliary_loss','main_td_positions','auxiliary_td_positions',
                  'physical_interactions','accepted_replay_decisions','actor_memory_rebuilds',
                  'actor_memory_version_failures','target_age','update_debt','her_compute_seconds',
                  'fresh_dg_steps','fresh_graph_batches']
        metrics={key:last('controller/'+key) for key in required}
        for key,value in metrics.items():
            if value is None:errors.append('missing controller dashboard scalar '+key)
        if (metrics['actor_memory_rebuilds'] or 0)<=0:errors.append('actor publication rebuild not exercised')
        if (metrics['actor_memory_version_failures'] or 0)>0:errors.append('actor memory version failure')
        row['controller']=dict(clock=clock,update_debt=debt,metrics=metrics)
        row['passed']=not errors
    result['passed']=all(row['passed'] for row in result['runs'])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study');p.add_argument('jobs');p.add_argument('train_root')
    p.add_argument('--required-frames',type=int,default=2000000);p.add_argument('--output',required=True)
    p.add_argument('--restart-baselines')
    p.add_argument('--reload-certificate')
    a=p.parse_args();result=audit(a.study,a.jobs,a.train_root,a.required_frames,a.restart_baselines,a.reload_certificate)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    raise SystemExit(0 if result['passed'] else 1)
