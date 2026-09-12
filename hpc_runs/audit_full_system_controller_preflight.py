"""Controller gates layered on the existing full-system DG preflight audit."""
import argparse
import csv
import json
from pathlib import Path
import torch
from hpc_runs.audit_dg_capacity_preflight import audit as audit_parent
from hpc_runs.audit_navigation8_algorithm_screen_preflight import _events
from hpc_runs.intrmotiv_offpolicy.sf_transport import updates_due


def load_runtime_checkpoint(path):
    # Reuse the evaluator's exact allowlist for original NumPy scalar metadata.
    from sf_working_directories.IntrMotiv.evaluation.place_fields import load_checkpoint_dict
    return load_checkpoint_dict(path,torch.device('cpu'))


def audit(study, jobs, root, required_frames=2000000):
    result=audit_parent(study,jobs,root,required_frames)
    by_name={r['run']:r for r in result['runs']}
    with Path(jobs).open() as stream:job_rows=list(csv.DictReader(stream,delimiter='\t'))
    for job in job_rows:
        row=by_name[job['experiment'].removeprefix('00_')]
        directory=Path(root)/job['train_root']/job['experiment']
        cfg=json.loads((directory/'config.json').read_text())
        if cfg.get('controller_learning','ppo')=='ppo':continue
        errors=row['errors']
        checkpoints=sorted((directory/'checkpoint_p0').glob('checkpoint_*.pth'))
        if not checkpoints:continue
        # The ordinary place-field evaluator uses this safe loader too.
        final=load_runtime_checkpoint(checkpoints[-1])
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
    a=p.parse_args();result=audit(a.study,a.jobs,a.train_root,a.required_frames)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    raise SystemExit(0 if result['passed'] else 1)
