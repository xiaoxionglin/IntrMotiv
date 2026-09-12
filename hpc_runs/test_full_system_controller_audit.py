import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from hpc_runs.audit_full_system_controller_preflight import audit, restart_errors


class ControllerRuntimeAudit(unittest.TestCase):
    def test_completed_ppo_requires_exact_checkpoint_bound_reload_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'00_test';(run/'checkpoint_p0').mkdir(parents=True)
            (run/'config.json').write_text(json.dumps(dict(controller_learning='ppo',device='gpu')))
            checkpoint=run/'checkpoint_p0/checkpoint_001.pth'
            torch.save(dict(env_steps=2000000,train_step=10),checkpoint)
            baseline=dict(run='00_test',baseline=str(checkpoint),env_steps=2000000,train_step=10)
            baseline_path=root/'baselines.json';baseline_path.write_text(json.dumps([baseline]))
            certificate=dict(run='00_test',checkpoint=str(checkpoint),frames=2000000,train_step=10,
                controller='ppo',device='cuda:0',exact_restore=True,model_and_buffers_exact=True,
                optimizer_exact=True,counters_exact=True,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            jobs=root/'jobs.tsv';jobs.write_text('experiment\ttrain_root\n00_test\t.\n')
            for proof,passed in ((None,False),(certificate,True),
                                 ({**certificate,'optimizer_exact':False},False),
                                 ({**certificate,'checkpoint_sha256':'incorrect'},False)):
                path=None
                if proof is not None:
                    path=root/'reload.json';path.write_text(json.dumps(dict(schema='intrmotiv/checkpoint-reload/v1',runs=[proof])))
                parent=dict(passed=True,runs=[dict(run='test',errors=[],passed=True)])
                with patch('hpc_runs.audit_full_system_controller_preflight.load_runtime_checkpoint',side_effect=lambda p:torch.load(p,weights_only=True)),patch(
                        'hpc_runs.audit_full_system_controller_preflight.audit_parent',return_value=parent):
                    result=audit('unused',jobs,root,restart_baselines=baseline_path,reload_certificate=path)
                self.assertEqual(result['passed'],passed)

    def test_restart_requires_new_physical_session_and_preserved_clocks(self):
        baseline=dict(env_steps=100,train_step=2,session=0,accepted=20,received=21,
                      publication=1,fresh_dg_steps=2,fresh_graph_batches=1,pending=1,
                      clock=dict(completed=3,main_positions=12,auxiliary_positions=0,target_at=2))
        state=dict(replay=dict(session=1,accepted=40,received=42,rows=[dict(stream=(1,0))],
                               rejected=dict(restart_pending_tail=1)),
                   publication=2,fresh_dg_steps=4,fresh_graph_batches=2,clock=baseline['clock'].copy())
        final=dict(env_steps=200,train_step=4,controller=state)
        self.assertEqual(restart_errors(final,baseline),[])
        state['replay']['session']=0
        state['clock']['completed']=0
        errors=restart_errors(final,baseline)
        self.assertIn('restart physical session did not advance exactly once',errors)
        self.assertIn('restart regressed completed',errors)

    def test_reports_main_debt_and_missing_auxiliary_without_passing_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'00_test';(run/'checkpoint_p0').mkdir(parents=True)
            cfg=dict(controller_learning='ddqn',controller_learning_starts=10,controller_decisions_per_update=2,
                     controller_td_positions=4,controller_target_updates=1,controller_her=True)
            (run/'config.json').write_text(json.dumps(cfg))
            state=dict(clock=dict(completed=1,main_positions=4,auxiliary_positions=0,target_at=1),
                       replay=dict(accepted=16,received=16),fresh_dg_steps=2,fresh_graph_batches=1,publication=2)
            torch.save(dict(controller=state,train_step=2,model={'controller_q.publication_version':torch.tensor(2)}),
                       run/'checkpoint_p0/checkpoint_001.pth')
            jobs=root/'jobs.tsv';jobs.write_text('experiment\ttrain_root\n00_test\t.\n')
            parent=dict(passed=True,runs=[dict(run='test',errors=[],passed=True)])
            with patch('hpc_runs.audit_full_system_controller_preflight.load_runtime_checkpoint',side_effect=lambda p:torch.load(p,weights_only=True)),patch('hpc_runs.audit_full_system_controller_preflight.audit_parent',return_value=parent),patch(
                    'hpc_runs.audit_full_system_controller_preflight._events',return_value={}):
                result=audit('unused',jobs,root)
            self.assertFalse(result['passed'])
            errors=result['runs'][0]['errors']
            self.assertTrue(any('unpaid main update debt' in e for e in errors))
            self.assertIn('no auxiliary HER positions learned',errors)
            self.assertIn('missing controller dashboard scalar main_loss',errors)


if __name__=='__main__':unittest.main()
