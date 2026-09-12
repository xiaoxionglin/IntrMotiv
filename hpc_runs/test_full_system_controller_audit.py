import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from hpc_runs.audit_full_system_controller_preflight import audit


class ControllerRuntimeAudit(unittest.TestCase):
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
            with patch('hpc_runs.audit_full_system_controller_preflight.audit_parent',return_value=parent),patch(
                    'hpc_runs.audit_full_system_controller_preflight._events',return_value={}):
                result=audit('unused',jobs,root)
            self.assertFalse(result['passed'])
            errors=result['runs'][0]['errors']
            self.assertTrue(any('unpaid main update debt' in e for e in errors))
            self.assertIn('no auxiliary HER positions learned',errors)
            self.assertIn('missing controller dashboard scalar main_loss',errors)


if __name__=='__main__':unittest.main()
