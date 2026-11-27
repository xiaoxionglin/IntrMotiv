"""Pure fake-environment tests: no DMLab import, installation or native execution."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from hpc_runs.first_hit_control.dmlab_evaluate import (
    EvaluationConfig, MatchedFirstHitEvaluator, execute_trial, install_callback, module_hash, run_on_nemo,
    verify_source_config,
)
from hpc_runs.first_hit_control.runtime_adapter import GoalLayout


class FakeActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Module()
        self.core.Hippo_n_feature = 3
        self.core.expanded_length = 1
        self.core.core_output_size = 3
        self.core.policy_graph = nn.Module()
        self.core.policy_graph.register_buffer("node_visits", torch.ones(3))
        self.register_buffer("logits", torch.tensor([[1., 0.]]))
        self.action_space = SimpleNamespace(n=2)
        self.inputs = []
        self.eval()

    def forward_tail(self, core, **kwargs):
        self.inputs.append(core.clone())
        return {"action_logits": self.logits}


class FakeAdapter(nn.Module):
    def __init__(self, horizon=3):
        super().__init__()
        self.layout = GoalLayout(width=8, goal_start=5, goals=3)
        self.horizon_scale = horizon
        self.q_heads = nn.Linear(1, 2)
        self.inputs, self.clocks = [], []
        self.eval()

    def forward(self, core, ids, clocks):
        self.inputs.append(core.clone())
        self.clocks.append(int(clocks.item()))
        return torch.tensor([[0., 1.]]) if int(ids.item()) == 2 else torch.tensor([[1., 0.]])


class FakeAPI(dict):
    """Fresh engine and recurrent-prefix callbacks with auditable call counts."""
    def __init__(self, *, cap=10000, physical_done=False, ambiguous=False):
        self.count, self.position, self.closed = 0, 0, False
        self.end, self.ambiguous = physical_done, ambiguous
        self.encodes = []
        actor = FakeActor()

        def fresh(seed):
            self.position = 0
            return 0, torch.zeros(1, 1)

        def advance(action):
            self.count += 1
            # Visiting goal 1 then goal 2 is possible; own-goal stop must prevent it.
            self.position = (2 if self.position == 1 else 1) if action == 0 else 2
            if self.end:
                self.position = 2  # Deliberately tempting autoreset observation.
            return self.position, self.end

        def encode(obs, state, command=None):
            self.encodes.append((obs, float(state.item()), command))
            dg = torch.zeros(1, 3)
            dg[0, obs] = 2
            if self.ambiguous:
                dg[0, (obs + 1) % 3] = 1
            core = torch.cat((dg, torch.tensor([[17., float(obs), 1., 0., 0.]])), dim=1)
            if command is not None:
                core[:, 5:] = 0
                core[0, command + 5] = 1
            return dg, core, state + 1

        def signature(obs, state):
            return {"obs": torch.tensor(obs), "state": state.clone()}

        def replay(seed, prefix):
            obs, state = fresh(seed)
            for action in prefix:
                _, _, state = encode(obs, state)
                obs, done = advance(action)
                if done:
                    raise RuntimeError("Prefix crossed episode boundary")
            return obs, state

        super().__init__(actor=actor, graph=actor.core.policy_graph, cfg=SimpleNamespace(),
            checkpoint="fake_checkpoint", device=torch.device("cpu"), decision_cap=cap,
            fresh=fresh, advance=advance, encode=encode, signature=signature, replay=replay,
            decisions=lambda: self.count, close=self.close)

    def close(self):
        self.closed = True

    def start(self, prefix=()):
        obs, state = self["replay"](17, prefix)
        return dict(source=obs, repeat=0, seed=17, prefix=list(prefix), signature=self["signature"](obs, state))


class DmlabEvaluateTests(unittest.TestCase):
    def test_own_command_stops_and_blocks_later_alternative_arrival(self):
        api, model = FakeAPI(), FakeAdapter()
        rows = execute_trial(api, model, policy="first_hit", start=api.start(), command=1, targets=[1, 2], horizon=3)
        self.assertTrue(rows[0]["hit"])
        self.assertFalse(rows[1]["hit"])
        self.assertTrue(rows[1]["stopped_at_other_command"])
        self.assertEqual(rows[1]["executed_actions"], [0])
        self.assertEqual(model.clocks, [3])

    def test_initial_command_stops_without_action_and_never_manufactures_lift(self):
        api, model = FakeAPI(), FakeAdapter()
        rows = execute_trial(api, model, policy="first_hit", start=api.start(), command=0, targets=[0, 2], horizon=3)
        self.assertTrue(rows[0]["initially_at_command"])
        self.assertEqual(rows[0]["hit_time"], 0)
        self.assertFalse(rows[1]["hit"])
        self.assertEqual(api.count, 0)

    def test_terminal_autoreset_never_scores_as_success(self):
        api, model = FakeAPI(physical_done=True), FakeAdapter()
        rows = execute_trial(api, model, policy="first_hit", start=api.start(), command=2, targets=[1, 2], horizon=3)
        self.assertTrue(all(r["censored"] and r["successor_unknown"] and not r["hit"] for r in rows))
        self.assertEqual(len(api.encodes), 1)

    def test_winner_primary_survives_multi_activation_exclusive_is_secondary(self):
        api, model = FakeAPI(ambiguous=True), FakeAdapter()
        rows = execute_trial(api, model, policy="first_hit", start=api.start(), command=1, targets=[1, 2], horizon=3)
        self.assertTrue(rows[0]["hit"])
        self.assertFalse(rows[0]["exclusive_hit"])
        self.assertEqual(rows[0]["elapsed"], 1)

    def test_raw_head_cannot_override_actual_processed_core_detector(self):
        api, model = FakeAPI(), FakeAdapter()
        original = api["encode"]
        def conflicting_raw(*args):
            _, core, state = original(*args)
            return torch.tensor([[0., 0., 99.]]), core, state
        api["encode"] = conflicting_raw
        rows = execute_trial(api, model, policy="first_hit", start=api.start(), command=1, targets=[1, 2], horizon=3)
        self.assertTrue(rows[0]["hit"])
        self.assertFalse(rows[1]["hit"])
        self.assertFalse(rows[1]["initially_at_target"])

    def test_full_core_preserves_physical_bypass_and_matches_both_routes(self):
        api, model = FakeAPI(), FakeAdapter()
        start = api.start()
        for policy in ("source", "first_hit"):
            execute_trial(api, model, policy=policy, start=start, command=2, targets=[1, 2], horizon=3)
        self.assertTrue(torch.equal(api["actor"].inputs[0], model.inputs[0]))
        self.assertEqual(model.inputs[0].tolist(), [[2., 0., 0., 17., 0., 0., 0., 1.]])
        self.assertEqual(module_hash(api["actor"]), module_hash(api["actor"]))

    def test_signature_mismatch_fails_before_acting(self):
        api, model = FakeAPI(), FakeAdapter()
        start = api.start()
        start["signature"]["state"] += 1
        with self.assertRaisesRegex(RuntimeError, "did not reproduce"):
            execute_trial(api, model, policy="first_hit", start=start, command=1, targets=[1, 2], horizon=3)
        self.assertEqual(api.count, 0)

    def test_recurrent_prefix_and_deadline_are_replayed_then_counted_down(self):
        api, model = FakeAPI(), FakeAdapter()
        start = api.start(prefix=(0,))
        rows = execute_trial(api, model, policy="first_hit", start=start, command=0, targets=[0, 2], horizon=3)
        self.assertEqual(model.clocks, [3, 2, 1])
        self.assertEqual(rows[0]["prefix_actions"], [0])
        self.assertEqual(rows[0]["elapsed"], 3)
        self.assertTrue(rows[0]["timeout"])
        self.assertEqual(rows[1]["hit_time"], 1)

    def test_full_pool_without_passive_edges_and_frozen_hashes(self):
        api, model = FakeAPI(), FakeAdapter()
        config = EvaluationConfig(horizon=3, discovery_episodes=1, prefix_cap=6, repeats=1, max_sources=3)
        rows, summary = MatchedFirstHitEvaluator(model, config)(api)
        self.assertEqual(summary["eligible_targets"], [0, 1, 2])
        self.assertEqual(summary["frozen_sha256_before"], summary["frozen_sha256_after"])
        self.assertTrue(api.closed)
        self.assertEqual(len(rows), 2 * 3 * 3 * 3)
        self.assertEqual(summary["outcomes"]["first_hit"]["declared_own_rows"], 9)
        self.assertEqual(summary["outcomes"]["first_hit"]["declared_nontrivial_own_rows"], 6)
        self.assertEqual(summary["outcomes"]["first_hit"]["n_lift_pairs"], 6)

    def test_missing_starts_and_censoring_keep_complete_denominators(self):
        api, model = FakeAPI(physical_done=True), FakeAdapter()
        config = EvaluationConfig(horizon=3, discovery_episodes=1, prefix_cap=6, repeats=2, max_sources=3)
        rows, summary = MatchedFirstHitEvaluator(model, config)(api)
        outcome = summary["outcomes"]["first_hit"]
        self.assertEqual(outcome["declared_own_rows"], 18)
        self.assertEqual(outcome["declared_nontrivial_own_rows"], 12)
        self.assertEqual(outcome["issued_nontrivial_own_rows"], 2)
        self.assertEqual(outcome["missing_nontrivial_own_rows"], 10)
        self.assertEqual(outcome["censored_nontrivial_own_rows"], 2)
        self.assertEqual(outcome["success_lower_bound_declared"], 0)
        self.assertEqual(outcome["success_upper_bound_declared"], 1)
        self.assertEqual(outcome["paired_lift_identification_lower"], -1)
        self.assertEqual(outcome["paired_lift_identification_upper"], 1)

    def test_preflight_no_discovered_start_still_has_declared_denominator(self):
        api, model = FakeAPI(cap=1), FakeAdapter()
        config = EvaluationConfig(horizon=3, discovery_episodes=1, prefix_cap=6, repeats=1, max_sources=1,
                                  phase="preflight", preflight_targets=(1, 2))
        rows, summary = MatchedFirstHitEvaluator(model, config)(api)
        self.assertEqual(summary["outcomes"]["first_hit"]["declared_nontrivial_own_rows"], 2)
        self.assertEqual(summary["outcomes"]["first_hit"]["missing_nontrivial_own_rows"], 2)
        self.assertFalse(any(row["issued"] for row in rows))

    def test_native_entry_refuses_before_imports_outside_slurm(self):
        from unittest.mock import patch
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "allocated NEMO"):
                run_on_nemo(SimpleNamespace())

    def test_collection_config_and_effective_overrides_are_bound(self):
        import hashlib
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text('{"env_frameskip": 8, "obs_scale": 255}')
            overrides = {"dmlab_level_cache_path": "/work/collect/cache", "dmlab_runfiles_path": "/work/assets",
                         "transfer_freeze_dg": True}
            metadata = {"source_config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "collection_evaluation_overrides": overrides}
            evaluation = dict(overrides, dmlab_level_cache_path="/work/eval/cache")
            verify_source_config(path, metadata, evaluation)
            with self.assertRaisesRegex(ValueError, "overrides differ"):
                verify_source_config(path, metadata, dict(evaluation, transfer_freeze_dg=False))
            with self.assertRaisesRegex(ValueError, "overrides differ"):
                verify_source_config(path, metadata, dict(evaluation, dmlab_runfiles_path="/work/other-assets"))
            path.write_text('{"env_frameskip": 4, "obs_scale": 255}')
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                verify_source_config(path, metadata, evaluation)

    def test_mutating_actor_is_rejected(self):
        api, model = FakeAPI(), FakeAdapter()
        original = api["encode"]
        def mutate(*args):
            api["graph"].node_visits.add_(1)
            return original(*args)
        api["encode"] = mutate
        with self.assertRaisesRegex(RuntimeError, "mutated"):
            MatchedFirstHitEvaluator(model, EvaluationConfig(horizon=3, discovery_episodes=1, prefix_cap=3,
                repeats=1, max_sources=3))(api)
        self.assertTrue(api.closed)

    def test_archived_patch_only_dispatches_without_changing_engine_helpers(self):
        import os
        relative = Path("sf_working_directories/IntrMotiv/evaluation/target_control_interventions.py")
        roots = [Path(os.environ.get("INTRMOTIV_RUNTIME_ROOT", ".")), Path("/home/xiaoxiong/SFgit/SF_hipposlam")]
        path = next((root / relative for root in roots if (root / relative).is_file()), None)
        if path is None:
            self.skipTest("Set INTRMOTIV_RUNTIME_ROOT to audit the actual canonical source")
        source = path.read_text()
        patched = install_callback(source)
        def nested(text):
            function = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)
                            and n.name == "run_landmark_matched_interventions")
            return {n.name: ast.dump(n) for n in function.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(nested(source), nested(patched))
        self.assertIn("first_hit_evaluator", patched)
        with self.assertRaises(ValueError):
            install_callback(patched)
        with self.assertRaises(ValueError):
            install_callback(source.replace("def replay(seed, prefix):", "def changed(seed, prefix):"))

    def test_full_scope_cannot_hide_target_or_source_truncation(self):
        with self.assertRaises(ValueError):
            EvaluationConfig(preflight_targets=(1, 2))
        api, model = FakeAPI(), FakeAdapter()
        with self.assertRaisesRegex(ValueError, "silently cap"):
            MatchedFirstHitEvaluator(model, EvaluationConfig(horizon=3, discovery_episodes=1, prefix_cap=3,
                repeats=1, max_sources=1))(api)


if __name__ == "__main__":
    unittest.main()
