"""Functional tests of finite-horizon target semantics and evidence boundaries."""

import unittest

import numpy as np

from hpc_runs.first_hit_control.finite import (
    FiniteMDP, collect_one_step_data, continuing_manager_q, evaluate_goal_policy,
    exhaustive_oracle, first_hit_q, fit_transition_model, greedy_policy,
    make_alias_witness, make_fork_cycle, make_stochastic_witness,
)


class FirstHitFiniteTests(unittest.TestCase):
    def test_one_step_data_stitches_unseen_long_routes(self):
        env = make_fork_cycle()
        data = collect_one_step_data(env, 8, 3)
        self.assertEqual(len(set(data["episode_id"])), len(data["state"]))
        self.assertTrue(data["truncated"].all())
        self.assertFalse(data["terminated"].any())
        p, support = fit_transition_model(env, data)
        np.testing.assert_array_equal(p, env.transition)
        # Goal 6 is three decisions away from state 0, absent from every
        # collected trajectory, but composable from the local edge evidence.
        q = first_hit_q(env, p, 3, support)
        self.assertEqual(q[2, 1, 0].max(), 0)
        self.assertEqual(q[3, 1, 0].max(), 1)
        result = evaluate_goal_policy(env, greedy_policy(q), 3)
        self.assertEqual(result["success"][1, 1, 0], 1)
        self.assertEqual(result["latency_mass"][1, 1, 0], 3)

    def test_exact_backups_match_independent_recursive_oracle(self):
        for env in (make_fork_cycle(), make_stochastic_witness(), make_alias_witness()):
            for horizon in range(7):
                q = first_hit_q(env, env.transition, horizon)
                np.testing.assert_allclose(q[horizon].max(-1), exhaustive_oracle(env, horizon), atol=1e-14)
                evaluation = evaluate_goal_policy(env, greedy_policy(q), horizon)
                actual = np.stack([evaluation["success"][g, g] for g in range(len(env.goal_states))])
                np.testing.assert_allclose(actual, exhaustive_oracle(env, horizon), atol=1e-14)

    def test_success_on_terminal_successor_precedes_physical_failure(self):
        env = make_stochastic_witness()
        q = first_hit_q(env, env.transition, 2)
        np.testing.assert_allclose(q[1, 0, 0], [0.1, 0.0])
        np.testing.assert_allclose(q[2, 0, 0], [0.1, 1.0])
        result = evaluate_goal_policy(env, greedy_policy(q), 2)
        self.assertEqual(result["success"][0, 0, 0], 1)
        self.assertEqual(result["latency_mass"][0, 0, 0], 2)

    def test_horizon_conditioning_changes_optimal_action(self):
        env = make_stochastic_witness()
        q = first_hit_q(env, env.transition, 2)
        self.assertEqual(int(q[1, 0, 0].argmax()), 0)
        self.assertEqual(int(q[2, 0, 0].argmax()), 1)

    def test_outcome_frequencies_retain_stochastic_failures(self):
        env = make_stochastic_witness()
        for seed in (8, 99, 123):
            data = collect_one_step_data(env, seed, 100)
            p, support = fit_transition_model(env, data)
            risky = (data["state"] == 0) & (data["action"] == 0)
            self.assertEqual(risky.sum(), 100)
            self.assertEqual(p[0, 0, 2], (data["next_state"][risky] == 2).mean())
            self.assertGreater(p[0, 0, 3], 0)
            q = first_hit_q(env, p, 2, support)
            self.assertEqual(int(q[2, 0, 0].argmax()), 1)

    def test_missing_support_never_invents_a_transition(self):
        env = make_fork_cycle()
        p, support = fit_transition_model(env, collect_one_step_data(env, 99, 4, [(0, 1)]))
        self.assertFalse(support[0, 1])
        self.assertTrue((p[0, 1] == 0).all())
        q = first_hit_q(env, p, 8, support)
        self.assertTrue(np.isneginf(q[1:, :, 0, 1]).all())
        self.assertEqual(q[8, 1, 0].max(), 0)
        self.assertEqual(exhaustive_oracle(env, 8)[1, 0], 1)
        evaluation = evaluate_goal_policy(env, greedy_policy(q), 8, support)
        self.assertEqual(evaluation["success"][1, 1, 0], 0)
        self.assertEqual(evaluation["unsupported_action_probability"][1, 1, 0], 0)

    def test_executed_alternative_commands_have_full_cross_matrix(self):
        env = make_fork_cycle()
        q = first_hit_q(env, env.transition, 3)
        values = evaluate_goal_policy(env, greedy_policy(q), 3)["success"]
        self.assertEqual(values.shape, (2, 2, 7))
        self.assertEqual(values[0, 0, 0], 1)
        self.assertEqual(values[1, 1, 0], 1)
        self.assertEqual(values[1, 0, 0], 0)
        self.assertEqual(values[0, 1, 0], 0)

    def test_continuing_retargeting_is_a_distinct_objective(self):
        env = make_fork_cycle()
        fixed = first_hit_q(env, env.transition, 8)
        continuing = continuing_manager_q(env, env.transition, 8)
        first_success = evaluate_goal_policy(env, greedy_policy(fixed), 8)["success"][1, 1, 0]
        continuing_success = evaluate_goal_policy(env, greedy_policy(continuing), 8)["success"][1, 1, 0]
        self.assertEqual(first_success, 1)
        self.assertEqual(continuing_success, 0)

    def test_alternative_stops_when_its_own_command_is_achieved(self):
        env = make_fork_cycle()
        # Force a route from 0 through left goal 5 and eventually right goal 6.
        # Evaluating right while commanding left must stop at 5, despite the
        # later right arrival that would occur if the command were held alive.
        q = first_hit_q(env, env.transition, 12)
        policy = greedy_policy(q)
        policy[:, 0] = policy[:, 1]
        policy[12, 0, 0] = [1, 0, 0]
        policy[11, 0, 1] = [1, 0, 0]
        policy[10, 0, 3] = [1, 0, 0]
        result = evaluate_goal_policy(env, policy, 12)
        self.assertEqual(result["success"][0, 1, 0], 0)
        self.assertEqual(result["success"][0, 1, 5], 0)

    def test_nontrivial_alternative_pair_denominator(self):
        env = make_fork_cycle()
        pairs = [(s, g) for s in env.start_states for g in range(len(env.goal_states))
                 if s != env.goal_states[g]]
        lift_pairs = [(s, g) for s, g in pairs if any(
            c != g and s != env.goal_states[c] for c in range(len(env.goal_states)))]
        self.assertEqual(len(pairs), 12)
        self.assertEqual(len(lift_pairs), 10)

    def test_post_audit_fixed_policy_successor_counterexample(self):
        # Added after the original fork comparison tied. This is an analytic
        # counterexample, not a newly selected result in the original suite.
        p = np.zeros((4, 2, 4))
        p[0, 0, 2], p[0, 0, 3], p[0, 1, 1] = 0.75, 0.25, 1
        p[1, 0, 2], p[1, 1, 3] = 1, 1
        p[2, :, 2], p[3, :, 3] = 1, 1
        env = FiniteMDP("post_audit_successor_witness", p,
                        np.array([False, False, True, True]), (2,), (0,))
        optimal = first_hit_q(env, p, 2)
        fixed = first_hit_q(env, p, 2, behavior_policy=np.full((4, 2), 0.5))
        np.testing.assert_allclose(optimal[2, 0, 0], [0.75, 1])
        np.testing.assert_allclose(fixed[2, 0, 0], [0.75, 0.5])
        self.assertEqual(evaluate_goal_policy(env, greedy_policy(optimal), 2)["success"][0, 0, 0], 1)
        self.assertEqual(evaluate_goal_policy(env, greedy_policy(fixed), 2)["success"][0, 0, 0], 0.75)

    def test_aliasing_places_half_success_bound_on_shared_policy(self):
        env = make_alias_witness()
        for action in (0, 1):
            policy = np.zeros((2, 1, 4, 2))
            policy[..., action] = 1
            outcomes = evaluate_goal_policy(env, policy, 1)["success"][0, 0, :2]
            self.assertEqual(outcomes.mean(), 0.5)
        np.testing.assert_array_equal(exhaustive_oracle(env, 1)[0, :2], [1, 1])

    def test_terminal_label_corruption_rejected(self):
        env = make_stochastic_witness()
        data = collect_one_step_data(env, 99, 5)
        data["terminated"][0] = not data["terminated"][0]
        with self.assertRaisesRegex(ValueError, "Physical terminal labels"):
            fit_transition_model(env, data)

    def test_unidentified_rows_rejected_without_support_mask(self):
        env = make_fork_cycle()
        p, support = fit_transition_model(env, collect_one_step_data(env, 8, 2, [(0, 1)]))
        with self.assertRaisesRegex(ValueError, "sum to one"):
            first_hit_q(env, p, 3)


if __name__ == "__main__":
    unittest.main()
