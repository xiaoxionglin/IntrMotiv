import unittest

from hpc_runs.graph_stabilized_recruitment import (
    GraphRecruitmentState,
    select_graph_evidence,
)


class GraphRecruitmentTests(unittest.TestCase):
    def test_history_uses_behavior_events_and_crosses_rollout_boundaries(self):
        state = GraphRecruitmentState(3, L=4, half_life_events=1)
        self.assertFalse(state.observe(0, 0))
        state.observe(None, None)
        self.assertTrue(state.observe(1, 0))
        self.assertGreater(state.confidence[0][1], 0.0)

    def test_gap_episode_reset_and_stale_generation(self):
        state = GraphRecruitmentState(3, L=2)
        state.observe(0, 0)
        state.observe(None, None)
        state.observe(None, None)
        self.assertFalse(state.observe(1, 0))  # gap is three, over L
        state.reset_episode()
        self.assertFalse(state.observe(2, 0))
        state.observe(0, 0)
        state.invalidate_row(0)
        self.assertFalse(state.record_transition(0, 1, 1, 0, 0))
        # The actor history also cannot bridge a structural reassignment.
        self.assertFalse(state.observe(1, 0))

    def test_half_life_and_birth_protection(self):
        state = GraphRecruitmentState(2, L=2, half_life_events=1)
        state.observe(0, 0)
        self.assertTrue(state.observe(1, 0))
        self.assertAlmostEqual(state.birth_support[0], 0.5)
        self.assertAlmostEqual(state.confidence[0][1], 1.0)
        state.observe(0, 0)
        self.assertAlmostEqual(state.confidence[0][1], 0.5)

    def test_configured_5k_and_10k_half_lives(self):
        for half_life, expected in ((5_000, 0.5), (10_000, 0.5)):
            state = GraphRecruitmentState(2, L=1, half_life_events=half_life)
            for _ in range(int(half_life)):
                self.assertTrue(state.record_transition(0, 1, 1, 0, 0))
            self.assertAlmostEqual(state.birth_support[0], expected, places=12)
            self.assertAlmostEqual(state.birth_support[1], expected, places=12)

    def test_isolated_only_after_birth_expiry(self):
        state = GraphRecruitmentState(2, L=1, half_life_events=1)
        self.assertIsNone(state.eligible_victim()[0])
        state.observe(0, 0)
        state.observe(1, 0)
        # Row 0 is connected, row 1 is connected; neither is eligible.
        self.assertIsNone(state.eligible_victim()[0])
        state.invalidate_row(1)
        state.birth_support[0] = 0.25
        row, reason = state.eligible_victim()
        self.assertEqual((row, reason), (0, "isolated"))

    def test_redundancy_is_mutual_and_chooses_lower_support(self):
        state = GraphRecruitmentState(3, L=4, redundancy_max_steps=4)
        state.birth_support = [0.0, 0.0, 1.0]
        state.confidence[0][1] = state.confidence[1][0] = 1.0
        state.elapsed[0][1] = state.elapsed[1][0] = 4.0
        self.assertEqual(state.redundant_losers(), {1})  # exact tie -> higher index
        state.confidence[0][2] = 0.8  # make unit 0 better supported
        self.assertEqual(state.redundant_losers(), {1})
        state.confidence[1][0] = 0.1  # one-way supported edge no longer qualifies
        self.assertNotIn(1, state.redundant_losers())

    def test_threshold_boundary_and_policy_preference(self):
        state = GraphRecruitmentState(2, L=4, connectivity_threshold=0.25)
        state.confidence[0][1] = state.confidence[1][0] = 0.25
        state.elapsed[0][1] = state.elapsed[1][0] = 4.0
        self.assertEqual(state.redundant_losers(), set())
        state.confidence[0][1] = state.confidence[1][0] = 0.25001
        self.assertEqual(state.redundant_losers(), {1})
        policy = [[0.0, 1.0], [1.0, 0.0]]
        passive = [[0.0, 0.0], [0.0, 0.0]]
        self.assertIs(select_graph_evidence(
            policy_graph_available=True,
            policy_confidence=policy, policy_elapsed=policy,
            passive_confidence=passive, passive_elapsed=passive,
        )[0], policy)
        self.assertIs(select_graph_evidence(
            policy_graph_available=False,
            policy_confidence=None, policy_elapsed=None,
            passive_confidence=passive, passive_elapsed=passive,
        )[0], passive)


if __name__ == "__main__":
    unittest.main()
