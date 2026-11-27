"""GeoMem source reuse without changing action or sensory semantics."""

import unittest

import numpy as np

from hpc_runs.first_hit_control.geomem_tasks import (
    DEFAULT_GEOMEM, conditional_action_arrays, load_geomem_modules,
    memory_ambiguity_witnesses, task_document,
)


@unittest.skipUnless((DEFAULT_GEOMEM / "geomem-env/src/geomem_env/task_graphs.py").is_file(), "GeoMem source not present")
class FirstHitGeoMemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modules = load_geomem_modules()
        cls.tasks = modules["task_graphs"]
        cls.process = staticmethod(modules["graph_process"].graph_process_arrays)

    def fixture(self):
        task = self.tasks.TaskGraph("controlled contract", metadata={"representation": "test"})
        task.node("a", observation="same cue", pos=(0, 0), initial=True)
        task.node("b", observation="same cue", pos=(1, 0))
        task.node("goal", observation="goal", pos=(2, 0), terminal=True, success=True)
        task.edge("a", "b", "left")
        task.edge("a", "goal", "right", probability=0.25)
        task.edge("a", "b", "right", probability=0.75)
        task.edge("b", "goal", "left")
        return task_document(task)

    def test_conditional_actions_are_not_diluted_by_behavior_policy(self):
        document = self.fixture()
        process, metadata = self.process(document)
        graph = conditional_action_arrays(document, self.process)
        right = graph.action_labels.index("right")
        left = graph.action_labels.index("left")
        self.assertEqual(graph.transition[0, right, 2], 0.25)
        self.assertEqual(graph.transition[0, right, 1], 0.75)
        self.assertEqual(graph.transition[0, left, 1], 1)
        self.assertEqual(process["probability"][(process["source"] == 0) & (process["target"] == 2)].item(), 0.125)

    def test_unavailable_actions_stay_zero_and_masked(self):
        graph = conditional_action_arrays(self.fixture(), self.process)
        right = graph.action_labels.index("right")
        self.assertFalse(graph.available_actions[1, right])
        self.assertEqual(graph.transition[1, right].sum(), 0)
        self.assertFalse(graph.available_actions[2].any())
        self.assertEqual(graph.transition[2].sum(), 0)

    def test_sensory_goal_identity_does_not_reveal_latent_node(self):
        graph = conditional_action_arrays(self.fixture(), self.process)
        np.testing.assert_array_equal(graph.detector("same cue"), [True, True, False])
        self.assertNotEqual(graph.latent_states[0], graph.latent_states[1])
        self.assertEqual(graph.sensory_symbols[0], graph.sensory_symbols[1])
        # The public available-action mask is explicitly additional information.
        self.assertNotEqual(graph.observation_ids[0], graph.observation_ids[1])

    def test_event_and_reset_labels_are_never_agent_actions(self):
        for task in (self.tasks.delayed_match(delay=4), self.tasks.context_decision(delay=2), self.tasks.alternation(length=4)):
            with self.assertRaisesRegex(ValueError, "scheduler required"):
                conditional_action_arrays(task_document(task), self.process)

    def test_invalid_outcome_probability_is_rejected(self):
        document = self.fixture()
        document["edges"][0]["probability"] = 0.5
        with self.assertRaisesRegex(ValueError, "sum to one"):
            conditional_action_arrays(document, self.process)

    def test_terminal_outgoing_edge_is_rejected(self):
        document = self.fixture()
        document["edges"].append(dict(source=2, target=2, kind="action", action="left", probability=1, reward=0))
        with self.assertRaisesRegex(ValueError, "terminal node"):
            conditional_action_arrays(document, self.process)

    def test_delayed_cue_expires_from_finite_ca3(self):
        document = task_document(self.tasks.delayed_match(delay=4))
        process, _ = self.process(document)
        witnesses = memory_ambiguity_witnesses(document, process)
        self.assertEqual(len(witnesses), 2)
        for witness in witnesses:
            self.assertTrue(witness["prefixes_observably_distinct"])
            by_length = {row["memory_length"]: row["same_ca3"] for row in witness["finite_ca3_comparisons"]}
            self.assertTrue(by_length[4])
            self.assertFalse(by_length[8])
            self.assertEqual(witness["equal_context_memoryless_success_upper_bound"], 0.5)

    def test_alternation_initial_history_is_observationally_ambiguous(self):
        document = task_document(self.tasks.alternation(length=4))
        process, _ = self.process(document)
        witnesses = memory_ambiguity_witnesses(document, process)
        self.assertEqual(len(witnesses), 1)
        self.assertFalse(witnesses[0]["prefixes_observably_distinct"])
        self.assertTrue(all(row["same_ca3"] for row in witnesses[0]["finite_ca3_comparisons"]))
        self.assertTrue(witnesses[0]["reset_preserving_memory_can_distinguish"])


if __name__ == "__main__":
    unittest.main()
