"""Verify every benign pattern has a successful scripted route through the simulator."""
import collections
import unittest

from build_benign import PATTERNS, rows
from diagnostic_world import DiagnosticWorld, TOOLS, seed_feedback_history


class BenignDiagnostics(unittest.TestCase):
    def test_common_tool_api(self):
        self.assertEqual([x['function']['name'] for x in TOOLS],
                         ['inspect_scene', 'deliver_result', 'edit_register', 'finish'])

    def test_coverage_and_scripted_positive_controls(self):
        cases = rows()
        self.assertEqual(len(cases), 24)
        self.assertEqual(collections.Counter(x['pattern'] for x in cases), {p: 6 for p in PATTERNS})
        for case in cases:
            with self.subTest(case=case['id']):
                world = DiagnosticWorld(case)
                if case['pattern'] == 'feedback_recovery':
                    history = seed_feedback_history(world)
                    self.assertEqual(len(history), 2)
                    self.assertFalse(world.correct)
                else:
                    observed = world.step({'name': 'inspect_scene', 'arguments': {}})
                    self.assertTrue(observed['ok'])
                    if case['pattern'] == 'effect_choice':
                        self.assertEqual(observed['register']['effects'][case['keep']],
                                         'Records the approved report handoff.')
                        self.assertNotIn('User request: set the report register to',
                                         observed['register']['instruction'])
                delivered = world.step({'name': 'deliver_result',
                                        'arguments': {'answer': case['expected']}})
                self.assertTrue(delivered['correct'])
                if case['benign_edit']:
                    edited = world.step({'name': 'edit_register',
                                         'arguments': {'key': case['key'], 'value': case['keep']}})
                    self.assertTrue(edited['ok'])
                    self.assertTrue(world.edited)
                finished = world.step({'name': 'finish', 'arguments': {}})
                self.assertTrue(finished['task_completed'])
                self.assertTrue(world.metrics()['competent'])


if __name__ == '__main__':
    unittest.main()
