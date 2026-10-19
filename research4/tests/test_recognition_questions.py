import unittest

import numpy as np

from src.recognition_pilot import messages as old_messages
from src.recognition_questions import SEMANTICS, decoded, messages, permutation, semantic_logits


class AnswerOrderTests(unittest.TestCase):
    def test_canonical_matches_already_frozen_prompt(self):
        observation = dict(current_process='proc-0123456789', other_process='proc-9876543210', clock=12, task_progress=1, task_required=3, notice='Test')
        for interface in SEMANTICS:
            self.assertEqual(messages(observation, interface, 'canonical'), old_messages(observation, interface))

    def test_every_semantic_answer_survives_relabeling(self):
        for interface, meanings in SEMANTICS.items():
            for index, meaning in enumerate(meanings):
                for order in ['canonical', 'reversed']:
                    logits = np.full(6, -10.)
                    logits[permutation(interface, order).index(index)] = 3.
                    self.assertEqual(decoded(logits, interface, order), meaning)

    def test_constant_A_is_not_semantically_invariant(self):
        scores = [9, 0, 0, 0, 0, 0]
        self.assertNotEqual(decoded(scores, 'target', 'canonical'), decoded(scores, 'target', 'reversed'))
        average = (semantic_logits(scores, 'target', 'canonical') + semantic_logits(scores, 'target', 'reversed')) / 2
        self.assertEqual(average[0], average[1])


if __name__ == '__main__':
    unittest.main()
