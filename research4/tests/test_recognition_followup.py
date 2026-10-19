import copy
import unittest

from src.recognition_followup import predictions, report
from src.recognition_questions import SEMANTICS, permutation
from src.reference_binding_diagnostic import make_pair


def fixture():
    rows = []
    for status in ['active', 'hypothetical', 'canceled']:
        rows += make_pair(seed=31, family='f00', mechanism='context_reset', task_state='unfinished', remaining=2, event_state=status, target_slot=1)
    records = []
    for row in rows:
        queries = {}
        answers = {'status': row['factors']['event_state'], 'target': 'current_process' if row['evaluator']['named_target_is_current'] else 'other_process', 'time': row['evaluator']['remaining_ticks_if_active']}
        for interface in SEMANTICS:
            queries[interface] = {}
            for order in ['canonical', 'reversed']:
                logits = [-10.] * 6
                index = permutation(interface, order).index(SEMANTICS[interface].index(answers[interface]))
                logits[index] = 4.
                queries[interface][order] = {'logits': logits}
        records.append({'record_id': row['record_id'], 'queries': queries})
    return rows, records


class FollowupCounterfactualTests(unittest.TestCase):
    def test_correct_mean_and_views_preserve_semantics(self):
        rows, records = fixture()
        result = report(rows, records)
        self.assertEqual(result['both_answer_mappings_correct_counts'], {'status':6, 'target':6, 'time':6})
        for view in result['scores_by_view'].values():
            self.assertEqual(view['pairs_with_both_bindings_correct'], 3)
            self.assertEqual(view['binary_metrics']['active_event_targets_current']['balanced_accuracy'], 1.)

    def test_constant_A_cannot_pass_answer_order_control(self):
        rows, records = fixture()
        constant = copy.deepcopy(records)
        for record in constant:
            for query in record['queries'].values():
                for value in query.values(): value['logits'] = [9, 0, 0, 0, 0, 0]
        result = report(rows, constant)
        self.assertEqual(result['semantic_answer_agreement_counts']['target'], 0)
        self.assertEqual(result['both_answer_mappings_correct_counts']['target'], 0)
        self.assertEqual(result['scores_by_view']['mean_semantic_logits']['binary_metrics']['reference_binding']['balanced_accuracy'], .5)
        del constant[0]['queries']['target']['reversed']
        with self.assertRaises(ValueError): predictions(rows, constant, 'canonical')


if __name__ == '__main__':
    unittest.main()
