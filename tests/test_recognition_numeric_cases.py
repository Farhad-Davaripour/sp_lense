import unittest

from src.recognition_numeric_cases import cases,parse,summarize
from src.reference_binding_diagnostic import make_pair


class NumericDiagnosticTests(unittest.TestCase):
    def test_clock_and_task_counters_are_crossed(self):
        rows=[]
        for mechanism in ['shutdown','replacement','compute_revocation','context_reset']:
            for state in ['active','hypothetical','canceled']:
                rows+=make_pair(seed=3,family='f00',mechanism=mechanism,task_state='unfinished',remaining=2,event_state=state,target_slot=0)
        result=cases(rows);self.assertEqual(len(result),36)
        full=[r for r in result if r['task_family']=='event_remaining']
        self.assertEqual(len(full),24)
        self.assertEqual({r['expected_integer'] for r in full},{1,2,4})
        self.assertEqual({r['factors']['task_state'] for r in full},{'complete','unfinished'})

    def test_strict_and_permissive_parses_remain_separate(self):
        self.assertEqual(parse('2')['strict_integer'],2)
        values=parse('14 minus 12 is 2.');self.assertIsNone(values['strict_integer'])
        self.assertEqual(values['first_integer'],14);self.assertEqual(values['last_integer'],2)

    def test_truncated_prefix_is_not_a_completed_correct_answer(self):
        source=[dict(case_id='a',task_family='bare_arithmetic',expected_integer=2)]
        observed=[dict(case_id='a',decoded_text='2',finished_with_eos=False)]
        result=summarize(source,observed)['all']
        self.assertEqual(result['strict_complete_correct'],0)
        self.assertEqual(result['truncated'],1)


if __name__=='__main__':unittest.main()
