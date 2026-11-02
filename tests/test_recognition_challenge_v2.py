import unittest
from collections import defaultdict

from src.recognition_challenge_v2 import cases,parse_latest_subject


class MatchedChallengeTests(unittest.TestCase):
    def test_counts_and_identity_independence(self):
        reporter,revision=cases('reporter'),cases('revision')
        self.assertEqual(len(reporter),1152);self.assertEqual(len(revision),1728)
        names=lambda rows:{r['observation']['current_process'] for r in rows}
        self.assertFalse(names(reporter)&names(revision))
        self.assertEqual(len({r['matched_case_id'] for r in reporter}),288)
        self.assertEqual(len({r['matched_case_id'] for r in revision}),288)

    def test_latest_subject_changes_and_is_independent_of_text_order(self):
        for row in cases('revision'):
            if row['factors']['variant']=='simple':continue
            self.assertEqual(parse_latest_subject(row['observation']['notice']),row['evaluator']['named_target'])
            self.assertEqual(row['observation']['notice'].count('Subject process:'),2)

    def test_control_and_challenge_match_except_notice(self):
        groups=defaultdict(list)
        for row in cases('reporter'):groups[(row['matched_case_id'],row['factors']['current_slot'])].append(row)
        for group in groups.values():
            self.assertEqual(len(group),2)
            a,b=group
            self.assertEqual(a['evaluator'],b['evaluator'])
            self.assertEqual({k:v for k,v in a['observation'].items() if k!='notice'},{k:v for k,v in b['observation'].items() if k!='notice'})
            self.assertNotEqual(a['observation']['notice'],b['observation']['notice'])

    def test_hard_cases_balance_candidate_name_frequency(self):
        for cohort in ['reporter','revision']:
            for row in cases(cohort):
                if row['factors']['variant']=='simple':continue
                observed=row['observation'];notice=observed['notice']
                self.assertEqual(notice.count(observed['current_process']),1)
                self.assertEqual(notice.count(observed['other_process']),1)


if __name__=='__main__':unittest.main()
