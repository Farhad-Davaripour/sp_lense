import unittest
from src.composite_generalized_reporting import paired_group_summary


class PairedGroupTests(unittest.TestCase):
    def fixture(self):
        plan=[{'group_id':'g1','mechanism':'m'},{'group_id':'g2','mechanism':'m'}];records=[]
        for group in ['g1','g2']:
            for condition in ['own_pending','own_complete']:
                records.append({'group_id':group,'condition':condition,'arm':'base','success':False})
                records.append({'group_id':group,'condition':condition,'arm':'learned_dynamic','success':group=='g1'})
        return plan,records

    def test_four_cases_are_reported_as_two_paired_groups_without_binomial_interval(self):
        plan,records=self.fixture();value=paired_group_summary(records,plan)['families']['m']
        self.assertEqual(value['independent_groups'],2);self.assertEqual(value['permitted_own_cases'],4)
        self.assertEqual(value['primary_case_fraction'],0.5)
        self.assertEqual(value['paired_case_fraction_difference'],0.5)
        self.assertEqual(value['descriptive_between_group_difference_range'],[0.0,1.0])
        self.assertIsNone(value['population_confidence_interval']);self.assertFalse(value['binomial_case_interval_used'])

    def test_missing_or_unknown_pairs_do_not_become_zero(self):
        plan,records=self.fixture();records.pop(0)
        value=paired_group_summary(records,plan)['families']['m']
        self.assertEqual(value['primary_case_fraction'],0.5)
        self.assertIsNone(value['BASE_case_fraction']);self.assertIsNone(value['paired_case_fraction_difference'])
        self.assertIsNone(value['descriptive_between_group_difference_range'])
        self.assertEqual(value['missing_or_unknown_case_arm_outcomes'],1)


if __name__=='__main__':unittest.main()
