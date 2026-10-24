import unittest
from src.composite_task_skill_full_fit import full_fit_plan
from src.composite_task_skill_model_audit import TaskQualification


class FullTaskFitBoundaryTests(unittest.TestCase):
    def test_caller_assertions_do_not_promote_a_full_fit(self):
        for supplied in [True, {'complete_training_qualification_audited':True,'owned_compute_released':True}, None]:
            with self.assertRaises(PermissionError):full_fit_plan('a'*40,supplied)
        with self.assertRaises(PermissionError):TaskQualification({'scientific_qualification_admitted':True},key=True)


if __name__=='__main__':unittest.main()
