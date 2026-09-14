import unittest
import numpy as np
import run_f1_cached_v2 as runner
import f1_cached_selection_v2 as selector
from test_f1_cached_selection_v2 import base_pool

class ReportingContractTests(unittest.TestCase):
    def test_all_wrapper_splits_are_supported_and_reportable(self):
        frozen=selector.select(base_pool())
        self.assertEqual(runner.REPORT_SPLITS, (("original40",40),("added40",40),("combined80",80)))
        for name,n in runner.REPORT_SPLITS:
            ids=[name+str(i) for i in range(n)]
            labels=["SELF"]*(n//4)+["OTHER"]*(3*n//4)
            out=selector.recompute_split_metrics(frozen,name,ids,labels,np.zeros(n))
            self.assertEqual(out["case_count"],n)
            self.assertEqual(out["metrics"]["f1"],0.0)
            self.assertIsNone(out["metrics"]["precision"])

if __name__=="__main__":unittest.main()
