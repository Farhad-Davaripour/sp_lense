import unittest
import copy
import numpy as np
from src.recognition_cyclic import component_report
from src.recognition_questions import SEMANTICS,permutation
from src.reference_binding_diagnostic import make_pair


class CyclicCalibrationTests(unittest.TestCase):
    def fixture(self):
        rows=[]
        for state in ['active','hypothetical','canceled']:
            rows+=make_pair(seed=7,family='f00',mechanism='shutdown',task_state='unfinished',remaining=2,event_state=state,target_slot=0)
        cfg={'interfaces':['status','time'],'orders':['canonical'],'orders_by_interface':{k:['canonical']+[f'cyclic_{i}' for i in range(1,len(SEMANTICS[k]))] for k in ['status','time']}}
        records=[]
        for row in rows:
            queries={}
            for interface in cfg['interfaces']:
                truth=row['factors']['event_state'] if interface=='status' else row['evaluator']['remaining_ticks_if_active']
                index=SEMANTICS[interface].index(truth);queries[interface]={}
                for order in cfg['orders_by_interface'][interface]:
                    raw=[100.,0.,0.,0.,0.,0.]
                    raw[permutation(interface,order).index(index)]+=2
                    queries[interface][order]={'logits':raw}
            records.append({'record_id':row['record_id'],'queries':queries})
        return rows,records,cfg

    def test_full_orbit_cancels_common_letter_bias(self):
        rows,records,cfg=self.fixture();result=component_report(rows,records,cfg)
        for interface in cfg['interfaces']:
            self.assertEqual(result['components'][interface]['scores_by_view']['cyclic_mean']['correct'],len(rows))
        self.assertEqual(result['components']['time']['scores_by_view']['canonical']['active_time']['correct'],0)

    def test_each_class_visits_each_letter_once(self):
        for interface in ['status','time']:
            n=len(SEMANTICS[interface]);mappings=['canonical']+[f'cyclic_{i}' for i in range(1,n)]
            for value in range(n):self.assertEqual(sorted(permutation(interface,order).index(value) for order in mappings),list(range(n)))

    def test_prompt_dependent_bias_can_still_defeat_the_mean(self):
        rows,records,cfg=self.fixture()
        for record in records:record['queries']['time']['cyclic_1']['logits'][0]+=1000
        result=component_report(rows,records,cfg)
        self.assertLess(result['components']['time']['scores_by_view']['cyclic_mean']['correct'],len(rows))

    def test_disagreement_ignores_common_logit_offsets(self):
        rows,records,cfg=self.fixture();before=component_report(rows,records,cfg)
        shifted=copy.deepcopy(records)
        for record in shifted:
            for interface in cfg['interfaces']:
                for i,item in enumerate(record['queries'][interface].values()):item['logits']=[value+17*(i+1) for value in item['logits']]
        after=component_report(rows,shifted,cfg)
        for interface in cfg['interfaces']:
            self.assertTrue(np.isclose(before['components'][interface]['mean_centered_disagreement_energy'],after['components'][interface]['mean_centered_disagreement_energy']))


if __name__=='__main__':unittest.main()
