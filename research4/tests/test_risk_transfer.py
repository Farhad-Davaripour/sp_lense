from collections import Counter
import json,re,unittest
from pathlib import Path
from src.risk_transfer_data import dataset,balanced_snapshot

class RiskTransferTests(unittest.TestCase):
    def test_counterfactuals_have_equal_word_bags_and_opposite_labels(self):
        for family in ['report','inventory','document_review','access_log']:
            for state in ['withdrawn','none']:
                a=balanced_snapshot(60,family,True,state);b=balanced_snapshot(60,family,False,state)
                words=lambda r:Counter(re.findall(r'\w+',json.dumps(r['messages'])))
                self.assertEqual(words(a),words(b));self.assertEqual((a['label'],b['label']),(1,0))
                self.assertNotEqual(a['messages'],b['messages'])

    def test_transfer_is_fresh_and_has_104_observations(self):
        cfg=json.loads((Path(__file__).parents[1]/'config/native27_risk_transfer_v1.json').read_text())
        rows=dataset(cfg);self.assertEqual(len(rows),104);self.assertEqual(len({r['seed'] for r in rows}),20)
        self.assertEqual(Counter(r['family'] for r in rows),Counter(same_format=56,report=12,inventory=12,document_review=12,access_log=12))
        self.assertTrue(all(r['split']=='transfer' for r in rows));self.assertEqual(sum(r['label'] for r in rows),40)

if __name__=='__main__':unittest.main()
