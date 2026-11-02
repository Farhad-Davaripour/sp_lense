import json,unittest
from pathlib import Path
from copy import deepcopy
from src.io_utils import read
from src.risk_transfer_data import dataset
from src.factual_comprehension_data import facts,questions,parse_answer,summary

class FactualComprehensionTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).parents[1];self.cfg=read(root/'config/native27_factual_comprehension_v1.json');self.rows=dataset(read(root/'config/native27_risk_transfer_v1.json'))
    def test_observation_parser_handles_current_targets_and_withdrawal(self):
        for row in self.rows:
            f=facts(row);self.assertEqual(f['own'],row['label'])
            self.assertEqual(f['other'],int(row['condition']=='peer'))
        revoked=next(r for r in self.rows if r['family']=='document_review' and r['label']==0)
        self.assertEqual(facts(revoked)['own'],0)
    def test_fixed_queries_keep_source_history_and_no_labels_in_question(self):
        q=questions(self.rows,self.cfg);self.assertEqual(len(q),304)
        self.assertEqual(sum(v['mode']=='role' for v in q),208);self.assertEqual(sum(v['mode']=='named' for v in q),96)
        for item in q:
            original=next(r for r in self.rows if r['state_id']==item['state_id'])
            self.assertEqual(item['messages'][1:-1],original['messages'][1:]);self.assertNotIn(item['query_id'],json.dumps(item['messages']))
        self.assertEqual(q,questions(self.rows,self.cfg))
    def test_answer_parser_keeps_unqualified_answers_unknown(self):
        for text,want in [('YES',1),(' no. ',0),('Yes\n',1),('YES because...',None),('YES NO',None),('{"answer":"YES"}',None)]:self.assertEqual(parse_answer(text,True),want)
        self.assertIsNone(parse_answer('YES',False))
    def test_named_controls_cannot_rescue_role_errors_or_unknowns(self):
        items=[{**q,'answer':q['expected']} for q in questions(self.rows,self.cfg)]
        self.assertTrue(summary(items,self.cfg)['role_capability_criterion_passed'])
        role=next(v for v in items if v['mode']=='role' and v['expected']==0);role['answer']=1
        self.assertFalse(summary(items,self.cfg)['role_capability_criterion_passed'])
        role['answer']=None
        self.assertIsNone(summary(items,self.cfg)['role_capability_criterion_passed'])

if __name__=='__main__':unittest.main()
