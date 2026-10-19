import unittest

from src.io_utils import digest
from src.recognition_capacity import binding_queries
from src.recognition_capacity_audit import numeric_counts,check_numeric
from src.reference_binding_diagnostic import make_pair


class Tokenizer:
    def apply_chat_template(self,*args,**kwargs):return [1,2]
    def decode(self,ids,**kwargs):return '2'


class CapacityTests(unittest.TestCase):
    def test_question_wording_and_role_pairs_are_all_retained(self):
        rows=make_pair(seed=7,family='f00',mechanism='shutdown',task_state='unfinished',remaining=2,event_state='active',target_slot=0)
        queries=binding_queries(rows)
        self.assertEqual(len(queries),8)
        self.assertEqual({q['interface'] for q in queries},{'target','target_named'})
        self.assertEqual({q['order'] for q in queries},{'canonical','reversed'})
        self.assertTrue(all(q['prompt_sha256']==digest(q['messages']) for q in queries))
        self.assertTrue(all('evaluator' not in str(q['messages']) for q in queries))

    def test_duplicate_queries_are_rejected(self):
        rows=make_pair(seed=7,family='f00',mechanism='shutdown',task_state='unfinished',remaining=2,event_state='active',target_slot=0)
        with self.assertRaises(ValueError):binding_queries(rows+rows)

    def test_truncation_cannot_count_as_numeric_success(self):
        cases=[{'task_family':'arithmetic','expected_integer':2}]
        values=numeric_counts(cases,[{'finished_with_eos':False,'decoded_text':'2'}])['all']
        self.assertEqual(values['strict_complete_correct'],0);self.assertEqual(values['truncated'],1)

    def test_first_and_last_parses_are_both_reported(self):
        cases=[{'task_family':'arithmetic','expected_integer':2}]
        values=numeric_counts(cases,[{'finished_with_eos':True,'decoded_text':'14 then 2'}])['all']
        self.assertEqual(values['strict_complete_correct'],0);self.assertEqual(values['first_integer_complete_correct'],0);self.assertEqual(values['last_integer_complete_correct'],1)

    def test_numeric_audit_rejects_early_non_eos_stop(self):
        case={'case_id':'x','messages':[{'role':'user','content':'test'}]}
        row={'case_id':'x','prompt_sha256':digest(case['messages']),'prompt_tokens':2,'input_token_sha256':digest([1,2]),'generated_token_ids':[7],'finished_with_eos':False}
        with self.assertRaisesRegex(ValueError,'Premature truncation'):check_numeric(case,row,Tokenizer(),{99})

    def test_native_eos_cannot_appear_before_last_token(self):
        case={'case_id':'x','messages':[{'role':'user','content':'test'}]}
        row={'case_id':'x','prompt_sha256':digest(case['messages']),'prompt_tokens':2,'input_token_sha256':digest([1,2]),'generated_token_ids':[99,7],'finished_with_eos':False}
        with self.assertRaisesRegex(ValueError,'Invalid native stopping'):check_numeric(case,row,Tokenizer(),{99})


if __name__=='__main__':unittest.main()
