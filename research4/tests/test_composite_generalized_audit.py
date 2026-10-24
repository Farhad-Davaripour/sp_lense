import unittest
from copy import deepcopy
from unittest.mock import patch
from src.composite_generalized_audit import native_budget_audit,leakage_audit,audit_cohort,replay_reference_episode
from src.composite_generalized_world import fixture
from src.composite_generalized_teacher import teacher_episode


class Tokenizer:
    eos_token_id=2
    def __init__(self,prompt=8):self.prompt=prompt
    def apply_chat_template(self,*args,**kwargs):return [1]*self.prompt
    def encode(self,text,**kwargs):return [ord(c)+10 for c in text]
    def decode(self,ids,**kwargs):return ''.join(chr(i-10) for i in ids)


class GeneralizedAuditTests(unittest.TestCase):
    def test_exact_public_reference_replay_never_calls_teacher_or_repairs_evidence(self):
        episode=teacher_episode(fixture('own_pending','dependency_graph'))
        with patch('src.composite_generalized_teacher.expert',side_effect=AssertionError('Audit cannot call the expert')):
            self.assertEqual(replay_reference_episode(episode)['status'],'exact_reference_actions_replayed')
        changed=deepcopy(episode);changed['actions'][0]['reply']={'fabricated':True}
        with self.assertRaises(ValueError):replay_reference_episode(changed)
        changed=deepcopy(episode);changed['primitive_trace'][-1]['tick']+=1
        with self.assertRaises(ValueError):replay_reference_episode(changed)

    def test_native_overflow_is_recorded_without_truncating_or_repairing_rows(self):
        row={'split':'training','state_id':'synthetic','text':'short',
             'teacher_action':{'tool':'catalog','arguments':{}}}
        result=native_budget_audit([row],Tokenizer(6145),Tokenizer())
        self.assertEqual(result['status'],'failed_native_budget_admission')
        self.assertEqual(result['maxima']['qwen_context'],6145)
        self.assertFalse(result['truncated_or_repaired']);self.assertEqual(row['text'],'short')
        self.assertEqual(native_budget_audit([row],Tokenizer(),Tokenizer())['status'],'complete_native_budget_audit_pass')
        combined=native_budget_audit([row],Tokenizer(6100),Tokenizer())
        self.assertEqual(combined['status'],'failed_native_budget_admission')
        self.assertEqual({r['kind'] for r in combined['overflow']},{'supervised_combined_context'})
        development=native_budget_audit([{**row,'split':'development'}],Tokenizer(6100),Tokenizer())
        self.assertEqual(development['status'],'complete_native_budget_audit_pass')
        self.assertIsNone(development['maxima']['supervised_combined_context'])
        with self.assertRaises(PermissionError):native_budget_audit([{**row,'split':'sealed'}],Tokenizer(),Tokenizer())

    def test_identity_program_and_exact_text_overlap_are_failed_admission(self):
        training={'role':'training','case_ids':['a'],'group_ids':['ga'],'helper_structural_fingerprints':['program_a']}
        development={'role':'development','case_ids':['b'],'group_ids':['gb'],'helper_structural_fingerprints':['program_b']}
        left=[{'state_id':'s1','text':'public training fixture','sequence':0}]
        right=[{'state_id':'s2','text':'different public task fixture','sequence':0}]
        self.assertEqual(leakage_audit(training,development,left,right)['status'],'structural_and_exact_leakage_checks_passed')
        for mutation in ['case_ids','group_ids','helper_structural_fingerprints','text']:
            changed={**development}
            if mutation!='text':changed[mutation]=training[mutation]
            else:right=[{'state_id':'s2','text':left[0]['text'],'sequence':0}]
            self.assertEqual(leakage_audit(training,changed,left,right)['status'],'failed_leakage_admission')
        with self.assertRaises(PermissionError):leakage_audit(training,{**development,'role':'sealed'},left,right)

    def test_sealed_content_is_rejected_before_cohort_or_replay_access(self):
        with self.assertRaises(PermissionError):audit_cohort([],[],{'role':'sealed'},None)
        with self.assertRaises(PermissionError):replay_reference_episode({'scene':{'role':'sealed'}})


if __name__=='__main__':unittest.main()
