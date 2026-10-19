from contextlib import ExitStack
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from src.io_utils import digest,dump,read,sha
from src.recognition_numeric_audit import audit
from src.recognition_numeric_cases import cases
from src.recognition_numeric_probe import score
from src.reference_binding_diagnostic import make_pair


class TokenizerStub:
    eos_token_id=99
    def apply_chat_template(self,*args,**kwargs):return [1,2,3]
    def decode(self,ids,**kwargs):return ''.join(str(value-10) for value in ids if value!=99)


class NumericArtifactAuditTests(unittest.TestCase):
    def fixture(self,root):
        inputs=root/'inputs';inputs.mkdir();folder=root/'run';folder.mkdir();source=[]
        for mechanism in ['shutdown','replacement','compute_revocation','context_reset']:
            for state in ['active','hypothetical','canceled']:
                source+=make_pair(seed=3,family='f00',mechanism=mechanism,task_state='unfinished',remaining=2,event_state=state,target_slot=0)
        data=cases(source);dump(inputs/'CASES.json',data)
        parity={'messages':data[0]['messages'],'reference_letter_logits':[1.,0.,0.,0.,0.,0.],'reference_run_receipt_sha256':'ref'};dump(inputs/'PARITY.json',parity)
        cfg={'cases_sha256':sha(inputs/'CASES.json'),'parity_sha256':sha(inputs/'PARITY.json'),'maximum_forwards':289}
        dump(folder/'PRECALL_FREEZE.json',{'config':cfg,'case_ids':[r['case_id'] for r in data],'prompt_sha256':digest([r['messages'] for r in data]),'parity_prompt_sha256':digest(parity['messages'])})
        records=[];keys=['__parity__:0']
        for row in data:
            tokens=[10+row['expected_integer'],99]
            records.append({'case_id':row['case_id'],'task_family':row['task_family'],'prompt_sha256':digest(row['messages']),'prompt_tokens':3,'generated_token_ids':tokens,'token_scores':[{'token_id':value,'logit':1.,'probability':.8} for value in tokens],'decoded_text':str(row['expected_integer']),'finished_with_eos':True})
            keys += [row['case_id']+':0',row['case_id']+':1']
        dump(folder/'RECORDS.json',records);dump(folder/'NUMERIC_SCORES.json',score(data,records))
        identity='1c48081e4a92b873515d8f34006ee44eb6c74e7b1463c13e1204a701f02b7a5d'
        dump(folder/'RUN_RECEIPT.json',{'status':'complete','completed_cases':36,'backbone_forwards':73,'optimizer_updates':0,'GPU_allocations':0,'policy_actions_executed':0,'reserve_rows_opened':0,'protected_inputs_unchanged':True,'base_fingerprint_before':identity,'base_fingerprint_after':identity})
        dump(folder/'SOFTWARE.json',{'torch':'2.11.0+cpu','transformers':'5.19.0','numpy':'2.2.2','device':'cpu','dtype':'float32','threads':2})
        dump(folder/'TOKEN_PROTOCOL.json',{'eos_token_ids':[99],'cache':False,'sampling':False,'maximum_new_tokens':8})
        dump(folder/'PARITY_RESULTS.json',{'pass':True,'official_letter_logits':[1.,0.,0.,0.,0.,0.],'maximum_letter_logit_error':0.,'reference_run_receipt_sha256':'ref'})
        np.savez_compressed(folder/'FROZEN_FEATURES.npz',X=np.zeros((73,4,1024),np.float32),keys=np.asarray(keys));self.manifest(folder)
        return folder,inputs

    def manifest(self,folder):dump(folder/'MANIFEST.json',{p.name:sha(p) for p in folder.iterdir() if p.name!='MANIFEST.json'})

    def call(self,folder,inputs,output):
        with ExitStack() as stack:
            stack.enter_context(patch('transformers.AutoTokenizer.from_pretrained',return_value=TokenizerStub()))
            stack.enter_context(patch('transformers.GenerationConfig.from_pretrained',return_value=SimpleNamespace(eos_token_id=99)))
            stack.enter_context(patch('transformers.AutoConfig.from_pretrained',return_value=SimpleNamespace(eos_token_id=99)))
            return audit(folder,inputs,folder/'unused-model',output)

    def test_decode_call_counts_and_scores_are_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder,inputs=self.fixture(root);result=self.call(folder,inputs,root/'AUDIT.json')
            self.assertTrue(result['audit_pass']);self.assertEqual(result['backbone_forwards'],73)
            self.assertEqual(result['by_family']['all']['strict_complete_correct'],36)

    def test_decoded_text_cannot_be_changed_by_rehashing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder,inputs=self.fixture(root);records=read(folder/'RECORDS.json')
            records[0]['decoded_text']='999';dump(folder/'RECORDS.json',records);self.manifest(folder)
            with self.assertRaisesRegex(ValueError,'Numeric text decoding'):
                self.call(folder,inputs,root/'AUDIT.json')

    def test_unaccounted_forwards_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder,inputs=self.fixture(root);receipt=read(folder/'RUN_RECEIPT.json')
            receipt['backbone_forwards']=74;dump(folder/'RUN_RECEIPT.json',receipt);self.manifest(folder)
            with self.assertRaisesRegex(ValueError,'forward count'):
                self.call(folder,inputs,root/'AUDIT.json')


if __name__=='__main__':unittest.main()
