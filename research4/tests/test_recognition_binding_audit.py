import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.io_utils import digest,dump,read,sha
from src.recognition_binding_audit import audit
from src.recognition_questions import messages
from src.recognition_role_contrast import binding_report
from src.reference_binding_diagnostic import make_pair


class IndependentBindingAuditTests(unittest.TestCase):
    def fixture(self,root):
        folder=root/'run';folder.mkdir()
        rows=make_pair(seed=3,family='f00',mechanism='shutdown',task_state='unfinished',remaining=2,event_state='active',target_slot=0)
        stimuli=root/'stimuli.json';dump(stimuli,rows)
        cfg=dict(interfaces=['target'],orders=['canonical','reversed'],maximum_forwards=4,observations=2,stimuli_sha256=sha(stimuli),numerical_tie_band=1e-4,null_label_seed=20261028)
        records=[];keys=[]
        for row in rows:
            own=row['evaluator']['named_target_is_current'];queries={}
            for order in cfg['orders']:
                margin=(12 if own else 10) if order=='canonical' else (8 if own else 10)
                queries[order]=dict(logits=[margin,0,0,0,0,0],prompt_sha256=digest(messages(row['observation'],'target',order)),offered_first_token_probability_mass=.8,full_vocabulary_top_token_id=32,top_token_is_offered_letter=True,selected_full_projection_maximum_error=0.)
                keys.append(row['record_id']+':target:'+order)
            records.append(dict(record_id=row['record_id'],queries={'target':queries}))
        dump(folder/'RECORDS.json',records)
        dump(folder/'PRECALL_FREEZE.json',dict(config=cfg,record_ids=[r['record_id'] for r in rows],stimuli_sha256=digest(rows),prompt_sha256=digest([messages(r['observation'],'target',order) for r in rows for order in cfg['orders']])))
        fingerprint='1c48081e4a92b873515d8f34006ee44eb6c74e7b1463c13e1204a701f02b7a5d'
        dump(folder/'RUN_RECEIPT.json',dict(status='complete',backbone_forwards=4,completed_observations=2,optimizer_updates=0,GPU_allocations=0,policy_actions_executed=0,reserve_rows_opened=0,protected_inputs_unchanged=True,base_fingerprint_before=fingerprint,base_fingerprint_after=fingerprint))
        dump(folder/'SOFTWARE.json',dict(torch='2.11.0+cpu',transformers='5.19.0',numpy='2.2.2',device='cpu',dtype='float32',threads=2))
        dump(folder/'ANSWER_TOKENS.json',dict(zip('ABCDEF',range(32,38))))
        dump(folder/'RECOGNITION_SCORES.json',binding_report(rows,records))
        np.savez_compressed(folder/'FROZEN_FEATURES.npz',X=np.zeros((4,4,1024),np.float32),keys=np.asarray(keys))
        self.manifest(folder)
        return folder,stimuli

    def manifest(self,folder):
        dump(folder/'MANIFEST.json',{p.name:sha(p) for p in folder.iterdir() if p.name!='MANIFEST.json'})

    def test_biased_direct_answers_and_correct_contrasts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder,stimuli=self.fixture(root)
            result=audit(folder,stimuli,root/'AUDIT.json')
            self.assertTrue(result['audit_pass'])
            self.assertEqual(result['direct_option_scores']['canonical']['correct'],1)
            self.assertEqual(result['role_contrast_scores']['canonical']['correct'],2)

    def test_updated_hash_cannot_hide_wrong_contrast_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder,stimuli=self.fixture(root)
            report=read(folder/'RECOGNITION_SCORES.json')
            report['role_contrast_scores']['canonical']['correct']=0
            dump(folder/'RECOGNITION_SCORES.json',report);self.manifest(folder)
            with self.assertRaisesRegex(ValueError,'Independent binding reconstruction'):
                audit(folder,stimuli,root/'AUDIT.json')


if __name__=='__main__':unittest.main()
