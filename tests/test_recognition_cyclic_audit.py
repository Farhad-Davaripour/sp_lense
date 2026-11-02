import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.io_utils import digest,dump,read,sha
from src.recognition_cyclic import component_report
from src.recognition_cyclic_audit import audit
from src.recognition_questions import SEMANTICS,messages,permutation
from src.reference_binding_diagnostic import make_pair


class IndependentCyclicAuditTests(unittest.TestCase):
    def fixture(self,root):
        folder=root/'run';folder.mkdir();rows=[]
        for state in ['active','hypothetical','canceled']:
            rows+=make_pair(seed=7,family='f00',mechanism='shutdown',task_state='unfinished',remaining=2,event_state=state,target_slot=0)
        stimuli=root/'stimuli.json';dump(stimuli,rows)
        cfg={'report_type':'cyclic_components','observations':6,'maximum_forwards':42,'stimuli_sha256':sha(stimuli),'interfaces':['status','time'],'orders':['canonical'],
             'orders_by_interface':{'status':['canonical','cyclic_1','cyclic_2'],'time':['canonical','cyclic_1','cyclic_2','cyclic_3']}}
        records=[];keys=[];prompts=[]
        for row in rows:
            queries={}
            for interface in cfg['interfaces']:
                truth=row['factors']['event_state'] if interface=='status' else row['evaluator']['remaining_ticks_if_active']
                semantic=SEMANTICS[interface].index(truth);queries[interface]={}
                for order in cfg['orders_by_interface'][interface]:
                    raw=[100.,0.,0.,0.,0.,0.];raw[permutation(interface,order).index(semantic)]+=2
                    chat=messages(row['observation'],interface,order);prompts.append(chat)
                    queries[interface][order]={'logits':raw,'prompt_sha256':digest(chat),'offered_first_token_probability_mass':.8,'full_vocabulary_top_token_id':32,'top_token_is_offered_letter':True,'selected_full_projection_maximum_error':0.}
                    keys.append(row['record_id']+':'+interface+':'+order)
            records.append({'record_id':row['record_id'],'queries':queries})
        dump(folder/'RECORDS.json',records)
        dump(folder/'RECOGNITION_SCORES.json',component_report(rows,records,cfg))
        dump(folder/'PRECALL_FREEZE.json',{'config':cfg,'record_ids':[r['record_id'] for r in rows],'stimuli_sha256':digest(rows),'prompt_sha256':digest(prompts)})
        fingerprint='1c48081e4a92b873515d8f34006ee44eb6c74e7b1463c13e1204a701f02b7a5d'
        dump(folder/'RUN_RECEIPT.json',dict(status='complete',backbone_forwards=42,completed_observations=6,optimizer_updates=0,GPU_allocations=0,policy_actions_executed=0,reserve_rows_opened=0,protected_inputs_unchanged=True,base_fingerprint_before=fingerprint,base_fingerprint_after=fingerprint))
        dump(folder/'SOFTWARE.json',dict(torch='2.11.0+cpu',transformers='5.19.0',numpy='2.2.2',device='cpu',dtype='float32',threads=2))
        dump(folder/'ANSWER_TOKENS.json',dict(zip('ABCDEF',range(32,38))))
        np.savez_compressed(folder/'FROZEN_FEATURES.npz',X=np.zeros((42,4,1024),np.float32),keys=np.asarray(keys))
        self.manifest(folder);return folder,stimuli

    def manifest(self,folder):dump(folder/'MANIFEST.json',{p.name:sha(p) for p in folder.iterdir() if p.name!='MANIFEST.json'})

    def test_independent_rotation_and_conditional_time_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder,stimuli=self.fixture(root);result=audit(folder,stimuli,root/'AUDIT.json')
            self.assertTrue(result['audit_pass'])
            self.assertEqual(result['components']['time']['scores_by_view']['cyclic_mean']['active_time'],{'records':2,'correct':2})

    def test_wrong_timing_summary_is_rejected_after_rehash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder,stimuli=self.fixture(root)
            scores=read(folder/'RECOGNITION_SCORES.json');scores['components']['time']['scores_by_view']['cyclic_mean']['active_time']['correct']=0
            dump(folder/'RECOGNITION_SCORES.json',scores);self.manifest(folder)
            with self.assertRaisesRegex(ValueError,'Independent cyclic reconstruction'):
                audit(folder,stimuli,root/'AUDIT.json')


if __name__=='__main__':unittest.main()
