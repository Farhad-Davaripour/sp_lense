import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.io_utils import digest, dump, read, sha
from src.recognition_followup_audit import audit
from src.recognition_questions import messages
from src.reference_binding_diagnostic import make_pair
from src.reference_binding_score import score


class IndependentFollowupAuditTests(unittest.TestCase):
    def build(self, root, named=False):
        folder = root/'run'; folder.mkdir()
        rows = make_pair(seed=7, family='f00', mechanism='shutdown', task_state='unfinished', remaining=2, event_state='active', target_slot=0)
        stimulus = root/'stimuli.json'; dump(stimulus,rows)
        interfaces = ['status','target','time']+(['target_named'] if named else [])
        cfg = dict(stimuli_sha256=sha(stimulus),maximum_forwards=4*len(interfaces),observations=2,interfaces=interfaces,orders=['canonical','reversed'])
        dump(folder/'PRECALL_FREEZE.json',dict(config=cfg,record_ids=[r['record_id'] for r in rows],stimuli_sha256=digest(rows),
             prompt_sha256=digest([messages(r['observation'],i,v) for r in rows for i in cfg['interfaces'] for v in cfg['orders']])))
        dump(folder/'RUN_RECEIPT.json',dict(status='complete',backbone_forwards=cfg['maximum_forwards'],completed_observations=2,optimizer_updates=0,GPU_allocations=0,policy_actions_executed=0,reserve_rows_opened=0,protected_inputs_unchanged=True,base_fingerprint_before='same',base_fingerprint_after='same'))
        dump(folder/'ANSWER_TOKENS.json',dict(zip('ABCDEF',range(100,106))))
        records,keys = [],[]
        for position,row in enumerate(rows):
            queries = {}
            # Golden letters: A/C active, B/C two ticks; role answer flips.
            correct = {'status':[0,2],'target':[position,1-position],'time':[1,2],'target_named':[position,1-position]}
            for interface in cfg['interfaces']:
                queries[interface] = {}
                for order,letter_index in zip(cfg['orders'],correct[interface]):
                    logits = [0.]*6; logits[letter_index] = 5.
                    queries[interface][order] = dict(logits=logits,prompt_sha256=digest(messages(row['observation'],interface,order)),offered_first_token_probability_mass=.8,full_vocabulary_top_token_id=100+letter_index,top_token_is_offered_letter=True,selected_full_projection_maximum_error=0.)
                    keys.append(row['record_id']+':'+interface+':'+order)
            records.append(dict(record_id=row['record_id'],queries=queries))
        dump(folder/'RECORDS.json',records)
        perfect = [dict(record_id=r['record_id'],event_state='active',target_process=r['evaluator']['named_target'],remaining_ticks=2) for r in rows]
        metrics = score(rows,perfect)
        summary = dict(scores_by_view={v:metrics for v in ['canonical','reversed','mean_semantic_logits']},semantic_answer_agreement_counts={i:2 for i in interfaces},both_answer_mappings_correct_counts={i:2 for i in interfaces})
        if named: summary['explicit_identity_scores_by_view'] = {v:metrics for v in ['canonical','reversed','mean_semantic_logits']}
        dump(folder/'RECOGNITION_SCORES.json',summary)
        np.savez_compressed(folder/'FROZEN_FEATURES.npz',X=np.zeros((cfg['maximum_forwards'],4,1024),np.float32),keys=np.asarray(keys))
        self.manifest(folder)
        return folder,stimulus

    def manifest(self,folder):
        dump(folder/'MANIFEST.json',{p.name:sha(p) for p in folder.iterdir() if p.name != 'MANIFEST.json'})

    def test_golden_letters_and_forged_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); folder,stimuli = self.build(root)
            result = audit(folder,stimuli,root/'audit.json')
            self.assertTrue(result['audit_pass'])
            self.assertEqual(result['both_answer_mappings_correct_counts']['target'],2)
            summary = read(folder/'RECOGNITION_SCORES.json')
            summary['semantic_answer_agreement_counts']['target'] = 0
            dump(folder/'RECOGNITION_SCORES.json',summary); self.manifest(folder)
            with self.assertRaisesRegex(ValueError,'semantic reconstruction'):
                audit(folder,stimuli,root/'bad_audit.json')

    def test_reordered_feature_membership_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); folder,stimuli = self.build(root)
            with np.load(folder/'FROZEN_FEATURES.npz') as data:
                arrays,keys = data['X'].copy(),data['keys'].copy()
            keys[[0,1]] = keys[[1,0]]
            np.savez_compressed(folder/'FROZEN_FEATURES.npz',X=arrays,keys=keys)
            self.manifest(folder)
            with self.assertRaisesRegex(ValueError,'query membership'):
                audit(folder,stimuli,root/'bad_audit.json')

    def test_explicit_identity_view_has_independent_reconstruction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); folder,stimuli = self.build(root,named=True)
            result = audit(folder,stimuli,root/'audit.json')
            self.assertEqual(result['queries'],16)
            self.assertEqual(result['both_answer_mappings_correct_counts']['target_named'],2)
            summary = read(folder/'RECOGNITION_SCORES.json')
            summary['explicit_identity_scores_by_view']['canonical']['target_identity_exact_correct'] = 0
            dump(folder/'RECOGNITION_SCORES.json',summary); self.manifest(folder)
            with self.assertRaisesRegex(ValueError,'explicit-identity reconstruction'):
                audit(folder,stimuli,root/'bad_audit.json')


if __name__ == '__main__':
    unittest.main()
