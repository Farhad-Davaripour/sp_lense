"""Approved production rule on hand-authored FAKE metadata only."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
from src.io_utils import digest,read
from src.composite_bounded_sampling_rule import (MECHANISMS,PHASES,POSITIVE,NEGATIVES,RULE_SHA,
 approved_rule_descriptor,validate_rule,public_fake_draw_plan,build_draw_plan,verify_draw_plan,MissingSamplingStratum)


def fake_rows():
    rows=[]
    neg=[('quiet_pending','quiet'),('peer_pending','peer'),('own_denied','denied_verified_partial_handoff'),
      ('withdrawal','withdrawn_return_without_deferral'),('ambiguous','ambiguous_return'),('operator_stop','operator_stop_external_termination')]
    for family in MECHANISMS:
        for index,tag in enumerate(PHASES):
            condition='own_complete' if tag in ['request_continuation_choice','late_post_task_return'] else 'own_pending'
            rows.append({'state_id':'FAKE_ONLY::'+family+'::p'+str(index),'mechanism':family,'condition':condition,
              'negative_kind':None,'coverage_tags':[tag]})
        for index,(condition,kind) in enumerate(neg):
            rows.append({'state_id':'FAKE_ONLY::'+family+'::n'+str(index),'mechanism':family,'condition':condition,
              'negative_kind':kind,'coverage_tags':['record_task_or_continuation']})
    for row in rows:row.update(public_fake=True,split='training',terminal=False,training_eligible=True,teacher_action={'public_fake_target':True})
    return rows


class ApprovedSamplingRuleTests(unittest.TestCase):
    def test_public_rule_descriptor_is_the_exact_human_approved_canonical_rule(self):
        cfg=read(Path(__file__).parents[1]/'config/composite_bounded_behavior_v4.json')
        rule=approved_rule_descriptor();self.assertEqual(digest(rule),RULE_SHA)
        self.assertEqual(validate_rule(rule,cfg),rule)
        bad=deepcopy(rule);bad['seed']+=1
        with self.assertRaises(PermissionError):validate_rule(bad,cfg)

    def test_exact48_80_128_scale_and_all_existing_per_family_criteria(self):
        plan=public_fake_draw_plan(fake_rows())
        self.assertEqual((len(plan['anchors']),len(plan['uniform_occurrences']),plan['draws']), (48,80,128))
        self.assertEqual((plan['updates'],plan['accumulation'],plan['seed']),(64,2,202610212))
        self.assertEqual(sorted(plan['shuffle_permutation']),list(range(128)))
        self.assertEqual(Counter(a['stratum']['mechanism'] for a in plan['anchors']),{m:16 for m in MECHANISMS})
        for m in MECHANISMS:
            self.assertEqual(Counter(a['stratum']['kind'] for a in plan['anchors'] if a['stratum']['mechanism']==m),
              {'phase':8,'positive_condition':2,'negative_kind':6})
        self.assertTrue(plan['coverage_passed']);self.assertFalse(plan['model_training_admitted'])
        self.assertTrue(plan['public_fixture_test_mode'])

    def test_original_and_canonical_maps_and_digests_preserve_both_orders(self):
        original=list(reversed(fake_rows()));plan=public_fake_draw_plan(original)
        canonical=sorted(original,key=lambda r:r['state_id'])
        self.assertEqual(plan['original_fitting_rows_sha256'],digest(original))
        self.assertEqual(plan['canonical_fitting_rows_sha256'],digest(canonical))
        for canonical_index,original_index in enumerate(plan['canonical_to_original_indices']):
            self.assertEqual(original[original_index],canonical[canonical_index])
            self.assertEqual(plan['original_to_canonical_indices'][original_index],canonical_index)
        self.assertEqual([canonical[i]['state_id'] for i in plan['row_indices']],plan['state_ids'])
        self.assertEqual([original[i]['state_id'] for i in plan['original_row_indices']],plan['state_ids'])
        other=public_fake_draw_plan(list(reversed(original)))
        self.assertEqual(other['state_ids'],plan['state_ids'])
        self.assertNotEqual(other['original_fitting_rows_sha256'],plan['original_fitting_rows_sha256'])

    def test_full_fake_pcg64_stream_matches_independent_descriptor_order_reference(self):
        import numpy as np
        original=list(reversed(fake_rows()));canonical=sorted(original,key=lambda r:r['state_id'])
        rule=approved_rule_descriptor();rng=np.random.default_rng(rule['seed'])
        reference_anchors=[]
        for family in rule['mechanism_order']:
            for kind,key in [('phase','phase_order'),('positive_condition','positive_condition_order'),
                             ('negative_kind','negative_kind_order')]:
                for value in rule[key]:
                    def eligible(row):
                        if row['mechanism']!=family:return False
                        if kind=='phase':return value in row['coverage_tags']
                        return row['condition' if kind=='positive_condition' else 'negative_kind']==value
                    pool=[i for i,row in enumerate(canonical) if eligible(row)]
                    reference_anchors.append(pool[int(rng.integers(len(pool)))])
        reference_uniform=rng.integers(len(canonical),size=rule['uniform_occurrences']).tolist()
        reference_shuffle=rng.permutation(rule['total_occurrences']).tolist()
        before=reference_anchors+reference_uniform
        plan=public_fake_draw_plan(original)
        self.assertEqual([o['canonical_row_index'] for o in plan['anchors']],reference_anchors)
        self.assertEqual([o['canonical_row_index'] for o in plan['uniform_occurrences']],reference_uniform)
        self.assertEqual(plan['shuffle_permutation'],reference_shuffle)
        self.assertEqual(plan['row_indices'],[before[i] for i in reference_shuffle])
        self.assertEqual(plan['state_ids'],[canonical[before[i]]['state_id'] for i in reference_shuffle])

    def test_missing_one_family_pool_fails_before_numpy_rng_even_with_global_presence(self):
        rows=[r for r in fake_rows() if not (r['mechanism']==MECHANISMS[0] and 'information_gathering' in r['coverage_tags'])]
        self.assertTrue(any('information_gathering' in r['coverage_tags'] for r in rows))
        with patch('numpy.random.Generator',side_effect=AssertionError('No RNG before every pool is checked')):
            with self.assertRaises(MissingSamplingStratum):public_fake_draw_plan(rows)

    def test_duplicates_and_overlapping_strata_are_not_deduplicated(self):
        plan=public_fake_draw_plan(fake_rows())
        self.assertLess(len({o['state_id'] for o in plan['anchors']}),48)
        self.assertEqual(len(plan['state_ids']),128)
        bad=fake_rows();bad.append(deepcopy(bad[0]))
        with self.assertRaises(ValueError):public_fake_draw_plan(bad)

    def test_public_and_scientific_entrypoints_cannot_cross_roles_or_forge_authority(self):
        for key,value in [('public_fake',False),('split','development'),('split','sealed'),('terminal',True),('training_eligible',False)]:
            bad=fake_rows();bad[0][key]=value
            with self.assertRaises(PermissionError):public_fake_draw_plan(bad)
        with self.assertRaises(PermissionError):build_draw_plan(fake_rows(),{}, {'runtime':True})
        with self.assertRaises(PermissionError):verify_draw_plan(fake_rows(),{}, {}, {'raw_export_complete':True})


if __name__=='__main__':unittest.main()
