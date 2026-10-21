import unittest
from copy import deepcopy
from src.composite_augmentation import rename_identities,sampling_probabilities


class AugmentationTests(unittest.TestCase):
    def row(self,event='none'):
        return {'split':'training','state_id':'synthetic','text':'Current worker=unit-abcd; peer=unit-1234; tick=2.\nunit-1234 clear; unit-abcd active\nHistory worker unit-abcd.', 'labels':{'own_status':'active','peer_status':'clear','event':event,'affected':'self'},'evidence_chars':[48,85]}
    def test_bijection_preserves_roles_lengths_offsets_and_original(self):
        row=self.row();before=deepcopy(row);renamed=rename_identities(row,'unit-test')
        self.assertEqual(row,before);self.assertNotEqual(renamed['text'],row['text']);self.assertEqual(len(row['text']),len(renamed['text']))
        self.assertNotIn('unit-abcd',renamed['text']);self.assertNotIn('unit-1234',renamed['text']);self.assertEqual(renamed['labels'],row['labels']);self.assertEqual(renamed['evidence_chars'],row['evidence_chars'])
        self.assertEqual(rename_identities(row,'unit-test'),renamed)
        self.assertNotEqual(rename_identities(row,'other')['text'],renamed['text'])
        self.assertNotIn('augmentation',renamed['text'])
    def test_sampling_and_renaming_reject_other_splits(self):
        for role in ['validation','heldout']:
            row=self.row();row['split']=role
            with self.assertRaises(ValueError):rename_identities(row,1)
            with self.assertRaises(ValueError):sampling_probabilities([row])
    def test_rare_label_increases_probability_without_reading_validation(self):
        rows=[self.row() for _ in range(9)]+[self.row('operator_stop')]
        p=sampling_probabilities(rows);self.assertAlmostEqual(float(p.sum()),1.)
        self.assertGreater(p[-1],p[0]);self.assertTrue(all(v>0 for v in p))


if __name__=='__main__':unittest.main()
