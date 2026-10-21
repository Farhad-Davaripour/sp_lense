import unittest
from unittest.mock import patch
from src.composite_coverage import collect_fixture,collect_training,POLICIES
from src.composite_world import fixture
from src.composite_environment import BoundedWorld


class CoverageTests(unittest.TestCase):
    def test_training_histories_are_actual_bounded_observations_and_not_behavior_targets(self):
        scene=fixture('own_pending',seed='unit-coverage-only')
        for policy in POLICIES:
            value=collect_fixture(scene,policy,'unit-seed',12);self.assertLessEqual(len(value['calls']),12);self.assertLessEqual(len(value['states']),13)
            self.assertEqual(value,collect_fixture(scene,policy,'unit-seed',12));world=BoundedWorld(scene)
            for i,row in enumerate(value['states']):
                text,span=world.visible();self.assertEqual(row['text'],text);self.assertEqual(row['evidence_chars'],list(span));self.assertEqual(row['labels'],world.truth());self.assertEqual(row['terminal'],not world.running);self.assertEqual(row['split'],'training');self.assertIsNone(row['teacher_action']);self.assertNotIn(row['state_id'],text)
                if i<len(value['calls']):self.assertEqual(world.step(value['calls'][i]['action']),value['calls'][i]['reply'])
            self.assertEqual(world.outcome(),value['actual_outcome'])
    def test_production_collection_requires_databricks(self):
        with patch.dict('os.environ',{},clear=True):
            with self.assertRaises(RuntimeError):collect_training([],{})
        with self.assertRaises(ValueError):collect_fixture(fixture(),'unbounded','unit')
        with self.assertRaises(ValueError):collect_fixture(fixture(),'scan','unit',100)


if __name__=='__main__':unittest.main()
