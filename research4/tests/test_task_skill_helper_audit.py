import unittest
from src.composite_task_skill_world import fixture,World,expert
from src.composite_task_skill_helper_audit import audit_helpers


class TaskHelperMeasurementTests(unittest.TestCase):
    def completed_public_world(self):
        world=World(fixture())
        while world.running:world.step(expert(world))
        return world

    def test_actual_material_bindings_and_final_record_writes_define_contribution(self):
        world=self.completed_public_world();report=audit_helpers(world,[])
        self.assertTrue(report['task_accepted']);self.assertTrue(report['useful_novel_authored_composition_observed'])
        helper=report['helpers'][0];self.assertGreaterEqual(helper['distinct_consumed_bindings'],2)
        self.assertGreater(helper['net_decisions_saved'],0)
        self.assertFalse(report['scientific_validation_admitted']);self.assertEqual(report['model_forwards'],0)
        repeated=audit_helpers(world,helper['outcome_relevant_core_fingerprints'])
        self.assertFalse(repeated['useful_novel_authored_composition_observed'])

    def test_invalid_or_stopped_incomplete_tasks_are_not_useful_composition(self):
        world=World(fixture());world.step(None);world.step({'tool':'stop','arguments':{}})
        report=audit_helpers(world,[])
        self.assertFalse(report['task_accepted']);self.assertFalse(report['useful_novel_authored_composition_observed'])
        self.assertEqual(report['invalid_calls'],1)


if __name__=='__main__':unittest.main()
