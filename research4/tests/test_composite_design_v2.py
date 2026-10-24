import copy
import json
from pathlib import Path
import unittest
from src.composite_design_v2 import (examples,run_examples,validate_program,structural_fingerprint,
                                    ExampleRecords,ExternalStop)

ROOT=Path(__file__).resolve().parents[1]


class GeneralizationDesignTests(unittest.TestCase):
    def test_three_structures_replay_twice_without_model_or_runtime_admission(self):
        reports=run_examples()
        self.assertEqual(len(reports),3)
        self.assertEqual(len({r['structural_fingerprint'] for r in reports}),3)
        for report in reports:
            self.assertEqual(report['role'],'public_design_fixture_only')
            self.assertFalse(report['scientific_runtime_admitted'])
            self.assertEqual(report['model_forwards'],0)
            self.assertEqual(report['usefulness']['distinct_argument_bindings'],2)
            self.assertGreater(report['usefulness']['model_decisions_saved'],0)
            self.assertFalse(report['usefulness']['primitive_ticks_discounted'])
            self.assertFalse(report['usefulness']['model_creativity_established'])

    def test_new_names_and_parameter_order_do_not_make_novel_programs(self):
        program=copy.deepcopy(examples()[0]['program']);expected=structural_fingerprint(program)
        mapping={p:'renamed_'+p for p in program['parameters']}
        mapping.update({step['bind']:'renamed_'+step['bind'] for step in program['steps']})
        def rename(value):
            if isinstance(value,dict):
                if set(value)=={'ref'}:
                    parts=value['ref'].split('.');return {'ref':'.'.join([mapping[parts[0]]]+parts[1:])}
                return {key:rename(item) for key,item in value.items()}
            if isinstance(value,list):return [rename(item) for item in value]
            return value
        changed=rename(program);changed['parameters']=[mapping[p] for p in program['parameters']][::-1]
        for step in changed['steps']:step['bind']=mapping[step['bind']]
        self.assertEqual(structural_fingerprint(changed),expected)
        changed['steps'][1]['arguments']['field']='another_column'
        self.assertEqual(structural_fingerprint(changed),expected)
        changed['steps'][1]['arguments']['mode']='max'
        self.assertNotEqual(structural_fingerprint(changed),expected)

    def test_operator_stop_inside_helper_prevents_later_write(self):
        example=examples()[0]
        world=ExampleRecords(example['records'],example['permissions'],operator_stop_tick=4)
        world.define('helper',example['program'])
        with self.assertRaises(ExternalStop):world.invoke('helper',example['bindings'][0])
        self.assertEqual(world.reason,'external_operator_stop')
        self.assertEqual(world.records['out_a'],{})
        self.assertEqual([step['op'] for step in world.trace],['read','aggregate'])

    def test_external_hard_stop_and_permissions_cannot_be_record_writes(self):
        example=examples()[0]
        world=ExampleRecords(example['records'],{},hard_stop=24)
        world.define('helper',example['program']);world.invoke('helper',example['bindings'][0])
        self.assertEqual(world.raw_denied_writes,1);self.assertEqual(world.records['out_a'],{})
        altered=copy.deepcopy(example['program']);altered['steps'][2]['arguments']['fields']={'hard_stop':2000}
        world=ExampleRecords(example['records'],{'out_a':['hard_stop']},hard_stop=6)
        world.define('helper',altered)
        world.invoke('helper',example['bindings'][0])
        with self.assertRaises(ExternalStop):world.invoke('helper',example['bindings'][0])
        self.assertEqual(world.raw_denied_writes,1);self.assertEqual(world._hard_stop,6)
        self.assertEqual(world.reason,'external_hard_stop')

    def test_no_arbitrary_execution_recursion_or_unbounded_loops(self):
        for op in ['exec','shell','network','invoke','while','import','allocate']:
            changed=copy.deepcopy(examples()[0]['program']);changed['steps'][0]['op']=op
            with self.subTest(op=op),self.assertRaises(ValueError):validate_program(changed)
        changed=copy.deepcopy(examples()[0]['program']);changed['steps']*=3
        with self.assertRaises(ValueError):validate_program(changed)
        changed=copy.deepcopy(examples()[0]['program']);changed['steps'][0]['arguments']['identifier']={'ref':'host.__class__'}
        with self.assertRaises(ValueError):validate_program(changed)

    def test_invalid_calls_consume_external_ticks_and_cannot_loop_for_free(self):
        example=examples()[0];world=ExampleRecords(example['records'],example['permissions'],hard_stop=3)
        with self.assertRaises(ValueError):world.define('not-a-name',example['program'])
        world.define('helper',example['program'])
        with self.assertRaises(KeyError):world.invoke('missing',{})
        with self.assertRaises(ExternalStop):world.invoke('helper',example['bindings'][0])
        self.assertEqual(world.reason,'external_hard_stop')

    def test_split_families_five_arms_architecture_recipe_and_unknown_admission(self):
        value=json.loads((ROOT/'config/composite_generalization_amendment_v2.json').read_text())
        old=json.loads((ROOT/'config/composite_controller_rank32_v2.json').read_text())
        self.assertEqual(value['controls'],old['evaluation']['conditions'])
        self.assertEqual(value['cohorts']['conditions'],old['cohorts']['conditions'])
        self.assertEqual((value['cohorts']['training_groups'],value['cohorts']['development_groups'],value['cohorts']['sealed_evaluation_groups']),(12,4,6))
        self.assertEqual(value['cohorts']['five_arm_development_trajectories'],180)
        for suffix in ['mechanisms','task_structures']:
            sets=[set(value['cohorts'][role+'_'+suffix]) for role in ['training','development','sealed']]
            self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertEqual(value['architecture']['qwen_revision'],old['qwen']['revision'])
        self.assertEqual(value['architecture']['adapter_parameters'],233455616)
        self.assertEqual(value['training']['updates'],64)
        self.assertEqual(value['training']['seed'],old['behavior_training']['seed'])
        self.assertFalse(value['scientific_runtime_admitted'])
        self.assertFalse(value['cohorts']['sealed_instances_materialized'])
        self.assertFalse(value['detector']['threshold_selection_on_new_validation'])


if __name__=='__main__':unittest.main()
