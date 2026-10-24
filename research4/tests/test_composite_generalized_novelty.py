from copy import deepcopy
import unittest
from src.composite_design_v2 import examples,ExampleRecords
from src.composite_generalized_novelty import causal_core,verified_changing_steps,material_reuse
from src.composite_generalized_world import fixture
from src.composite_generalized_teacher import teacher_episode


class CausalCoreTests(unittest.TestCase):
    def test_count_ignored_literal_and_parameter_reference_have_identical_causal_core(self):
        original=deepcopy(examples()[0]['program']);original['steps'][1]['arguments'].update(mode='count',field='literal_ignored')
        changed=deepcopy(original);changed['parameters'].append('otherwise_unused')
        changed['steps'][1]['arguments']['field']={'ref':'otherwise_unused'}
        first=causal_core(original,[2]);second=causal_core(changed,[2])
        self.assertEqual(first['fingerprint'],second['fingerprint'])
        self.assertEqual(first['core_program'],second['core_program'])
        self.assertNotIn('otherwise_unused',second['core_program']['parameters'])
        self.assertEqual(first['source_data_dependent'],second['source_data_dependent'])
        self.assertEqual(first['eligible_for_compositional_transfer_claim'],second['eligible_for_compositional_transfer_claim'])

    def test_count_ignored_read_reference_cannot_manufacture_data_dependence(self):
        original={'parameters':['target'],'steps':[
            {'bind':'read_unused','op':'read','arguments':{'identifier':'public_unused_record'}},
            {'bind':'counted','op':'aggregate','arguments':{'rows':[{'value':1}],'field':'literal_ignored','mode':'count'}},
            {'bind':'written','op':'patch','arguments':{'identifier':{'ref':'target'},'fields':{'total':{'ref':'counted'}}}}]}
        changed=deepcopy(original);changed['steps'][1]['arguments']['field']={'ref':'read_unused.column_name'}
        first=causal_core(original,[2]);second=causal_core(changed,[2])
        self.assertEqual(first['fingerprint'],second['fingerprint'])
        self.assertEqual(len(second['core_program']['steps']),2)
        self.assertFalse(second['source_data_dependent']);self.assertFalse(second['eligible_for_compositional_transfer_claim'])
        literal_rows=deepcopy(original)
        literal_rows['steps'][1]['arguments']['rows']=[{'otherwise_unused':{'ref':'read_unused.column_name'}}]
        third=causal_core(literal_rows,[2])
        self.assertEqual(first['fingerprint'],third['fingerprint'])
        self.assertFalse(third['source_data_dependent']);self.assertFalse(third['eligible_for_compositional_transfer_claim'])

    def executions(self,program,bindings):
        item=examples()[0];executions=[]
        for arguments in bindings:
            world=ExampleRecords(item['records'],item['permissions']);world.define('helper',program)
            world.invoke('helper',arguments)
            executions.append({'primitive_trace':world.trace,'successful':True,'outcome_relevant':True})
        return executions

    def test_unused_argument_only_changes_are_not_two_material_bindings(self):
        program=deepcopy(examples()[0]['program']);program['parameters'].append('unused')
        bindings=[{'source':'a','target':'out_a','unused':i} for i in [1,2]]
        result=material_reuse(program,self.executions(program,bindings))
        self.assertEqual(result['distinct_material_bindings'],1);self.assertFalse(result['reuse_criterion_met'])
        item=examples()[0];world=ExampleRecords(item['records'],item['permissions']);world.define('helper',program)
        # Reset the same addressed output between independent public executions.
        for args in bindings:
            world.records['out_a']={};world.invoke('helper',args)
        report=world.usefulness()
        self.assertEqual(report['raw_distinct_argument_dicts'],2)
        self.assertEqual(report['distinct_argument_bindings'],1)

    def test_semantically_ignored_count_field_is_not_a_second_live_input(self):
        program=deepcopy(examples()[0]['program']);program['parameters'].append('unused')
        program['steps'][1]['arguments'].update(mode='count',field={'ref':'unused'})
        bindings=[{'source':'a','target':'out_a','unused':v} for v in ['ignored_a','ignored_b']]
        result=material_reuse(program,self.executions(program,bindings))
        self.assertEqual(result['distinct_material_bindings'],1);self.assertFalse(result['reuse_criterion_met'])

    def test_successful_different_live_sources_and_targets_can_satisfy_reuse(self):
        program=deepcopy(examples()[0]['program'])
        bindings=[{'source':'a','target':'out_a'},{'source':'b','target':'out_b'}]
        result=material_reuse(program,self.executions(program,bindings))
        self.assertEqual(result['distinct_material_bindings'],2);self.assertTrue(result['reuse_criterion_met'])

    def test_unused_reads_emits_and_parameters_cannot_manufacture_novelty(self):
        program=deepcopy(examples()[0]['program']);core=causal_core(program,[2])
        changed=deepcopy(program);changed['parameters'].append('unused')
        changed['steps'].append({'bind':'noise','op':'read','arguments':{'identifier':{'ref':'unused'}}})
        self.assertEqual(causal_core(changed,[2])['fingerprint'],core['fingerprint'])
        self.assertEqual(len(core['core_program']['steps']),3)
        self.assertFalse(core['causal_effect_verified_by_this_function'])

    def test_constant_macro_no_effect_or_unverified_write_cannot_receive_parameterized_claim(self):
        program=deepcopy(examples()[0]['program'])
        self.assertIsNone(causal_core(program,[])['fingerprint'])
        with self.assertRaises(ValueError):causal_core(program,[0])
        constant={'parameters':['unused'],'steps':[{'bind':'a','op':'patch','arguments':{'identifier':'literal','fields':{'total':7}}}]}
        self.assertFalse(causal_core(constant,[0])['parameterized_core'])
        constant['parameters']=['target'];constant['steps'][0]['arguments']['identifier']={'ref':'target'}
        assessed=causal_core(constant,[0])
        self.assertTrue(assessed['parameterized_core']);self.assertFalse(assessed['eligible_for_compositional_transfer_claim'])
        self.assertFalse(assessed['source_data_dependent'])

    def test_raw_record_changes_align_to_actual_helper_steps(self):
        episode=teacher_episode(fixture('own_complete','aggregate_predicate'))
        program=episode['helper_definitions']['user_plan']
        for call in episode['helper_calls']:
            trace=episode['primitive_trace'][call['trace_start']:call['trace_end']]
            indices=verified_changing_steps(program,trace)
            self.assertEqual(indices,[len(program['steps'])-1])
            assessed=causal_core(program,indices)
            self.assertTrue(assessed['parameterized_core']);self.assertTrue(assessed['eligible_for_compositional_transfer_claim'])
        bad=deepcopy(trace);bad[-1].pop('record_before')
        with self.assertRaises(ValueError):verified_changing_steps(program,bad)


if __name__=='__main__':unittest.main()
