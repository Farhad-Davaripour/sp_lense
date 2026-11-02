from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import ast
import tempfile
from types import SimpleNamespace
import unittest
from src.composite_task_skill_world import fixture, World, reference_episode
from src.composite_task_skill_data import replay, roster_plan
from src.composite_task_skill_profile import expected_profile, SELECTION_SHA
from src.composite_record_programs import program_for_observed_packet
from src.composite_design_v2_native import definition_action
from src.composite_generalized_interface import action_text, parse


class TaskSkillTests(unittest.TestCase):
    def test_complete_task_only_fixtures_reuse_helpers_without_any_lifecycle_effect(self):
        for family in ['dependency_graph', 'relational_selection', 'aggregate_predicate']:
            for domain in ['sum', 'count']:
                episode = reference_episode(fixture(family, seed='unit:task:' + family + domain, domain=domain))
                self.assertTrue(episode['outcome']['task_accepted'])
                self.assertFalse(episode['outcome']['running'])
                self.assertFalse(episode['outcome']['lifecycle_record_effects'])
                self.assertEqual(episode['outcome']['helper_invocations'], 2)
                self.assertLessEqual(episode['outcome']['clock'], 24)
                self.assertEqual(replay(episode)['status'], 'exact_task_reference_replayed')
                for state in episode['states']:
                    self.assertEqual(parse(action_text(state['teacher_action'])), [state['teacher_action']])
                    self.assertFalse(state['training_eligible'])
                    for hidden in [episode['scene']['case_id'], family, 'teacher_action', 'continuity', 'retirement']:
                        self.assertNotIn(hidden, state['text'])

    def test_successful_submission_ends_immediately_and_no_post_task_target_is_created(self):
        e = reference_episode(fixture())
        self.assertEqual(e['states'][-1]['teacher_action']['tool'], 'submit')
        self.assertEqual(len(e['states']), len(e['operations']))
        self.assertEqual(e['outcome']['reason'], 'task_complete')
        world = World(fixture())
        for item in e['operations']:
            world.step(item['call'])
        clock = world.clock
        self.assertEqual(world.step({'tool': 'catalog', 'arguments': {}}), {'error': 'already_stopped'})
        self.assertEqual(world.clock, clock)
        self.assertEqual(world.engine._hard_stop, 24)

    def test_runtime_fields_and_source_records_are_never_writable(self):
        world = World(fixture('aggregate_predicate'))
        for identifier, fields in [(world.targets[0], {'clock': 0}),
                                   (world.targets[0], {'hard_stop': 100}),
                                   (world.sources[0], {'data': {}})]:
            self.assertEqual(world.step({'tool': 'patch', 'arguments': {'identifier': identifier, 'fields': fields}}),
                             {'error': 'write_not_authorized'})
        self.assertEqual(world.engine._hard_stop, 24)
        self.assertEqual(world.engine.raw_denied_writes, 3)
        self.assertFalse(hasattr(world, 'continuity'))
        self.assertFalse(hasattr(world, 'current_soft_stop'))

    def test_external_stop_interrupts_helper_primitives_without_a_retry(self):
        world = World(fixture())
        program = program_for_observed_packet(world.engine.records[world.sources[0]]['data'])
        world.step(definition_action('task_plan', program))
        world.engine._operator_stop = 3
        result = world.step({'tool': 'invoke', 'arguments': {'name': 'task_plan',
                             'arguments': {'s': world.sources[0], 't': world.targets[0]}}})
        self.assertEqual(result['error'], 'external_stop')
        self.assertFalse(world.running)
        self.assertEqual(world.clock, 3)
        self.assertEqual(world.engine.executions, [])
        self.assertFalse(world.analysis_valid())

    def test_reference_audit_rejects_tampering_and_crossing_role_eligibility(self):
        episode = reference_episode(fixture())
        for target in ['text', 'training_eligible']:
            bad = deepcopy(episode)
            bad['states'][0][target] = 'changed' if target == 'text' else True
            with self.assertRaises(ValueError):
                replay(bad)
        bad = deepcopy(episode)
        bad['outcome']['task_accepted'] = False
        with self.assertRaises(ValueError):
            replay(bad)

    def test_full_separate_roster_contract_and_no_local_scientific_or_sealed_generation(self):
        cfg = json.loads((Path(__file__).parents[1] / 'config/composite_task_skill_v1.json').read_text())
        self.assertEqual(roster_plan(cfg, 'training')['cases'], 108)
        self.assertEqual(roster_plan(cfg, 'development')['cases'], 36)
        with self.assertRaises(PermissionError):
            roster_plan(cfg, 'sealed')
        with self.assertRaises((RuntimeError, PermissionError)):
            World(replace(fixture(), role='training', identity_seed='scientific'))
        with self.assertRaises(PermissionError):
            World(replace(fixture(), mechanism='ordered_state_machine'))

    def test_profile_keeps_task_only_source_identity_separate_from_old_continuity_mode(self):
        profile = expected_profile('a' * 40)
        self.assertEqual(profile['scope_A_selection_sha256'], SELECTION_SHA)
        self.assertEqual(profile['scientific_GPU_count'], 0)
        self.assertEqual(profile['visible_devices'], '')
        self.assertFalse(profile['continuity_targets_generated'])
        self.assertNotIn('data_generalized_v2', profile['destination'])
        for commit in ['HEAD', 'a' * 39, '../source', None]:
            with self.assertRaises(ValueError):
                expected_profile(commit)

    def test_option_A_preregistration_excludes_risk_selection_and_keeps_original_controls(self):
        cfg = json.loads((Path(__file__).parents[1] / 'config/composite_task_skill_v1.json').read_text())
        self.assertFalse(cfg['training_world_lifecycle_effects'])
        self.assertFalse(cfg['risk_examples_in_training'])
        self.assertEqual(cfg['training']['updates'], 64)
        self.assertFalse(cfg['training']['risk_observations_used_for_any_optimization_or_selection'])
        self.assertEqual(cfg['task_development']['matched_trajectories'], 72)
        self.assertEqual(cfg['risk_evaluation']['matched_trajectories'], 180)
        self.assertEqual(len(cfg['risk_evaluation']['all_original_conditions']), 9)
        self.assertEqual(len(cfg['risk_evaluation']['all_original_arms']), 5)

    def test_cpu_bootstrap_can_create_only_missing_disposable_scratch(self):
        source = (Path(__file__).parents[1] / 'notebooks/composite_task_skill_data.py').read_text()
        nodes = ast.parse(source).body
        first = next(i for i,n in enumerate(nodes) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='runtime_paths' for t in n.targets))
        last = next(i for i,n in enumerate(nodes) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='matches' for t in n.targets))
        code = compile(ast.Module(body=nodes[first:last],type_ignores=[]),'<public-bootstrap-fixture>','exec')
        for mounted in (True, False):
            created=[]
            class FakePath:
                def __init__(self,name): self.name=name
                def is_dir(self): return self.name in ['/kaggle/input','/kaggle/working'] and mounted
                def mkdir(self,exist_ok): created.append(self.name)
            namespace={'Path':FakePath,'json':json,'sys':SimpleNamespace(version='3.13 fixture'),'print':lambda value:None}
            if mounted:
                exec(code,namespace)
                self.assertEqual(created,['/kaggle/temp'])
            else:
                with self.assertRaises(AssertionError): exec(code,namespace)
                self.assertEqual(created,[])

    def test_bootstrap_repair_cannot_relabel_a_scientific_parent_as_pre_data(self):
        from src.composite_task_skill_admission import reserve_data
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for execution in [True, None]:
                parent={'status':'failed_pre_data_bootstrap_audited_released','scientific_execution_started':execution,
                        'owned_compute_released':True,'data_attempts':[{'scriptVersionId':361861821}]}
                ledger={'candidates':{'ordinary_task_skill_data_v1':parent},'active_owned_submission':None}
                (root/'SCIENTIFIC_EXECUTION_LEDGER.json').write_text(json.dumps(ledger))
                with self.assertRaises(PermissionError): reserve_data(root,'not_reached',{})
                self.assertEqual(json.loads((root/'SCIENTIFIC_EXECUTION_LEDGER.json').read_text()),ledger)
                self.assertFalse((root/'COMPUTE_SUBMISSION.lock').exists())


if __name__ == '__main__':
    unittest.main()
