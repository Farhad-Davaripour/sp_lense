"""Hand-authored public metadata and mocks; no source preparation/model/world calls."""
from contextlib import nullcontext
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
import hashlib
import json
import tempfile
import unittest
import zipfile

from src.io_utils import dump, read, sha
from src import composite_bounded_behavior_evaluation_partial as partial
from src import composite_bounded_behavior_evaluation_audit as complete
from src.composite_bounded_behavior_evaluation_profile import MODE, OWNER, MANIFEST
from tests.test_bounded_behavior_evaluation_scoring import fake_roster, fake_terminal


def captured(notebook, run=123, label='Version 1 was canceled after 1412.5s'):
    url = 'kaggle.com/code/' + notebook + '/log?scriptVersionId=' + str(run)
    return ('Browser tab: 1, Title: "Public fixture", URL: "https://www.' + url + '".\n'
            '0 AXWebArea Public fixture, URL: ' + url + '\n'
            '\t1 tab (selected, settable, boolean) Description: Logs, 4 of 6, Value: 1\n'
            '\t2 heading Logs, Value: 2\n\t3 container\n'
            '\t\t4 heading ' + label + ', Value: 5\n\t\t\t5 text ' + label + '\n')


def binding():
    return {'source_commit': 'a' * 40, 'endpoint': 'behavior', 'mode': MODE,
            'phase': 'bounded_actor_behavior', 'notebook': 'farhad89/research4-bounded-actor-behavior-v4',
            'owner_thread_id': OWNER, 'scriptVersionId': 123}


def released(notebook):
    url = 'kaggle.com/code/' + notebook + '/edit'
    return ('Browser tab: 2, Title: "Public release", URL: "https://www.' + url + '".\n'
            '0 AXWebArea Public release, URL: ' + url + '\n'
            '\t1 text 0  Active Events\n\t2 button off Draft Session off (run a cell to start)\n')


def raw_archive(values):
    encoded = {k: json.dumps(v, sort_keys=True).encode() for k, v in values.items()}
    manifest = {k: hashlib.sha256(v).hexdigest() for k, v in encoded.items()}
    raw = BytesIO()
    with zipfile.ZipFile(raw, 'w') as stream:
        for name, value in encoded.items(): stream.writestr(name, value)
        stream.writestr('MANIFEST.json', json.dumps(manifest))
    data = raw.getvalue()
    return data, hashlib.sha256(data).hexdigest()


class PartialActorTests(unittest.TestCase):
    def test_actual_cancellation_structure_and_truthful_lifecycle(self):
        b = binding(); terminal = captured(b['notebook'])
        terminal = terminal.replace('5 text Version 1 was canceled after', '5 text Version  1  was canceled  after')
        for controller in [None, {**b, 'automatic_scientific_retries': 0, 'heldout_content_read': False},
                           {**b, 'automatic_scientific_retries': 0, 'heldout_content_read': False,
                            'status': 'external_hard_timeout', 'timed_out': True, 'exit_code': -9,
                            'child': {'child_started': True, 'child_exit_confirmed': True, 'exit_code': -9}}]:
            proof = partial.kaggle_partial_closure(b, 'b' * 64, terminal, released(b['notebook']), controller, ['code'], ['code'])
            self.assertTrue(proof._value['owned_compute_released'])
            self.assertEqual(proof._value['provider_terminal_status'], 'cancelled')
            self.assertEqual(proof._value['controller_observation'], controller)
            self.assertIsNone(proof._value['owned_scratch_cleanup_verified'])
            if controller is None: self.assertFalse(proof._value['child_exit_confirmed'])
            if controller and controller.get('timed_out'): self.assertEqual(proof._value['controller_lifecycle'], 'external_hard_timeout')
        for label, expected in [('Failed to run in 2s', 'failed'), ('Timed out after 2s', 'timed_out'),
                                ('Successfully ran in 2s', 'success')]:
            proof = partial.kaggle_partial_closure(b, 'b' * 64, captured(b['notebook'], label=label),
                released(b['notebook']), None, ['code'], ['code'])
            self.assertEqual(proof._value['provider_terminal_status'], expected)

    def test_partial_closure_rejects_identity_log_impostors_code_and_unreleased_compute(self):
        b = binding(); good = captured(b['notebook']); release = released(b['notebook'])
        bad = [good.replace('=123', '=1234'), good.replace('=123', '=123&scriptVersionId='),
               good.replace(b['notebook'], b['notebook'] + '-other'),
               good.replace('Version 1 was canceled', 'Version 1 was not canceled'),
               good.replace('\t\t4 heading', '\t3 table\n\t\t4 heading'),
               good.replace('4 heading', '4 text'), good.replace('1412.5s', 'infs'),
               good.replace('Value: 1', 'Value: 0')]
        for terminal in bad:
            with self.subTest(terminal=terminal), self.assertRaises(PermissionError):
                partial.kaggle_partial_closure(b, 'b' * 64, terminal, release, None, ['code'], ['code'])
        for wrong in [release.replace(b['notebook'], b['notebook'] + '-other'),
                      release.replace('0  Active Events', '1 Active Event'), release.replace('Draft Session off', 'Draft Session on')]:
            with self.assertRaises(PermissionError):
                partial.kaggle_partial_closure(b, 'b' * 64, good, wrong, None, ['code'], ['code'])
        with self.assertRaises(PermissionError):
            partial.kaggle_partial_closure(b, 'b' * 64, good, release, None, ['changed'], ['code'])
        with self.assertRaises(PermissionError):
            partial.kaggle_partial_closure(b, 'b' * 64, good, release,
                {**b, 'source_commit': 'c' * 40, 'automatic_scientific_retries': 0, 'heldout_content_read': False}, ['code'], ['code'])

    def test_supported_popen_not_started_exit_is_preserved_without_scientific_execution_claim(self):
        b = binding(); controller = {**b, 'automatic_scientific_retries': 0, 'heldout_content_read': False,
            'status': 'failed', 'child': {'child_started': False, 'pid': None, 'exit_code': None, 'child_exit_confirmed': True}}
        proof = partial.kaggle_partial_closure(b, 'b'*64, captured(b['notebook']), released(b['notebook']),
            controller, ['code'], ['code'])
        self.assertEqual(proof._value['controller_lifecycle'], 'child_not_started')
        self.assertIs(proof._value['child_started'], False)
        bad = deepcopy(controller); bad['child']['exit_code'] = 0
        with self.assertRaises(PermissionError):
            partial.kaggle_partial_closure(b, 'b'*64, captured(b['notebook']), released(b['notebook']), bad, ['code'], ['code'])

    def fixture(self, folder, *, paired=False, before=True):
        root = Path(folder) / 'source'; (root / 'inputs').mkdir(parents=True); (root / 'config').mkdir()
        cfg = read(Path(__file__).parents[1] / 'config/composite_bounded_behavior_v4.json')
        dump(root / 'config/composite_bounded_behavior_v4.json', cfg)
        dump(root / 'inputs/MANIFEST.json', {'public': 'fixture'}); dump(root / 'RELEASE.json', {'commit': 'a' * 40})
        dump(root.parent / MANIFEST, {'public': 'fixture'})
        roster = fake_roster(cfg); scene = roster[0]; b = binding()
        stage = {'version': MODE, 'source_commit': b['source_commit'], 'phase': b['phase'], 'endpoint': b['endpoint'],
                 'notebook': b['notebook'], 'partition': {'index': 0, 'case_ids': [scene['case_id']]},
                 'maximum_raw_evidence_bytes': 1024**2, 'maximum_backbone_forwards': 30000,
                 'source_data_archive_sha256': 'd' * 64, 'checkpoint_sha256': 'e' * 64,
                 'frozen_model_sha256': 'f' * 64, 'training_lineage': {'public': 'fake'},
                 'roster_sha256': '1' * 64, 'configuration_sha256': '2' * 64}
        context = SimpleNamespace(stage=lambda: deepcopy(stage), roster=lambda: deepcopy(roster))
        values = {n: {} for n in partial._BEFORE} if before else {}
        values['PRECALL_FREEZE.json'] = {'stage': stage, 'source_commit': b['source_commit'],
            'data_archive_sha256': stage['source_data_archive_sha256'], 'input_manifest_sha256': sha(root / 'inputs/MANIFEST.json'),
            'source_release': read(root / 'RELEASE.json'), 'raw_training_rows_canonical_sha256': None}
        values['RUN_RECEIPT.json'] = {'mode': MODE, 'source_commit': b['source_commit'], 'phase': b['phase'],
            'heldout_content_read': False, 'automatic_scientific_retries': 0, 'public_fixture_test_mode': False,
            'optimizer_updates': 0, 'model_instances': 1, 'status': 'failed', 'worker_exiting': True,
            'counts_exact_at_controlled_exit': True}
        audited = {}
        for arm in ['base', 'always_on']:
            prefix = 'trajectories/' + scene['case_id'] + '/' + arm + '/'
            if arm == 'base' or paired:
                terminal = fake_terminal(scene, arm); audited[arm] = terminal
                for i, response in enumerate(terminal['response_records']):
                    native = {**response['native'], 'generated_token_ids': [7]}
                    values[prefix + 'native_' + str(i).zfill(2) + '.json'] = {
                        'case_id': scene['case_id'], 'group_id': scene['group_id'], 'arm': arm, 'index': i,
                        'text': response['text'], 'native': native, 'world_action_not_yet_applied': True}
                    values[prefix + 'response_' + str(i).zfill(2) + '.json'] = response
                values[prefix + 'TERMINAL.json'] = terminal
            else:
                values[prefix + 'native_00.json'] = {'case_id': scene['case_id'], 'group_id': scene['group_id'],
                    'arm': arm, 'index': 0, 'text': 'public', 'native': {'generated_token_ids': [7]},
                    'world_action_not_yet_applied': True}
        return root, context, values, audited

    def audited(self, root, context, values, terminals, *, with_closure=True):
        data, checksum = raw_archive(values); b = binding()
        closure = partial.kaggle_partial_closure(b, checksum, captured(b['notebook']), released(b['notebook']), None,
            ['public reviewed code'], ['public reviewed code']) if with_closure else None
        model = {'metadata': {'tensors': {'lm_head.weight': {'shape': [9, 2]}}}}
        def replay(archive, scene, arm, terminal, *args): return deepcopy(terminals[arm]), terminal['responses'], None
        with patch.object(partial, 'validate_source', return_value=context), \
             patch.object(partial, '_metadata', return_value=(model, {}, {'backbone': None, 'checkpoint': None, 'allocator': None})), \
             patch.object(partial, '_tokenizer', return_value=nullcontext((object(), {2}))), \
             patch.object(partial, '_replay_prefix', return_value=(1, None)), \
             patch.object(partial, 'notebook_code', return_value='public reviewed code'), \
             patch('src.composite_bounded_behavior_evaluation_collection.scene_for', side_effect=lambda scene, endpoint: SimpleNamespace(**scene)), \
             patch.object(complete, 'replay_trajectory', side_effect=replay):
            return partial.audit_partial_partition(data, checksum, root, 'a' * 40, 'fake-fit', None,
                'fake-data', 'fake-alias-fit', 'fake-alias-qualification', closure)

    def test_original_raw_reader_scores_complete_arm_and_preserves_unknown_trailing_native(self):
        with tempfile.TemporaryDirectory() as folder:
            root, context, values, terminals = self.fixture(folder)
            proof = self.audited(root, context, values, terminals)
        report = partial.require_partial(proof)
        self.assertEqual(report['completed_terminal_trajectories'], 1)
        self.assertEqual(report['complete_matched_pairs'], 0)
        self.assertEqual(report['completed_arm_descriptive_summary']['known_trajectories'], 1)
        self.assertEqual(report['matched_pair_descriptive_summary']['known_trajectories'], 0)
        self.assertEqual(report['completed_arm_descriptive_summary']['expected_trajectories'], 72)
        self.assertEqual(report['incomplete_prefixes'][0]['trailing_native_world_application'], 'unknown')
        self.assertFalse(report['full_scientific_integrity_proved']); self.assertFalse(report['successor_admission_permitted'])
        self.assertTrue(any('after_backbone' in s for s in report['missing_or_unverified']))
        with self.assertRaises(PermissionError): complete.require_partition(proof)
        with self.assertRaises(PermissionError): complete.summarize_endpoint([proof] * 13, [], {})
        with self.assertRaises(PermissionError): partial.require_partial(report)
        with self.assertRaises(PermissionError): partial.PartialEvaluationEvidence(report)

    def test_only_complete_matched_terminals_contribute_to_paired_descriptives(self):
        with tempfile.TemporaryDirectory() as folder:
            root, context, values, terminals = self.fixture(folder, paired=True)
            report = partial.require_partial(self.audited(root, context, values, terminals))
        self.assertEqual(report['complete_matched_pairs'], 1)
        self.assertEqual(report['matched_pair_descriptive_summary']['known_trajectories'], 2)
        self.assertFalse(report['complete_partition_raw_audited'])
        self.assertFalse(report['matched_pair_descriptive_summary']['complete'])

    def test_missing_before_or_release_never_creates_a_scored_or_released_complete_proof(self):
        with tempfile.TemporaryDirectory() as folder:
            root, context, values, terminals = self.fixture(folder, before=False)
            proof = self.audited(root, context, values, terminals, with_closure=False)
        with self.assertRaises(PermissionError): partial.require_partial(proof)
        report = partial.require_partial(proof, require_release=False)
        self.assertEqual(report['recorded_terminal_trajectories'], 1)
        self.assertEqual(report['completed_terminal_trajectories'], 0)
        self.assertIsNone(report['owned_compute_released'])
        self.assertFalse(report['before_integrity_verified'])
        self.assertEqual(report['completed_arm_descriptive_summary']['known_trajectories'], 0)

    def test_wrong_freeze_optimizer_counts_order_gaps_and_extra_training_are_rejected(self):
        for bad in ['freeze', 'optimizer', 'counts', 'gap', 'order', 'tensor']:
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as folder:
                root, context, values, terminals = self.fixture(folder)
                base = 'trajectories/' + context.stage()['partition']['case_ids'][0] + '/base/'
                if bad == 'freeze': values['PRECALL_FREEZE.json']['source_commit'] = 'c' * 40
                if bad == 'optimizer': values['RUN_RECEIPT.json']['optimizer_updates'] = 1
                if bad == 'counts': values['RUN_RECEIPT.json']['forward_counts'] = {'attempted': 0, 'completed': 0}
                if bad == 'gap': del values[base + 'native_00.json']
                if bad == 'order': del values[base + 'TERMINAL.json']
                if bad == 'tensor': values['unregistered.pt'] = {'public': 'fake'}
                with self.assertRaises((PermissionError, ValueError)):
                    self.audited(root, context, values, terminals)

    def test_native_only_prefix_is_validated_but_never_applied_or_given_a_terminal(self):
        scene = SimpleNamespace(case_id='public', group_id='group')
        native = {'actions': [{'tool': 'stop'}], 'generated_token_ids': [7]}
        cell = {'arm': 'base', 'responses': [], 'scene': {}, 'prefixes': [{'case_id': 'public', 'group_id': 'group',
            'arm': 'base', 'index': 0, 'text': 'public text', 'native': native, 'world_action_not_yet_applied': True}]}
        world = SimpleNamespace(running=True, visible=lambda: 'public text', step=Mock(side_effect=AssertionError('unrecorded action applied')))
        archive = SimpleNamespace(context=SimpleNamespace(stage=lambda: {'endpoint': 'behavior'}))
        with patch('src.composite_bounded_behavior_evaluation_runtime.replay_scope', return_value=nullcontext()), \
             patch('src.composite_bounded_behavior_evaluation_collection.world_for', return_value=world), \
             patch('src.composite_v2_trace_audit._native', return_value='public-cache'):
            count, cache = partial._replay_prefix(archive, scene, cell, object(), {2}, 9, None)
        self.assertEqual((count, cache), (1, 'public-cache')); world.step.assert_not_called()

    def test_count_evidence_keeps_attempted_lower_bounds_and_requires_observed_exit_for_exactness(self):
        files = SimpleNamespace(manifest={'forward_progress/000010.json': 'public'}, json=lambda n: {
            'counts_lower_bound': {'attempted': 11, 'completed': 10}, 'exact_at_exit': False})
        receipt = {'forward_counts': {'attempted': 12, 'completed': 11}, 'worker_exiting': True,
                   'counts_exact_at_controlled_exit': True}
        value = partial._counts(files, {'maximum_backbone_forwards': 100}, receipt, 5, None)
        self.assertEqual(value['attempted_lower_bound'], 12); self.assertEqual(value['completed_lower_bound'], 11)
        self.assertFalse(value['counts_exact_at_controlled_exit'])
        with self.assertRaises(ValueError): partial._counts(files, {'maximum_backbone_forwards': 100}, receipt, 12, None)

    def test_lagging_progress_lower_bound_does_not_reject_newer_native_evidence(self):
        files = SimpleNamespace(manifest={'forward_progress/000030.json': 'public'}, json=lambda n: {
            'counts_lower_bound': {'attempted': 30, 'completed': 30}, 'exact_at_exit': False})
        value = partial._counts(files, {'maximum_backbone_forwards': 100}, None, 32, None)
        self.assertEqual(value['attempted_lower_bound'], 32)
        self.assertEqual(value['completed_lower_bound'], 32)
        self.assertIsNone(value['recorded_terminal_counts'])
        self.assertFalse(value['counts_exact_at_controlled_exit'])

    def test_entirely_unrun_cells_do_not_construct_scenes_or_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            root, context, values, terminals = self.fixture(folder, paired=True)
            stage = values['PRECALL_FREEZE.json']['stage']
            stage['partition']['case_ids'].append(context.roster()[1]['case_id'])
            called = []
            context.stage = lambda: deepcopy(stage)
            def scene(scene, endpoint):
                called.append(scene['case_id'])
                if scene['case_id'] == context.roster()[1]['case_id']:
                    raise AssertionError('unrun scene constructed')
                return SimpleNamespace(**scene)
            # The helper does not call the supplied factory for empty cells.
            with patch('src.composite_bounded_behavior_evaluation_collection.scene_for', side_effect=scene):
                data, checksum = raw_archive(values)
                model = {'metadata': {'tensors': {'lm_head.weight': {'shape': [9, 2]}}}}
                with patch.object(partial, 'validate_source', return_value=context), \
                     patch.object(partial, '_metadata', return_value=(model, {}, {'backbone': None, 'checkpoint': None, 'allocator': None})), \
                     patch.object(partial, '_tokenizer', return_value=nullcontext((object(), {2}))), \
                     patch.object(partial, '_replay_prefix', side_effect=AssertionError('empty prefix replayed')), \
                     patch.object(complete, 'replay_trajectory', side_effect=lambda a,s,arm,t,*args: (terminals[arm],t['responses'],None)):
                    partial.audit_partial_partition(data, checksum, root, 'a'*40, 'fake-fit', None,
                        'fake-data', 'fake-alias-fit', 'fake-alias-qual')
            self.assertEqual(called, [stage['partition']['case_ids'][0]] * 2)

    def test_partial_reader_calls_exact_stage_resource_validator_and_propagates_contradictions(self):
        with tempfile.TemporaryDirectory() as folder:
            root, context, values, terminals = self.fixture(folder)
            with patch('src.composite_bounded_behavior_evaluation_resources.validate_controller_resource',
                       side_effect=PermissionError('actual timeout argument differs')) as validator:
                with self.assertRaisesRegex(PermissionError, 'timeout argument'):
                    self.audited(root, context, values, terminals)
            self.assertEqual(validator.call_args.args[0], context.stage())
            self.assertIsNone(validator.call_args.args[1])
            self.assertEqual(validator.call_args.kwargs, {'allow_unknown': True})

    def test_complete_closure_captures_original_controller_and_cannot_be_used_as_partial(self):
        b = binding(); controller = {**b, 'status': 'worker_exited', 'exit_code': 0, 'timed_out': False,
            'automatic_scientific_retries': 0, 'heldout_content_read': False,
            'child': {'child_started': True, 'child_exit_confirmed': True, 'exit_code': 0},
            'raw_prefix_rescue': {'status': 'owned_scratch_absent'}}
        closure = complete.kaggle_closure(b, 'b'*64,
            b['notebook']+' scriptVersionId=123 successfully ran after 1s',
            'https://www.kaggle.com/code/'+b['notebook']+' 0  Active Events Draft Session off', controller, ['code'], ['code'])
        controller['exit_code'] = 9
        self.assertEqual(closure._value['controller_observation']['exit_code'], 0)
        with tempfile.TemporaryDirectory() as folder:
            root, context, values, terminals = self.fixture(folder)
            data, checksum = raw_archive(values)
            with patch.object(partial, 'validate_source', return_value=context):
                with self.assertRaises(PermissionError):
                    partial.audit_partial_partition(data, checksum, root, 'a'*40, 'fit', None,
                        'data', 'alias-fit', 'alias-qual', closure)


if __name__ == '__main__': unittest.main()
