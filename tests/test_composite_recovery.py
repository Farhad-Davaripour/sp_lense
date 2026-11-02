import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.composite_recovery import execution_config, PROFILE

ROOT = Path(__file__).resolve().parents[1]


class RecoveryTests(unittest.TestCase):
    def fixture(self, root):
        (root / 'inputs').mkdir()
        (root / 'config').mkdir()
        config = json.loads((ROOT / 'config/composite_controller_rank32_v2.json').read_text())
        for name in ['composite_controller_rank32_v2.json', 'composite_detector_rank32_v5.json']:
            (root / 'config' / name).write_bytes((ROOT / 'config' / name).read_bytes())
        pin = {'platform': 'kaggle', 'source_commit': 'a' * 40,
               'hard_timeout_seconds': 4500, 'notebook': 'farhad89/proposed-recovery',
               'original_scientific_input_manifest_sha256': 'frozen-science-inputs',
               'transport_manifest_sha256': 'frozen-transport',
               'recovery': {'profile': PROFILE, 'parent_run_id': '361312460',
                            'authorization_file': 'RECOVERY_AUTHORIZATION.json'}}
        auth = {'status': 'explicit_human_exception', 'authorized_by': 'human_user',
                'human_message_evidence_locator': 'synthetic-test-message-never-authorizes-compute',
                'profile': PROFILE, 'parent_run_id': '361312460', 'source_commit': 'a' * 40,
                'notebook': pin['notebook'],
                'original_scientific_input_manifest_sha256': pin['original_scientific_input_manifest_sha256'],
                'transport_manifest_sha256': pin['transport_manifest_sha256'],
                'maximum_fresh_recovery_submissions': 1,
                'fresh_full_run_not_exact_continuation': True,
                'use_selected_checkpoint_weights': False, 'initialization_seed': 202610213,
                'maximum_optimizer_updates': 1200,
                'validation_updates': [200, 400, 600, 800, 1000, 1200],
                'hard_timeout_seconds': 4500, 'worker_seconds': 4200,
                'scientific_visible_GPU_count': 1, 'maximum_backbone_forwards': 30000,
                'automatic_scientific_retries': 0, 'maximum_concurrent_research_runs': 1}
        for field, name in [('recipe_sha256', 'composite_detector_rank32_v5.json'),
                            ('original_protocol_sha256', 'composite_controller_rank32_v2.json')]:
            auth[field] = hashlib.sha256((root / 'config' / name).read_bytes()).hexdigest()
        self.write(root, pin, auth)
        return config, pin, auth

    def write(self, root, pin, auth):
        (root / 'inputs/EXECUTION_PLATFORM.json').write_text(json.dumps(pin))
        (root / 'inputs/RECOVERY_AUTHORIZATION.json').write_text(json.dumps(auth))

    def test_original_profile_and_bound_remain_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, pin, auth = self.fixture(root)
            pin.pop('recovery'); pin['hard_timeout_seconds'] = 2700
            self.write(root, pin, auth)
            self.assertIs(execution_config(root, 'a' * 40, 'detector-rank32-v5', config), config)
            pin['hard_timeout_seconds'] = 4500; self.write(root, pin, auth)
            with self.assertRaises(PermissionError):
                execution_config(root, 'a' * 40, 'detector-rank32-v5', config)

    def test_proposal_and_unbound_exception_fail_before_scientific_stage(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'RESEARCH4_EXECUTION_PLATFORM': 'kaggle'}):
            root = Path(temp); config, pin, auth = self.fixture(root)
            for changed in [{'status': 'proposal_not_authorized'}, {'authorized_by': 'agent'},
                            {'human_message_evidence_locator': None}, {'source_commit': 'other'},
                            {'parent_run_id': 'other'}, {'maximum_fresh_recovery_submissions': 2},
                            {'use_selected_checkpoint_weights': True}, {'maximum_optimizer_updates': 600},
                            {'validation_updates': [200, 400, 600]}, {'scientific_visible_GPU_count': 2},
                            {'automatic_scientific_retries': 1}]:
                with self.subTest(changed=changed):
                    self.write(root, pin, {**auth, **changed})
                    with self.assertRaises(PermissionError):
                        execution_config(root, 'a' * 40, 'detector-rank32-v5', config)

    def test_explicit_synthetic_exception_changes_only_time_bounds(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'RESEARCH4_EXECUTION_PLATFORM': 'kaggle'}):
            root = Path(temp); config, pin, auth = self.fixture(root); original = copy.deepcopy(config)
            result = execution_config(root, 'a' * 40, 'detector-rank32-v5', config)
            self.assertEqual(config, original)
            self.assertEqual(result['bounds']['worker_seconds'], 4200)
            self.assertEqual(result['bounds']['hard_timeout_minutes'], 75)
            self.assertFalse(result['execution_recovery']['exact_continuation'])
            result.pop('execution_recovery')
            result['bounds'].update(worker_seconds=2400, hard_timeout_minutes=45)
            self.assertEqual(result, original)
            for mode in ['behavior-all_blocks', 'evaluation']:
                with self.assertRaises(PermissionError):
                    execution_config(root, 'a' * 40, mode, config)
            with patch.dict(os.environ, {'RESEARCH4_EXECUTION_PLATFORM': 'databricks'}):
                with self.assertRaises(PermissionError):
                    execution_config(root, 'a' * 40, 'detector-rank32-v5', config)

    def test_changed_scientific_recipe_or_original_bounds_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'RESEARCH4_EXECUTION_PLATFORM': 'kaggle'}):
            root = Path(temp); config, pin, auth = self.fixture(root)
            changed = copy.deepcopy(config); changed['bounds']['maximum_backbone_forwards_per_run'] = 60000
            with self.assertRaises(ValueError):
                execution_config(root, 'a' * 40, 'detector-rank32-v5', changed)
            (root / 'config/composite_detector_rank32_v5.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'recipe'):
                execution_config(root, 'a' * 40, 'detector-rank32-v5', config)


if __name__ == '__main__':
    unittest.main()
