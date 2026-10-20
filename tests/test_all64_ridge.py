"""All64 uses the ridge-only state and an unchanged finite validation panel."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import struct
import tempfile
import unittest
from unittest.mock import patch
from src.all64_ridge import configuration, inspect_initializer, validate_admission, summarize
from src.config import ROOT
from src.utils import write_json
import yaml


class All64RidgeTests(unittest.TestCase):
    def test_only_one_site_and_no_gradient_refinement_are_admitted(self):
        config = configuration()
        self.assertEqual(config['method']['blocks'], 64)
        self.assertEqual(config['method']['sites_per_block'], 1)
        self.assertEqual(config['method']['gradient_refinement_updates'], 0)
        self.assertEqual(config['initializer']['fitted_functions'], 64)
        self.assertFalse(config['comparison']['heldout_execution'])
        job = yaml.safe_load((ROOT / 'resources/all64_ridge.job.yml').read_bytes())['resources']['jobs']['all64_ridge']
        self.assertEqual(job['timeout_seconds'], config['bounds']['whole_seconds'])
        self.assertTrue(all(task['max_retries'] == 0 for task in job['tasks']))

    def test_a_changed_or_refined_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'initializer.safetensors'
            header = {f'{i}/output.{factor}': {'dtype': 'F32', 'shape': [5120, 102]}
                      for i in range(64) for factor in ('down', 'up')}
            raw_header = json.dumps(header).encode()
            raw = struct.pack('<Q', len(raw_header)) + raw_header
            path.write_bytes(raw)
            config = {'initializer': {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}}
            self.assertEqual(inspect_initializer(path, config)['functions'], 64)
            path.write_bytes(raw[:-1] + b'x')
            with self.assertRaisesRegex(ValueError, 'ridge-only initializer'):
                inspect_initializer(path, config)

    def test_expired_or_extended_admission_is_rejected(self):
        start = dt.datetime.now(dt.timezone.utc)
        admission = {'phase': 'all64_ridge', 'Free_Edition_only': True, 'automatic_retries': 0,
                     'submitted_utc': start.isoformat(),
                     'absolute_end_utc': (start + dt.timedelta(hours=3)).isoformat()}
        validate_admission(admission)
        admission['absolute_end_utc'] = (start + dt.timedelta(hours=4)).isoformat()
        with self.assertRaisesRegex(ValueError, 'extend or reset'):
            validate_admission(admission)
        admission['submitted_utc'] = (start - dt.timedelta(hours=4)).isoformat()
        admission['absolute_end_utc'] = (start - dt.timedelta(hours=1)).isoformat()
        with self.assertRaises(TimeoutError):
            validate_admission(admission)

    def test_stopped_cases_are_not_reported_as_complete_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / 'ADMISSION.json', {'packet_zip': 'frozen', 'packet_sha256': 'hash'})
            write_json(root / 'ridge_all64_one_site/episodes/case.json', {
                'case_id': 'case', 'completed': False, 'runtime_fault': {'type': 'TimeoutError'},
                'score': {'negative_control': False, 'workflow_success': False}})
            packet = {'cases.json': [{'id': 'case', 'condition': 'self_unfinished'}]}
            with patch('src.matched_validation.read_packet', return_value=packet):
                result = summarize(root)
            self.assertEqual(result['status'], 'ALL64_RIDGE_STOPPED_PARTIAL')
            self.assertEqual(result['groups']['own']['completed'], 0)
            self.assertEqual(result['groups']['own']['records'], 1)
            self.assertEqual(result['unexecuted'], 23)
            self.assertFalse(result['new_teacher_arm_executed'])
