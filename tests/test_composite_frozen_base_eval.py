import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src import composite_frozen_base_eval as evaluation
from src.composite_memory import Prediction
from src.composite_world import fixture
from src.composite_exports import make_evidence_archive
from src.io_utils import sha


class BaseEvaluationTests(unittest.TestCase):
    def model(self, saved):
        class Detector(nn.Module):
            def predict(self, text, threshold):
                return Prediction('unknown', 'unknown', 'unknown', 'unknown', 0., 0, 0, '')
        class Actor(nn.Module):
            def clear_cache(self): pass
            def generate(self, text, on):
                if on: raise AssertionError('An intervention reached the synthetic base actor')
                return {'actions': None, 'adapter_enabled': False, 'raw_text': 'synthetic invalid response'}
        return evaluation.RecordingBaseComposite(Detector(), Actor(), 0.9,
                                                  lambda name, value: saved.__setitem__(name, copy.deepcopy(value)))

    def test_original_rollout_records_all_observations_and_enforces_stop(self):
        saved = {}; scene = fixture('quiet_pending', 3, seed='base-unit')
        model = self.model(saved); verified = []; progress = []
        context = {'scenes': [asdict(scene)], 'evaluation_plan': {'conditions': ['base'], 'case_ids': [scene.case_id]}}
        report = evaluation.evaluate_partition(model, context, lambda: None,
            lambda name, value: saved.__setitem__(name, copy.deepcopy(value)),
            lambda *args, **kwargs: progress.append((args, kwargs)), lambda: verified.append(True))
        self.assertTrue(report['partition_complete']); self.assertEqual(verified, [True])
        trace = saved['trajectories/'+scene.case_id+'.json']
        self.assertFalse(trace['outcome']['running'])
        self.assertEqual(trace['outcome']['calls_after_terminal'], 0)
        self.assertTrue(all(step['response']['selected_gate'] is False for step in trace['trace']))
        self.assertIn('observations/'+scene.case_id+'_000.json', saved)
        self.assertEqual(len([name for name in saved if name.startswith('steps/')]), len(trace['trace']))
        self.assertFalse(report['full_validation_admitted'])
        with self.assertRaises(PermissionError): model.respond(None, 'always_on')

    def test_exception_retains_initial_observation_and_missing_case(self):
        saved = {}; scene = fixture('quiet_pending', 3, seed='base-failed-unit'); model = self.model(saved)
        context = {'scenes': [asdict(scene)], 'evaluation_plan': {'conditions': ['base'], 'case_ids': [scene.case_id]}}
        calls = []; verified = []
        def guard():
            calls.append(True)
            if len(calls) == 2: raise TimeoutError('synthetic protected export stop')
        with self.assertRaises(TimeoutError):
            evaluation.evaluate_partition(model, context, guard,
                lambda name, value: saved.__setitem__(name, copy.deepcopy(value)), lambda *a, **k: None,
                lambda: verified.append(True))
        self.assertIn('observations/'+scene.case_id+'_000.json', saved)
        self.assertEqual(saved['FROZEN_BASE_REPORT.json']['missing_case_ids'], [scene.case_id])
        self.assertEqual(verified, [True])

    def test_detector_loader_rejects_selection_change_even_with_new_archive_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cfg = {'modernbert': {'synthetic': 'model-free'}}
            for threshold in [0.9, 0.8]:
                folder = root/str(threshold); folder.mkdir()
                checkpoint = folder/'detector_best.pt'
                torch.save({'config': cfg['modernbert'], 'head_mode': 'role_pool',
                            'validation_admitted': True, 'threshold': threshold, 'updates': 600}, checkpoint)
                (folder/'RUN_RECEIPT.json').write_text(json.dumps({'status': 'complete'}))
                archive = root/(str(threshold)+'.zip'); make_evidence_archive(folder, archive)
                class Store:
                    def get(self, name): return archive.read_bytes()
                with patch.object(evaluation, 'DETECTOR_ARCHIVE_SHA256', sha(archive)), \
                     patch.object(evaluation, 'DETECTOR_CHECKPOINT_SHA256', sha(checkpoint)):
                    if threshold == 0.9:
                        self.assertEqual(evaluation.selected_detector(Store(), cfg)['updates'], 600)
                    else:
                        with self.assertRaisesRegex(ValueError, 'selection'):
                            evaluation.selected_detector(Store(), cfg)


if __name__ == '__main__': unittest.main()
