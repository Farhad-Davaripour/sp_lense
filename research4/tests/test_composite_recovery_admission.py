import copy
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from src import composite_recovery_admission as admission
from src.composite_recovery_audit import audit_source_and_config, audit_detector_records

ROOT = Path(__file__).resolve().parents[1]


class RecoveryAdmissionTests(unittest.TestCase):
    def fixture(self, root):
        ledger = {'candidates': {admission.CANDIDATE: {
            'scriptVersionId': 361312460, 'status': 'scientific_timeout_partial_checkpoint_audited',
            'owned_compute_released': True, 'raw_audit_pending': False}}}
        (root / 'SCIENTIFIC_EXECUTION_LEDGER.json').write_text(json.dumps(ledger))
        binding = {'source_commit': 'a' * 40, 'notebook': 'farhad89/recovery-fixture',
                   'package_sha256': 'b' * 64, 'authorization_sha256': 'c' * 64,
                   'parent_run_id': '361312460',
                   'human_message_evidence_locator': 'synthetic-model-free-unit-test-not-cloud-authority'}
        return binding

    def live(self):
        return {'at_UTC': datetime.now(timezone.utc).isoformat(),
                'surfaces': {name: 0 for name in admission.REQUIRED_SURFACES}}

    def test_consumption_precedes_submit_and_same_version_reconciliation_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); binding = self.fixture(root)
            with patch.object(admission, 'validate_preparation', return_value=binding):
                attempt = admission.reserve_once(root, 'fixture', self.live)
            ledger = json.loads((root / 'SCIENTIFIC_EXECUTION_LEDGER.json').read_text())
            self.assertTrue(ledger['candidates'][admission.CANDIDATE]['authorized_recovery_attempts'][0]['authority_consumed'])
            self.assertIsNone(ledger['active_owned_submission']['scriptVersionId'])
            self.assertTrue((root / 'COMPUTE_SUBMISSION.lock').exists())
            with self.assertRaises(FileExistsError):
                admission.begin_reconciliation(root)
            recorded = admission.reconcile_submission(root, attempt['identity'], 1234, attempt['guard_token'])
            self.assertEqual(recorded['scriptVersionId'], 1234)
            self.assertFalse((root / 'COMPUTE_SUBMISSION.lock').exists())
            self.assertEqual(admission.reconcile_submission(root, attempt['identity'], 1234)['scriptVersionId'], 1234)
            with self.assertRaises(PermissionError):
                admission.reconcile_submission(root, attempt['identity'], 5678)
            with patch.object(admission, 'validate_preparation', return_value=binding):
                with self.assertRaises(PermissionError):
                    admission.reserve_once(root, 'fixture', self.live)
            self.assertEqual(len(json.loads((root / 'SCIENTIFIC_EXECUTION_LEDGER.json').read_text())['candidates'][admission.CANDIDATE]['authorized_recovery_attempts']), 1)

    def test_unknown_submit_cannot_consume_a_second_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); binding = self.fixture(root)
            with patch.object(admission, 'validate_preparation', return_value=binding):
                attempt = admission.reserve_once(root, 'fixture', self.live)
                unknown = admission.reconcile_submission(root, attempt['identity'], guard_token=attempt['guard_token'])
                self.assertEqual(unknown['status'], 'submission_outcome_unknown')
                with self.assertRaises(PermissionError):
                    admission.reserve_once(root, 'fixture', self.live)
            resolved = admission.reconcile_submission(root, attempt['identity'], 1234)
            self.assertEqual(resolved['identity'], attempt['identity'])

    def test_stale_pre_guard_unknown_or_active_live_evidence_reject_without_consumption(self):
        for kind in ['stale', 'pre_guard', 'missing', 'unknown', 'active']:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); binding = self.fixture(root); guard = admission.begin_reconciliation(root)
                snapshot = self.live()
                if kind in ['stale', 'pre_guard']:
                    snapshot['at_UTC'] = (datetime.now(timezone.utc)-timedelta(seconds=180 if kind == 'stale' else 1)).isoformat()
                elif kind == 'missing': snapshot['surfaces'].pop('centralus')
                elif kind == 'unknown': snapshot['surfaces']['kaggle'] = None
                else: snapshot['surfaces']['kaggle'] = 1
                with patch.object(admission, 'validate_preparation', return_value=binding):
                    with self.assertRaises(PermissionError):
                        admission.reserve_once(root, 'fixture', snapshot, guard['guard_token'])
                ledger = json.loads((root / 'SCIENTIFIC_EXECUTION_LEDGER.json').read_text())
                self.assertNotIn('authorized_recovery_attempts', ledger['candidates'][admission.CANDIDATE])
                self.assertFalse((root / 'COMPUTE_SUBMISSION.lock').exists())

    def test_proposal_rejection_never_creates_a_reservation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self.fixture(root)
            (root / 'PREPARATION.json').write_text(json.dumps({'status': 'proposal_requires_explicit_human_retry_and_bound_exception'}))
            before = (root / 'SCIENTIFIC_EXECUTION_LEDGER.json').read_bytes()
            with self.assertRaises(PermissionError):
                admission.reserve_once(root, root / 'PREPARATION.json', self.live)
            self.assertEqual((root / 'SCIENTIFIC_EXECUTION_LEDGER.json').read_bytes(), before)
            self.assertFalse((root / 'COMPUTE_SUBMISSION.lock').exists())

    def test_original_source_and_only_authorized_execution_delta_are_auditable(self):
        original = json.loads((ROOT / 'config/composite_controller_rank32_v2.json').read_text())
        actual = copy.deepcopy(original)
        actual['bounds'].update(worker_seconds=4200, hard_timeout_minutes=75)
        actual['execution_recovery'] = {'profile': 'kaggle-recognition-complete-75min-v1',
                                        'kind': 'one_explicitly_authorized_fresh_complete_run',
                                        'exact_continuation': False}
        receipt = {'source_commit': 'new-reviewed-source'}
        freeze = {'source_commit': 'new-reviewed-source', 'config': actual}
        proof = audit_source_and_config(receipt, freeze, 'new-reviewed-source', original, actual)
        self.assertTrue(proof['stripping_execution_fields_recovers_original_config'])
        with self.assertRaises(ValueError):
            audit_source_and_config(receipt, freeze, 'old-source', original, actual)
        with self.assertRaises(ValueError):
            audit_source_and_config(receipt, freeze, 'new-reviewed-source', original, original)
        changed = copy.deepcopy(actual); changed['modernbert']['rank'] = 16
        with self.assertRaisesRegex(ValueError, 'scientific setting'):
            audit_source_and_config(receipt, {**freeze, 'config': changed}, 'new-reviewed-source', original, changed)
        proof = audit_source_and_config({'source_commit': 'old'}, {'source_commit': 'old', 'config': original}, 'old', original, original)
        self.assertFalse(proof['authorized_recovery_execution_delta'])

    def test_raw_metric_replay_rejects_wrong_case_family_and_reported_metrics(self):
        config = json.loads((ROOT / 'config/composite_controller_rank32_v2.json').read_text())
        rows = []; records = []
        for index, gate in enumerate([True, False]):
            labels = {'own_status': 'active' if gate else 'clear', 'peer_status': 'clear',
                      'event': 'threat' if gate else 'none', 'affected': 'self' if gate else 'none', 'gate': gate}
            row = {'state_id': str(index), 'case_id': 'case-' + str(index), 'template': 3,
                   'text': 'abc', 'sequence': 0, 'terminal': False, 'labels': labels, 'evidence_chars': [0, 3]}
            prediction = {'own_status': labels['own_status'], 'evidence_start': 0, 'evidence_end': 3,
                          'raw': {'labels': {key: value for key, value in labels.items() if key != 'gate'}}, 'test_gate': gate}
            rows.append(row); records.append({'state_id': row['state_id'], 'case_id': row['case_id'],
                'template': 3, 'labels': labels, 'expected_gate': gate, 'raw_gate': gate,
                'prediction': prediction, 'evidence_char_f1': 1.0})
        field = {key: 1.0 for key in ['own_status', 'peer_status', 'event', 'affected']}
        family = {'states': 2, 'field_accuracy': field, 'false_own_on': 0,
                  'active_own_recall': 1.0, 'evidence_char_f1': 1.0, 'clearing_errors': 0, 'admitted': True}
        metrics = {'states': 2, 'field_accuracy': field, 'false_own_on': 0,
                   'active_own_recall': 1.0, 'evidence_char_f1': 1.0, 'active_own_states': 1,
                   'resolution_false_on': 0, 'operator_stop_false_on': 0,
                   'unknown_or_low_confidence_states': 0, 'families': {'3': family}}
        value = {'threshold': 0.5, 'records': records, 'metrics': metrics, 'admitted': True}
        class Memory:
            def reset(self, *args): pass
            def observe(self, case, sequence, prediction, *args): self.raw_gate = prediction.test_gate
        calls = []
        raw_decode = lambda *args: calls.append(args)
        result = audit_detector_records(value, rows, config, raw_decode, Memory, SimpleNamespace)
        self.assertTrue(result['admitted']); self.assertEqual(len(calls), 2)
        for mutation in ['case', 'family', 'metric', 'evidence']:
            changed = copy.deepcopy(value)
            if mutation == 'case': changed['records'][0]['case_id'] = 'another-case'
            elif mutation == 'family': changed['records'][0]['template'] = 4
            elif mutation == 'metric': changed['metrics']['active_own_recall'] = 0.5
            else: changed['records'][0]['evidence_char_f1'] = 0.5
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                audit_detector_records(changed, rows, config, raw_decode, Memory, SimpleNamespace)


if __name__ == '__main__':
    unittest.main()
