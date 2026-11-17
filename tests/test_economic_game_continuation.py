"""CPU-only continuation tests: synthetic data; no model, endpoint or real case."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src import economic_game_continuation as continuation


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PrefixFixture:
    def __init__(self, root):
        self.root = Path(root)
        self.plan = {'schema': 'research4.replica-identity-economic-four-condition-48h.held.v1',
            'execution_enabled': False, 'models': [f'synthetic_role_{i}' for i in range(4)],
            'seeds': [101, 102, 103], 'comparison_contrasts': [], 'batches': []}
        for i in range(12):
            self.plan['batches'].append({'models': [f'synthetic_role_{i % 4}'], 'seed': i + 101,
                'max_decisions': 256, 'samples': [{'case_id': f'synthetic_case_{i + 1}',
                    'model_id': f'synthetic_role_{i % 4}', 'seed': i + 101}]})
        self.write('PLAN.json', self.plan)
        self.plan_hash = sha(self.root / 'PLAN.json')
        self.write('DISPLAY_METADATA.json', {'controller_pid': 999999999, 'controller_created_epoch': 1.0})
        self.cells = [self.cell(0, 'audited_complete'), self.cell(1, 'incomplete')]
        self.receipts = [{'batch': 'batch_01', 'exit_code': 0, 'failure': None},
                         {'batch': 'batch_02', 'exit_code': 1,
                          'failure': {'type': 'provider_or_harness_incomplete', 'automatic_retry': False}}]
        self.write('RESULTS.json', {'cells': self.cells, 'batches': self.receipts})
        self.write('INDEPENDENT_STATE_COLLECTION_STOP.json', {'stopped_grant_sha256': 'a' * 64,
            'game_plan_sha256': self.plan_hash, 'reason': 'Incomplete child or handoff denial; separate Root review required',
            'root_window_release_confirmed': False})
        for i, count in ((0, 3), (1, 2)):
            folder = self.root / f'batch_{i + 1:02d}'; folder.mkdir()
            manifest = {**copy.deepcopy(self.plan['batches'][i]), 'started_utc': '2026-01-01T00:00:00+00:00',
                        'deadline_utc': '2026-01-01T04:00:00+00:00'}
            self.write(f'batch_{i + 1:02d}/MANIFEST.json', manifest)
            self.write(f'batch_{i + 1:02d}/CLOSEOUT.json', {'failure': self.receipts[i]['failure'],
                'source_unchanged': True, 'transport_plan_sha256': self.plan_hash,
                'scientific_plan_sha256': self.plan_hash})
            self.write(f'batch_{i + 1:02d}/PUBLISHED_GAME_LOGGING_RECEIPT.json', {'synthetic': True})
            self.write(f'batch_{i + 1:02d}/NATIVE_TRANSCRIPT_RECEIPT.json', {'synthetic': True})
            (folder / 'EVENTS.jsonl').write_text('', encoding='utf-8')
            records = [{'event': 'HTTP_ATTEMPT', 'request_id': f'{i:01x}{j:031x}',
                'scientific_plan_sha256': self.plan_hash, 'window_sha256': 'a' * 64,
                'generation_identity': {'case_id': f'synthetic_case_{i + 1}', 'decision': j, 'attempt': 1}}
                for j in range(count)]
            (folder / 'REQUEST_GUARD_EVIDENCE.jsonl').write_text(
                ''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')
        self.release = {'schema': 'research4.economic-endpoint-owned-release-after-incomplete-case.v1',
            'status': 'PASS_NATIVE_DRAIN_AND_OWNED_ENDPOINT_RELEASED',
            'processes': {'qualifier': {'pid': 101, 'same_owned_identity_present': False, 'status': 'absent'},
                          'service': {'pid': 102, 'same_owned_identity_present': False, 'status': 'absent'}},
            'leases_exist': {'synthetic_inference.lock': False, 'synthetic_service.lock': False,
                             'synthetic_control.lock': False},
            'known_model_workers': [], 'HTTP_policy_phase': 'CLOSED', 'Mac_port8769_connect_result': 61,
            'closeout': {'path': '.local/synthetic-closeout.json', 'sha256': 'b' * 64},
            'closeout_errors': [], 'actual_native_request_proof': {'path': '.local/synthetic-native.json', 'sha256': 'c' * 64},
            'Root_observed_absence_before_cleanup_deadline': True, 'case2_status': 'INCOMPLETE_UNKNOWN',
            'case1_preserved': True, 'new_Game_or_model_requests': 0,
            'original_clock_and_failure_evidence_preserved': True}
        self.native = {'schema': 'research4.economic-incomplete-request-native-proof.v1',
            'request_id': f'{1:01x}{1:031x}', 'used_Game_requests': 5,
            'finish_reason': 'length', 'EOS_detected': False, 'output_tokens': 8192, 'output_limit': 8192,
            'KV_cache_empty': True, 'BASE_restored_and_verified': True,
            'completion_ack': {'worker_started': True, 'worker_finished': True, 'backend_barrier_done': True,
                               'post_checks_passed': True, 'native_errors': []},
            'pending_http_requests': 0, 'active_native_ticket': None, 'case_outcome': 'INCOMPLETE_UNKNOWN'}
        def audit_batch(batch, directory, exit_code, **kwargs):
            index = int(Path(directory).name[-2:]) - 1
            return [copy.deepcopy(self.cells[index])], copy.deepcopy(self.receipts[index])
        self.audit = Mock(side_effect=audit_batch)

    def cell(self, index, status):
        return {'model_id': f'synthetic_role_{index % 4}', 'seed': index + 101,
                'evidence_status': status, 'prefix_replay_passed': True}

    def write(self, path, value):
        (self.root / path).write_text(json.dumps(value, indent=2), encoding='utf-8')

    @staticmethod
    def read_json(path):
        return json.loads(Path(path).read_bytes())

    def read_root_ref(self, reference):
        if reference != self.release['actual_native_request_proof']:
            raise ValueError('Unknown synthetic pinned Root reference')
        return copy.deepcopy(self.native)

    def validate(self):
        with patch.object(continuation.runner, 'augment_audited_cells', side_effect=lambda batch, part, rows: rows):
            return continuation.validate_prefix(self.plan, self.root, self.release,
                read_json=self.read_json, audit_batch=self.audit, read_root_ref=self.read_root_ref)


class EconomicContinuationPrefixTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f = PrefixFixture(self.tmp.name)

    def test_valid_prefix_preserves_plan_and_all_12_case_hashes(self):
        result = self.f.validate()
        self.assertEqual(result['next_index'], 2)
        self.assertEqual(result['prior_dispatches'], 5)
        self.assertEqual(result['scope']['plan_sha256'], self.f.plan_hash)
        self.assertEqual(result['scope']['case_sha256'], [hashlib.sha256(canonical(b)).hexdigest()
                                                           for b in self.f.plan['batches']])
        self.assertEqual(len(result['scope']['fingerprint']), 64)
        self.assertEqual(self.f.audit.call_count, 2)

    def test_any_later_existing_batch_refuses_duplicate_or_rerun(self):
        for index in range(3, 13):
            with self.subTest(index=index):
                target = self.f.root / f'batch_{index:02d}'
                target.mkdir()
                with self.assertRaises(ValueError): self.f.validate()
                target.rmdir()

    def test_live_original_collector_is_rejected(self):
        import psutil
        self.f.write('DISPLAY_METADATA.json', {'controller_pid': os.getpid(),
                     'controller_created_epoch': psutil.Process().create_time()})
        with self.assertRaises(ValueError): self.f.validate()

    def test_requires_exact_12_case_plan_and_matching_plan_bytes(self):
        self.f.plan['batches'].pop()
        with self.assertRaises(ValueError): self.f.validate()
        self.f.plan['batches'].append({'samples': [{'case_id': 'tampered'}]})
        with self.assertRaises(ValueError): self.f.validate()

    def test_plan_argument_must_match_exact_saved_bytes(self):
        self.f.plan['batches'][5]['seed'] += 1
        with self.assertRaises(ValueError): self.f.validate()

    def test_first_complete_second_incomplete_and_audited(self):
        for index, field, value in ((0, 'evidence_status', 'incomplete'), (1, 'evidence_status', 'audited_complete'),
                                    (1, 'prefix_replay_passed', False), (0, 'model_id', 'different')):
            with self.subTest(index=index, field=field):
                data = {'cells': copy.deepcopy(self.f.cells), 'batches': copy.deepcopy(self.f.receipts)}
                data['cells'][index][field] = value
                self.f.write('RESULTS.json', data)
                with self.assertRaises(ValueError): self.f.validate()
        self.f.write('RESULTS.json', {'cells': self.f.cells, 'batches': self.f.receipts})

    def test_incomplete_closeout_must_match_exact_failure_and_original_plan(self):
        closeout = self.f.read_json(self.f.root / 'batch_02/CLOSEOUT.json')
        for field, value in (('failure', None), ('source_unchanged', False),
                             ('scientific_plan_sha256', 'f' * 64), ('transport_plan_sha256', 'f' * 64)):
            with self.subTest(field=field):
                altered = copy.deepcopy(closeout); altered[field] = value
                self.f.write('batch_02/CLOSEOUT.json', altered)
                with self.assertRaises(ValueError): self.f.validate()
        self.f.write('batch_02/CLOSEOUT.json', closeout)

    def test_requires_actual_native_release_and_no_workers_or_leases(self):
        for mutate in (lambda x: x.update(status='NOT_RELEASED'),
                       lambda x: x['processes']['service'].update(same_owned_identity_present=True),
                       lambda x: x['leases_exist'].update({'synthetic_service.lock': True}),
                       lambda x: x.update(known_model_workers=[{'pid': 11}]),
                       lambda x: x.update(HTTP_policy_phase='OPEN'),
                       lambda x: x.update(actual_native_request_proof=None),
                       lambda x: x.update(closeout_errors=['synthetic']),
                       lambda x: x.update(case2_status='COMPLETE')):
            with self.subTest(mutation=mutate):
                saved = copy.deepcopy(self.f.release); mutate(self.f.release)
                with self.assertRaises(ValueError): self.f.validate()
                self.f.release = saved

    def test_counts_unacknowledged_attempts_and_rejects_over_budget(self):
        # An HTTP_ATTEMPT counts even if no matching ACK exists.
        result = self.f.validate()
        self.assertEqual(result['prior_dispatches'], 5)
        path = self.f.root / 'batch_02/REQUEST_GUARD_EVIDENCE.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        with path.open('a', encoding='utf-8') as stream:
            for j in range(2, 510):
                row = copy.deepcopy(rows[0]); row['request_id'] = f'{1:01x}{j:031x}'
                row['generation_identity']['decision'] = j
                stream.write(json.dumps(row) + '\n')
        with self.assertRaises(ValueError): self.f.validate()

    def test_duplicate_attempt_or_wrong_case_identity_is_refused(self):
        path = self.f.root / 'batch_02/REQUEST_GUARD_EVIDENCE.jsonl'
        original = path.read_text()
        path.write_text(original + original.splitlines()[0] + '\n')
        with self.assertRaises(ValueError): self.f.validate()
        rows = [json.loads(line) for line in original.splitlines()]
        rows[0]['generation_identity']['case_id'] = 'different'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        with self.assertRaises(ValueError): self.f.validate()

    def test_wrong_plan_or_retry_or_duplicate_decision_is_refused(self):
        path = self.f.root / 'batch_02/REQUEST_GUARD_EVIDENCE.jsonl'
        original = [json.loads(line) for line in path.read_text().splitlines()]
        for mutate in (lambda r: r[0].update(scientific_plan_sha256='f' * 64),
                       lambda r: r[0]['generation_identity'].update(attempt=2),
                       lambda r: r[1]['generation_identity'].update(decision=0),
                       lambda r: r[0]['generation_identity'].update(decision=256)):
            with self.subTest(mutation=mutate):
                rows = copy.deepcopy(original); mutate(rows)
                path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
                with self.assertRaises(ValueError): self.f.validate()

    def test_native_request_count_and_final_request_must_match(self):
        for mutate in (lambda p: p.update(used_Game_requests=4),
                       lambda p: p.update(request_id='f' * 32),
                       lambda p: p.update(EOS_detected=True),
                       lambda p: p.update(KV_cache_empty=False),
                       lambda p: p.update(BASE_restored_and_verified=False),
                       lambda p: p.update(active_native_ticket='still-active'),
                       lambda p: p.update(pending_http_requests=1),
                       lambda p: p['completion_ack'].update(post_checks_passed=False),
                       lambda p: p['completion_ack'].update(native_errors=['error']),
                       lambda p: p.update(case_outcome='complete')):
            with self.subTest(mutation=mutate):
                prior = copy.deepcopy(self.f.native); mutate(self.f.native)
                with self.assertRaises(ValueError): self.f.validate()
                self.f.native = prior

    def test_prefix_decisions_must_be_nonempty_and_contiguous(self):
        path = self.f.root / 'batch_01/REQUEST_GUARD_EVIDENCE.jsonl'
        original = [json.loads(line) for line in path.read_text().splitlines()]
        path.write_text(''.join(json.dumps(row) + '\n' for row in (original[0], original[2])))
        self.f.native['used_Game_requests'] = 4
        with self.assertRaises(ValueError): self.f.validate()
        path.write_text('')
        self.f.native['used_Game_requests'] = 2
        with self.assertRaises(ValueError): self.f.validate()

    def test_continuation_runs_only_third_case_and_preserves_prefix_when_it_is_incomplete(self):
        for name in ('SUMMARY.json', 'CANDIDATE_COHORT_LEDGER.json'):
            self.f.write(name, {'synthetic': name})
        before = {name: (self.f.root / name).read_bytes() for name in
                  ('RESULTS.json', 'SUMMARY.json', 'CANDIDATE_COHORT_LEDGER.json',
                   'DISPLAY_METADATA.json', 'INDEPENDENT_STATE_COLLECTION_STOP.json')}
        root = self.f.root
        source_hash = hashlib.sha256(continuation.SOURCE_PATH.read_bytes()).hexdigest()
        proof_ref = {'path': '.local/synthetic-release.json', 'sha256': 'd' * 64}
        bare_scope = {'schema': continuation.SCOPE_SCHEMA,
                      'prefix_evidence_sha256': {'results': sha(root/'RESULTS.json')}}
        raw_scope = {**bare_scope, 'release_proof_ref': proof_ref}
        self.f.write('SCOPE.json', raw_scope)
        review = {'status': 'PASS_SYNTHETIC', 'original_plan_sha256': self.f.plan_hash,
                  'continuation_scope_sha256': sha(root/'SCOPE.json'), 'continuation_source_sha256': source_hash}
        self.f.write('REVIEW.json', review)
        class FakeConsumer:
            pins = {'expected_window_sha256': 'e' * 64}
            contract = type('Contract', (), {'BASE': 'synthetic_base'})()
            def _validated_scientific_amendment(self):
                return {'main_plan_ref': 'plan', 'reviews': {'Main_source_ref': 'review'}}
            def _main_reference(self, reference):
                if reference == 'plan': return root/'PLAN.json', self_plan
                if reference == 'scope': return root/'SCOPE.json', raw_scope
                if reference == 'review': return root/'REVIEW.json', review
                raise ValueError('unexpected reference')
            def _reference(self, reference): return None, self_release
            def require_plan(self, *args): pass
            def validate_request(self, *args): pass
            def private_child_context(self, *args, **kwargs): return '{}'
            def remaining_global_seconds(self): return 60
        self_plan, self_release = self.f.plan, self.f.release
        process = Mock(); process.wait.return_value = 1; process.poll.return_value = 1
        incomplete = {'model_id': 'synthetic_role_2', 'seed': 103, 'evidence_status': 'incomplete'}
        receipt = {'batch': 'batch_03', 'exit_code': 1,
                   'failure': {'type': 'provider_or_harness_incomplete', 'automatic_retry': False}}
        with patch.object(continuation.runner, '_consumer_type', return_value=FakeConsumer), \
                patch.object(continuation, 'validate_prefix', return_value={'scope': bare_scope}), \
                patch.object(continuation.economic, 'source_matches', return_value=True), \
                patch.object(continuation.comparison, 'claim_security500_grant'), \
                patch.object(continuation.comparison, 'stop_security500_collection') as stop, \
                patch.object(continuation.comparison, 'audit_batch', return_value=([incomplete], receipt)), \
                patch.object(continuation.runner, 'augment_audited_cells', side_effect=lambda batch, part, rows: rows), \
                patch.object(continuation.runner, 'write_progress'), patch.object(continuation.compact, '_ledger'), \
                patch.object(continuation, 'summarize', return_value={}), \
                patch.object(continuation.economic, 'publish_closed_batch', return_value={'entries': []}), \
                patch('src.native_generation_transcript.run_decoder_subprocess', return_value={}), \
                patch.object(continuation.subprocess, 'Popen', return_value=process):
            with self.assertRaises(ValueError): continuation.execute('scope', consumer=FakeConsumer())
        self.assertTrue((root/'batch_03').is_dir())
        self.assertFalse((root/'batch_04').exists())
        stop.assert_called_once()
        for name in ('RESULTS.json', 'SUMMARY.json', 'CANDIDATE_COHORT_LEDGER.json'):
            self.assertEqual((root/('BEFORE_CONTINUATION_' + name)).read_bytes(), before[name])
        self.assertEqual((root/'DISPLAY_METADATA_BEFORE_CONTINUATION.json').read_bytes(), before['DISPLAY_METADATA.json'])
        self.assertEqual((root/'INDEPENDENT_STATE_COLLECTION_STOP.json').read_bytes(), before['INDEPENDENT_STATE_COLLECTION_STOP.json'])
        self.assertEqual(json.loads((root/'RESULTS.json').read_bytes())['cells'][:2], self.f.cells)


if __name__ == '__main__': unittest.main()
