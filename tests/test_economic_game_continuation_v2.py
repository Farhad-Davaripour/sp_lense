"""CPU-only tests for the eight-case continuation; no model or study data."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src import economic_game_continuation_v2 as v2
from tests.test_economic_game_continuation import PrefixFixture, canonical


class FourCaseFixture(PrefixFixture):
    def __init__(self, root):
        super().__init__(root)
        self.first_window = 'a' * 64
        self.second_window = 'b' * 64
        self.first_admission = 'd' * 64
        self.second_admission = 'e' * 64
        for i in (0, 1):
            path = self.root/f'batch_{i + 1:02d}'/'REQUEST_GUARD_EVIDENCE.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for row in rows: row['economic_admission_sha256'] = self.first_admission
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        for i, count in ((2, 3), (3, 2)):
            status = 'audited_complete' if i == 2 else 'incomplete'
            self.cells.append(self.cell(i, status))
            failure = None if i == 2 else {'type': 'provider_or_harness_incomplete', 'automatic_retry': False}
            self.receipts.append({'batch': f'batch_{i + 1:02d}', 'exit_code': 0 if i == 2 else 1, 'failure': failure})
            folder = self.root/f'batch_{i + 1:02d}'; folder.mkdir()
            self.write(f'batch_{i + 1:02d}/MANIFEST.json', self.plan['batches'][i])
            self.write(f'batch_{i + 1:02d}/CLOSEOUT.json', {'failure': failure, 'source_unchanged': True,
                'scientific_plan_sha256': self.plan_hash, 'transport_plan_sha256': self.plan_hash})
            self.write(f'batch_{i + 1:02d}/PUBLISHED_GAME_LOGGING_RECEIPT.json', {'synthetic': True})
            self.write(f'batch_{i + 1:02d}/NATIVE_TRANSCRIPT_RECEIPT.json', {'synthetic': True})
            (folder/'EVENTS.jsonl').write_text('')
            records = [{'event': 'HTTP_ATTEMPT', 'request_id': f'{i:01x}{j:031x}',
                'scientific_plan_sha256': self.plan_hash, 'window_sha256': self.second_window,
                'economic_admission_sha256': self.second_admission,
                'generation_identity': {'case_id': f'synthetic_case_{i + 1}', 'decision': j, 'attempt': 1}}
                for j in range(count)]
            (folder/'REQUEST_GUARD_EVIDENCE.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
        self.write('RESULTS.json', {'cells': self.cells, 'batches': self.receipts})
        self.write('CONTINUATION_STOP.json', {'status': 'STOPPED_NO_NEXT_CASE'})
        self.grant_root = self.root/'grants'
        for adm, win in ((self.first_admission,self.first_window),(self.second_admission,self.second_window)):
            (self.grant_root/adm).mkdir(parents=True)
            (self.grant_root/adm/(win+'.STOP.json')).write_text(json.dumps({'window_sha256':win,'status':'STOPPED'}))
        self.latest = {'schema': v2.LATEST_RELEASE_SCHEMA, 'status': v2.LATEST_RELEASE_STATUS,
            'http_policy':'CLOSED', 'matching_model_workers': [], 'mac_port_8769_connect_ex': 61,
            'mac_processes_present': {'1':False,'2':False}, 'mac_locks_present': {'lease':False},
            'closeout_errors': [], 'closeout_reports_both_leases_absent': True,
            'finish_reason':'length', 'completion_tokens':8192, 'max_tokens':8192,'EOS_detected':False,
            'KV_cache_empty':True, 'BASE_restored_and_verified':True, 'adapter_wrappers_after':0,
            'native_errors':[], 'worker_finished':True, 'backend_barrier_done':True, 'post_checks_passed':True,
            'request_id':f'{3:01x}{1:031x}', 'dispatch_ordinal':10}

    def validate(self):
        with patch.object(v2.runner, 'augment_audited_cells', side_effect=lambda batch, part, rows:rows):
            return v2.validate_prefix(self.plan,self.root,self.release,self.latest,read_json=self.read_json,
                audit_batch=self.audit,read_root_ref=self.read_root_ref,grant_root=self.grant_root)


class EconomicContinuationV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.f=FourCaseFixture(self.tmp.name)

    def test_exact_four_prior_cases_and_eight_unstarted(self):
        result=self.f.validate(); scope=result['scope']
        self.assertEqual(result['next_index'],4); self.assertEqual(result['prior_dispatches'],10)
        self.assertEqual(scope['allowed_indices'],list(range(4,12)))
        self.assertEqual(scope['prior_window_sha256'],[self.f.first_window,self.f.second_window])
        self.assertEqual(scope['max_remaining_dispatches'],2048)
        self.assertEqual(scope['case_sha256'],[hashlib.sha256(canonical(b)).hexdigest() for b in self.f.plan['batches']])
        self.assertEqual(self.f.audit.call_count,4)

    def test_latest_native_and_physical_release_fail_closed(self):
        for change in (lambda r:r.update(dispatch_ordinal=9), lambda r:r.update(request_id='f'*32),
                lambda r:r.update(EOS_detected=True), lambda r:r.update(KV_cache_empty=False),
                lambda r:r.update(BASE_restored_and_verified=False),lambda r:r.update(mac_port_8769_connect_ex=0),
                lambda r:r['mac_processes_present'].update({'1':True}),
                lambda r:r['mac_locks_present'].update({'lease':True}),
                lambda r:r.update(closeout_errors=['error'])):
            with self.subTest(change=change):
                old=copy.deepcopy(self.f.latest);change(self.f.latest)
                with self.assertRaises(ValueError):self.f.validate()
                self.f.latest=old

    def test_old_release_and_old_native_count_are_still_required(self):
        self.f.native['used_Game_requests']-=1
        with self.assertRaises(ValueError):self.f.validate()
        self.f.native['used_Game_requests']+=1
        self.f.release['case2_status']='COMPLETE'
        with self.assertRaises(ValueError):self.f.validate()

    def test_cannot_rerun_any_future_case_or_reuse_prior_windows(self):
        for i in range(5,13):
            path=self.f.root/f'batch_{i:02d}';path.mkdir()
            with self.assertRaises(ValueError):self.f.validate()
            path.rmdir()
        stop=self.f.grant_root/self.f.second_admission/(self.f.second_window+'.STOP.json')
        stop.unlink()
        with self.assertRaises((ValueError,FileNotFoundError)):self.f.validate()

    def test_unknown_cases_must_not_be_promoted(self):
        data={'cells':copy.deepcopy(self.f.cells),'batches':self.f.receipts}
        for i in (1,3):
            bad=copy.deepcopy(data);bad['cells'][i]['evidence_status']='audited_complete'
            self.f.write('RESULTS.json',bad)
            with self.assertRaises(ValueError):self.f.validate()

    def test_attempt_identity_and_complete_cumulative_count(self):
        path=self.f.root/'batch_04/REQUEST_GUARD_EVIDENCE.jsonl'
        original=[json.loads(line) for line in path.read_text().splitlines()]
        for mutate in (lambda rows:rows[1]['generation_identity'].update(decision=0),
                       lambda rows:rows[1]['generation_identity'].update(attempt=2),
                       lambda rows:rows[1].update(window_sha256=self.f.first_window),
                       lambda rows:rows[1].update(scientific_plan_sha256='f'*64),
                       lambda rows:rows.pop()):
            rows=copy.deepcopy(original);mutate(rows)
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaises(ValueError):self.f.validate()

    def test_new_driver_preserves_both_old_stops_and_stops_on_new_incomplete(self):
        for name in ('SUMMARY.json','CANDIDATE_COHORT_LEDGER.json'):
            self.f.write(name,{'synthetic':name})
        root=self.f.root
        saved={name:(root/name).read_bytes() for name in ('RESULTS.json','SUMMARY.json',
            'CANDIDATE_COHORT_LEDGER.json','DISPLAY_METADATA.json','INDEPENDENT_STATE_COLLECTION_STOP.json',
            'CONTINUATION_STOP.json')}
        a_ref={'path':'.local/a.json','sha256':'1'*64};b_ref={'path':'.local/b.json','sha256':'2'*64}
        bare={'schema':v2.SCOPE_SCHEMA,'prior_window_sha256':[self.f.first_window,self.f.second_window],
              'prefix_evidence_sha256':{'results':hashlib.sha256(saved['RESULTS.json']).hexdigest()}}
        scope={**bare,'release_proof_ref':a_ref,'latest_release_proof_ref':b_ref}
        self.f.write('SCOPE.json',scope)
        review={'status':'PASS_SYNTHETIC','original_plan_sha256':self.f.plan_hash,
            'continuation_scope_sha256':hashlib.sha256((root/'SCOPE.json').read_bytes()).hexdigest(),
            'continuation_source_sha256':hashlib.sha256(v2.SOURCE_PATH.read_bytes()).hexdigest()}
        self.f.write('REVIEW.json',review)
        plan=self.f.plan; release=self.f.release; latest=self.f.latest
        class FakeConsumer:
            pins={'expected_window_sha256':'c'*64}
            contract=type('Contract',(),{'BASE':'synthetic_base'})()
            def _validated_scientific_amendment(self):return {'main_plan_ref':'plan','reviews':{'Main_source_ref':'review'}}
            def _main_reference(self,ref):
                if ref=='plan':return root/'PLAN.json',plan
                if ref=='scope':return root/'SCOPE.json',scope
                if ref=='review':return root/'REVIEW.json',review
                raise ValueError('unexpected ref')
            def _reference(self,ref):return (None,release if ref==a_ref else latest)
            def require_plan(self,*a):pass
            def validate_request(self,*a):pass
            def private_child_context(self,*a,**kw):return '{}'
            def remaining_global_seconds(self):return 60
        process=Mock();process.wait.return_value=1;process.poll.return_value=1
        cell={'model_id':'synthetic_role_0','seed':105,'evidence_status':'incomplete'}
        receipt={'batch':'batch_05','exit_code':1,'failure':{'type':'provider_or_harness_incomplete','automatic_retry':False}}
        with patch.object(v2.runner,'_consumer_type',return_value=FakeConsumer), \
             patch.object(v2,'validate_prefix',return_value={'scope':bare}), \
             patch.object(v2.economic,'source_matches',return_value=True), \
             patch.object(v2.comparison,'claim_security500_grant'), \
             patch.object(v2.comparison,'stop_security500_collection') as stop, \
             patch.object(v2.comparison,'audit_batch',return_value=([cell],receipt)), \
             patch.object(v2.runner,'augment_audited_cells',side_effect=lambda batch,part,rows:rows), \
             patch.object(v2.runner,'write_progress'),patch.object(v2.compact,'_ledger'), \
             patch.object(v2,'summarize',return_value={}), \
             patch.object(v2.economic,'publish_closed_batch',return_value={'entries':[]}), \
             patch('src.native_generation_transcript.run_decoder_subprocess',return_value={}), \
             patch.object(v2.subprocess,'Popen',return_value=process):
            with self.assertRaises(ValueError):v2.execute('scope',consumer=FakeConsumer())
        self.assertTrue((root/'batch_05').is_dir());self.assertFalse((root/'batch_06').exists())
        stop.assert_called_once()
        for name in ('RESULTS.json','SUMMARY.json','CANDIDATE_COHORT_LEDGER.json'):
            self.assertEqual((root/('BEFORE_CONTINUATION_V2_'+name)).read_bytes(),saved[name])
        self.assertEqual((root/'DISPLAY_METADATA_BEFORE_CONTINUATION_V2.json').read_bytes(),saved['DISPLAY_METADATA.json'])
        for name in ('INDEPENDENT_STATE_COLLECTION_STOP.json','CONTINUATION_STOP.json'):
            self.assertEqual((root/name).read_bytes(),saved[name])
        self.assertEqual(json.loads((root/'RESULTS.json').read_bytes())['cells'][:4],self.f.cells)


if __name__=='__main__': unittest.main()
