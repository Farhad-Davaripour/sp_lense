"""Continue only the eight untouched cases after two stopped windows.

The original plan, native transport and per-request stop policy are unchanged.
This driver requires a fresh admission, a reviewed scope, an audited prefix,
and proof that the previous endpoint was physically released.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from . import economic_game_runner as runner
from . import economic_game_comparison as economic
from . import infrastructure_preservation_comparison as comparison
from . import company_world_evaluation as evaluation
from . import compact_game_comparison as compact
from .company_preservation_comparison import summarize

REPOSITORY = runner.REPOSITORY
SOURCE_PATH = Path(__file__).resolve()
RELEASE_SCHEMA = 'research4.economic-endpoint-owned-release-after-incomplete-case.v1'
RELEASE_STATUS = 'PASS_NATIVE_DRAIN_AND_OWNED_ENDPOINT_RELEASED'
LATEST_RELEASE_SCHEMA = 'research4.continuation-case04-native-independent-release.v1'
LATEST_RELEASE_STATUS = 'PASS_NATIVE_REQUEST_AND_RELEASE'
SCOPE_SCHEMA = 'research4.economic-game-untouched-eight-continuation.v2'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def _read_root_ref(reference):
    from .economic_game_consumer import _pinned_bytes, _json
    return _json(_pinned_bytes(reference, runner.models.PROVIDER352_ROOT)[1])


def validate_prefix(plan, directory, release_proof, latest_release, *, read_json=comparison.read_json,
                    audit_batch=comparison.audit_batch, read_root_ref=_read_root_ref,
                    grant_root=REPOSITORY / '.local/economic_grants'):
    """Re-audit prior cases and prove the remaining unique request ceiling."""
    directory = Path(directory)
    if (read_json(directory / 'PLAN.json') != plan or len(plan.get('batches', [])) != 12 or len(plan.get('models', [])) != 4
            or any(x.get('max_decisions') != 256 for x in plan['batches'])
            or any((directory / f'batch_{i:02d}').exists() for i in range(5, 13))):
        raise ValueError('Continuation requires exactly eight untouched cases')
    proof = release_proof
    if (type(proof) is not dict or proof.get('schema') != RELEASE_SCHEMA
            or proof.get('status') != RELEASE_STATUS or proof.get('HTTP_policy_phase') != 'CLOSED'
            or proof.get('known_model_workers') != [] or proof.get('Mac_port8769_connect_result') != 61
            or any(v is not False for v in proof.get('leases_exist', {}).values())
            or not proof.get('leases_exist') or proof.get('case2_status') != 'INCOMPLETE_UNKNOWN'
            or any(proof.get('processes', {}).get(name, {}).get('same_owned_identity_present') is not False
                   or proof.get('processes', {}).get(name, {}).get('status') != 'absent'
                   for name in ('qualifier', 'service'))
            or not isinstance(proof.get('actual_native_request_proof'), dict)
            or proof.get('closeout_errors') != []
            or proof.get('case1_preserved') is not True):
        raise ValueError('Previous owned endpoint release is not proven')
    last = latest_release
    if (type(last) is not dict or last.get('schema') != LATEST_RELEASE_SCHEMA
            or last.get('status') != LATEST_RELEASE_STATUS or last.get('http_policy') != 'CLOSED'
            or last.get('matching_model_workers') != [] or last.get('mac_port_8769_connect_ex') != 61
            or not last.get('mac_processes_present') or any(x is not False for x in last['mac_processes_present'].values())
            or not last.get('mac_locks_present') or any(x is not False for x in last['mac_locks_present'].values())
            or last.get('closeout_errors') != [] or last.get('closeout_reports_both_leases_absent') is not True
            or last.get('finish_reason') != 'length' or last.get('completion_tokens') != 8192
            or last.get('max_tokens') != 8192 or last.get('EOS_detected') is not False
            or last.get('KV_cache_empty') is not True or last.get('BASE_restored_and_verified') is not True
            or last.get('adapter_wrappers_after') != 0 or last.get('native_errors') != []
            or any(last.get(key) is not True for key in ('worker_finished', 'backend_barrier_done', 'post_checks_passed'))):
        raise ValueError('Latest native length stop and physical release are not proven')
    result = read_json(directory / 'RESULTS.json')
    cells, receipts = result.get('cells', []), result.get('batches', [])
    failure = {'type': 'provider_or_harness_incomplete', 'automatic_retry': False}
    if (len(cells) != 4 or len(receipts) != 4
            or [x.get('evidence_status') for x in cells] != ['audited_complete', 'incomplete', 'audited_complete', 'incomplete']
            or any(receipts[i].get('exit_code') != (0 if i % 2 == 0 else 1)
                   or receipts[i].get('failure') != (None if i % 2 == 0 else failure) for i in range(4))):
        raise ValueError('Saved completed/unknown prefix differs')
    evidence = {'plan': _sha(directory / 'PLAN.json'), 'results': _sha(directory / 'RESULTS.json')}
    stop_path = directory / 'INDEPENDENT_STATE_COLLECTION_STOP.json'
    stop = read_json(stop_path)
    if stop.get('game_plan_sha256') != evidence['plan'] or not isinstance(stop.get('stopped_grant_sha256'), str):
        raise ValueError('Original stopped grant is not bound to the plan')
    evidence['prior_collection_stop'] = _sha(stop_path)
    second_stop_path = directory / 'CONTINUATION_STOP.json'
    if read_json(second_stop_path).get('status') != 'STOPPED_NO_NEXT_CASE':
        raise ValueError('First continuation is not stopped')
    evidence['prior_continuation_stop'] = _sha(second_stop_path)
    meta_path = directory / 'DISPLAY_METADATA.json'
    meta = read_json(meta_path)
    import psutil
    try:
        old_process = psutil.Process(meta['controller_pid'])
        if abs(old_process.create_time() - meta['controller_created_epoch']) < 1:
            raise ValueError('Original collector is still active')
    except psutil.NoSuchProcess:
        pass
    except (KeyError, TypeError, psutil.AccessDenied) as error:
        raise ValueError('Original collector absence cannot be verified') from error
    evidence['prior_display_metadata'] = _sha(meta_path)
    request_ids = set()
    final_requests, windows, admissions, counts = {}, {}, {}, {}
    for i in range(4):
        batch, part = plan['batches'][i], directory / f'batch_{i + 1:02d}'
        sample = batch['samples'][0]
        closeout = read_json(part / 'CLOSEOUT.json')
        if (closeout.get('scientific_plan_sha256') != evidence['plan']
                or closeout.get('transport_plan_sha256') != evidence['plan']
                or closeout.get('source_unchanged') is not True
                or closeout.get('failure') != receipts[i].get('failure')):
            raise ValueError('Saved closeout does not bind original plan and outcome')
        if (cells[i].get('model_id') != sample['model_id'] or cells[i].get('seed') != sample['seed']
                or cells[i].get('prefix_replay_passed') is not True):
            raise ValueError('Saved prefix identity or replay differs')
        replayed, receipt = audit_batch(batch, part, receipts[i]['exit_code'], continuity_transfer=True)
        replayed = runner.augment_audited_cells(batch, part, replayed)
        if replayed != [cells[i]] or receipt != receipts[i]:
            raise ValueError('Saved prefix failed independent replay')
        for name in ('MANIFEST.json', 'EVENTS.jsonl', 'CLOSEOUT.json', 'REQUEST_GUARD_EVIDENCE.jsonl',
                     'PUBLISHED_GAME_LOGGING_RECEIPT.json', 'NATIVE_TRANSCRIPT_RECEIPT.json'):
            evidence[f'batch_{i + 1:02d}/{name}'] = _sha(part / name)
        decisions = set()
        with (part / 'REQUEST_GUARD_EVIDENCE.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                event = json.loads(line)
                if event.get('event') == 'HTTP_ATTEMPT':
                    identity = event.get('generation_identity', {})
                    request = event.get('request_id')
                    decision = identity.get('decision')
                    window, admission = event.get('window_sha256'), event.get('economic_admission_sha256')
                    segment = i // 2
                    if (event.get('scientific_plan_sha256') != evidence['plan']
                            or type(window) is not str or re.fullmatch(r'[0-9a-f]{64}', window) is None
                            or type(admission) is not str or re.fullmatch(r'[0-9a-f]{64}', admission) is None
                            or (segment in windows and windows[segment] != window)
                            or (segment in admissions and admissions[segment] != admission)
                            or identity.get('case_id') != sample['case_id'] or identity.get('attempt') != 1
                            or type(decision) is not int or not 0 <= decision < 256 or decision in decisions
                            or not isinstance(request, str) or len(request) != 32 or request in request_ids):
                        raise ValueError('Prior native dispatch identity differs')
                    decisions.add(decision)
                    request_ids.add(request)
                    windows[segment], admissions[segment] = window, admission
                    final_requests[i] = request
        if not decisions or decisions != set(range(len(decisions))):
            raise ValueError('Prior case dispatches are missing or non-contiguous')
        counts[i] = len(decisions)
    if windows[0] != stop['stopped_grant_sha256'] or windows[0] == windows[1]:
        raise ValueError('Two distinct prior stopped windows are required')
    for segment in (0, 1):
        grant_stop = Path(grant_root) / admissions[segment] / (windows[segment] + '.STOP.json')
        stopped = read_json(grant_stop)
        if stopped != {'window_sha256': windows[segment], 'status': 'STOPPED'}:
            raise ValueError('A prior window STOP receipt is missing or different')
        evidence[f'prior_grant_stop_{segment + 1}'] = _sha(grant_stop)
    prior = len(request_ids)
    if prior > 1024 or prior + 8 * 256 > 3072:
        raise ValueError('Original cumulative request budget cannot be proven')
    native = read_root_ref(proof['actual_native_request_proof'])
    ack = native.get('completion_ack', {})
    if (native.get('schema') != 'research4.economic-incomplete-request-native-proof.v1'
            or native.get('request_id') != final_requests[1] or native.get('used_Game_requests') != counts[0] + counts[1]
            or native.get('case_outcome') != 'INCOMPLETE_UNKNOWN' or native.get('finish_reason') != 'length'
            or native.get('output_tokens') != 8192 or native.get('output_limit') != 8192
            or native.get('EOS_detected') is not False
            or native.get('KV_cache_empty') is not True or native.get('BASE_restored_and_verified') is not True
            or native.get('pending_http_requests') != 0 or native.get('active_native_ticket') is not None
            or any(ack.get(k) is not True for k in ('worker_started', 'worker_finished', 'backend_barrier_done', 'post_checks_passed'))
            or ack.get('native_errors') != []):
        raise ValueError('Native same-request release and cumulative count are not proven')
    evidence['native_release_proof'] = proof['actual_native_request_proof']['sha256']
    if last.get('request_id') != final_requests[3] or last.get('dispatch_ordinal') != prior:
        raise ValueError('Latest native cumulative dispatch count or request differs')
    scope = {'schema': SCOPE_SCHEMA, 'plan_sha256': evidence['plan'],
        'case_sha256': [_canonical_sha(batch) for batch in plan['batches']],
        'allowed_indices': list(range(4, 12)), 'prior_dispatches': prior,
        'max_remaining_dispatches': 8 * 256, 'max_cumulative_dispatches': 3072,
        'prior_window_sha256': [windows[0], windows[1]],
        'prefix_evidence_sha256': evidence, 'release_proof_sha256': _canonical_sha(proof),
        'latest_release_proof_sha256': _canonical_sha(last),
        'continuation_source_sha256': _sha(SOURCE_PATH)}
    scope['fingerprint'] = _canonical_sha(scope)
    return {'next_index': 4, 'prior_dispatches': prior, 'scope': scope}


def execute(scope_ref, *, consumer):
    """Execute each of cases 5–12 once, stopping on incompletion."""
    if not isinstance(consumer, runner._consumer_type()):
        raise ValueError('Fresh admitted economic consumer required')
    admission = consumer._validated_scientific_amendment()
    path, frozen = consumer._main_reference(admission['main_plan_ref'])
    scope_path, expected = consumer._main_reference(scope_ref)
    review_path, review = consumer._main_reference(admission['reviews']['Main_source_ref'])
    if (not str(review.get('status', '')).startswith('PASS')
            or review.get('original_plan_sha256') != _sha(path)
            or review.get('continuation_scope_sha256') != _sha(scope_path)
            or review.get('continuation_source_sha256') != _sha(SOURCE_PATH)
            or expected.get('schema') != SCOPE_SCHEMA):
        raise ValueError('Independent review must bind this exact continuation')
    _, proof = consumer._reference(expected['release_proof_ref'])
    _, latest = consumer._reference(expected['latest_release_proof_ref'])
    actual = validate_prefix(frozen, path.parent, proof, latest)
    actual['scope']['release_proof_ref'] = expected['release_proof_ref']
    actual['scope']['latest_release_proof_ref'] = expected['latest_release_proof_ref']
    if actual['scope'] != expected:
        raise ValueError('Reviewed continuation scope or preserved prefix changed')
    if consumer.pins['expected_window_sha256'] in expected['prior_window_sha256']:
        raise ValueError('Prior stopped window cannot be reused')
    destination = path.parent
    result = comparison.read_json(destination / 'RESULTS.json')
    cells, receipts = copy.deepcopy(result['cells']), copy.deepcopy(result['batches'])
    attempted = [batch['samples'][0]['case_id'] for batch in frozen['batches'][:4]]
    with (destination / 'CONTINUATION_V2_RUNNER.lock').open('x', encoding='utf-8') as lock:
        lock.write(json.dumps({'pid': os.getpid(), 'scope_sha256': _sha(scope_path)})); lock.flush(); os.fsync(lock.fileno())
        try:
            consumer.require_plan(_sha(path)); consumer.validate_request(consumer.contract.BASE)
            comparison.claim_security500_grant(destination, _sha(path), 'all', consumer)
            for name in ('RESULTS.json', 'SUMMARY.json', 'CANDIDATE_COHORT_LEDGER.json'):
                original = destination / name
                if original.is_file():
                    raw = original.read_bytes()
                    if name == 'RESULTS.json' and hashlib.sha256(raw).hexdigest() != expected['prefix_evidence_sha256']['results']:
                        raise ValueError('Prefix results changed before snapshot')
                    with (destination / ('BEFORE_CONTINUATION_V2_' + name)).open('xb') as stream:
                        stream.write(raw)
            # Presentation metadata is not scientific evidence; preserve its original bytes first.
            old_meta = destination / 'DISPLAY_METADATA.json'
            saved_meta = destination / 'DISPLAY_METADATA_BEFORE_CONTINUATION_V2.json'
            with saved_meta.open('xb') as stream: stream.write(old_meta.read_bytes())
            import psutil
            meta = comparison.read_json(old_meta)
            meta.update(controller_pid=os.getpid(), controller_created_epoch=psutil.Process().create_time())
            evaluation.write_json(old_meta, meta)
            for index in range(4, 12):
                batch = frozen['batches'][index]
                consumer._validated_scientific_amendment()
                if not economic.source_matches(batch): raise ValueError('Original source changed')
                runner.write_progress(destination, frozen, cells, active_role=batch['models'][0])
                attempted.append(batch['samples'][0]['case_id']); compact._ledger(destination, frozen, cells, attempted)
                part = destination / f'batch_{index + 1:02d}'; part.mkdir()
                context = consumer.private_child_context(_sha(path), source_scope={'plan_path': str(path), 'batch_index': index})
                from .economic_game_consumer import ECONOMIC_CONTEXT_ENV
                env = {**os.environ, ECONOMIC_CONTEXT_ENV: context}
                with (part / 'operator.log').open('xb') as output:
                    process = subprocess.Popen([sys.executable, '-m', runner.MODULE, '--execute-child', '--output', str(part)],
                        cwd=REPOSITORY, env=env, stdout=output, stderr=subprocess.STDOUT)
                    try: exit_code = process.wait(timeout=min(14490, max(1, consumer.remaining_global_seconds() + 5)))
                    finally:
                        if process.poll() is None: comparison.stop_owned_worker(process)
                new_cells, receipt = comparison.audit_batch(batch, part, exit_code, continuity_transfer=True)
                new_cells = runner.augment_audited_cells(batch, part, new_cells)
                cells.extend(new_cells); receipts.append(receipt)
                evaluation.write_json(destination / 'RESULTS.json', {'cells': cells, 'batches': receipts})
                compact._ledger(destination, frozen, cells, attempted)
                evaluation.write_json(destination / 'SUMMARY.json', summarize(cells, models=frozen['models'], seeds=frozen['seeds'],
                    protocol=frozen['schema'], horizon_hours=48, contrasts=frozen['comparison_contrasts']))
                published = economic.publish_closed_batch(part)
                evaluation.write_json(part / 'PUBLISHED_GAME_LOGGING_RECEIPT.json', published)
                from . import native_generation_transcript as transcript
                try:
                    native = transcript.run_decoder_subprocess(part, published['entries'], transcript.CPU_REFERENCE_PYTHON)
                    evaluation.write_json(part / 'NATIVE_TRANSCRIPT_RECEIPT.json', native)
                except Exception as error:
                    evaluation.write_json(part / 'NATIVE_TRANSCRIPT_ERROR.json', {'status': 'OFFLINE_NATIVE_TRANSCRIPT_UNAVAILABLE',
                        'exception_type': type(error).__name__, 'native_science_results_reclassified': False})
                if (receipt['exit_code'] != 0 or receipt['failure'] is not None
                        or any(x['evidence_status'] != 'audited_complete' for x in new_cells)):
                    raise ValueError('Incomplete child preserved; no next case')
                consumer.validate_request(runner.models.MODEL_SPECS[batch['models'][0]]['service_model_id'])
            evaluation.write_json(destination / 'CONTINUATION_V2_CLOSEOUT.json', {'cases': len(cells),
                'previous_unknown_preserved': True, 'root_window_release_confirmed': False})
            runner.write_progress(destination, frozen, cells, running=False)
        except BaseException:
            comparison.stop_security500_collection(destination, consumer)
            evaluation.write_json(destination / 'CONTINUATION_V2_STOP.json', {
                'status': 'STOPPED_NO_NEXT_CASE', 'previous_incomplete_remains_unknown': True,
                'root_window_release_confirmed': False})
            compact._ledger(destination, frozen, cells, attempted)
            runner.write_progress(destination, frozen, cells, running=False)
            raise
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', type=Path, required=True)
    args = parser.parse_args()
    scope = args.scope.resolve()
    if not scope.is_relative_to((REPOSITORY / '.local').resolve()) or not scope.is_file():
        raise ValueError('Continuation scope must be a Main-local evidence file')
    reference = {'path': scope.relative_to(REPOSITORY).as_posix(), 'sha256': _sha(scope)}
    from .economic_game_consumer import ECONOMIC_CONTEXT_ENV, consumer_from_context
    consumer = consumer_from_context(os.environ.get(ECONOMIC_CONTEXT_ENV, 'null'))
    return execute(reference, consumer=consumer)


if __name__ == '__main__':
    raise SystemExit(main())
