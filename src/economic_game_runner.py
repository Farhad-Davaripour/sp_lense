"""Fresh, guarded economic comparison collector and one-case child.

No default execution, model loading or grant exists. The caller supplies an
actual newly admitted economic consumer; Root retains service and release.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import company_models as models
from . import company_world_evaluation as evaluation
from . import compact_game_comparison as compact
from . import economic_game_comparison as economic
from . import economic_game_metrics as observations
from . import infrastructure_preservation_comparison as comparison
from .company_preservation_comparison import summarize

REPOSITORY = Path(__file__).resolve().parents[1]
PROGRESS = REPOSITORY / '.local/ongoing_curriculum_iteration2_48h_20261114/presentation/PUBLIC_PROGRESS.json'
MODULE = 'src.economic_game_runner'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _consumer_type():
    from .economic_game_consumer import EconomicGameConsumer
    return EconomicGameConsumer


def _records(path, case_id):
    rows = []
    if path.exists():
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                item = json.loads(line)
                if item.get('case_id') == case_id and item.get('type') in ('simulator_action', 'episode_result'):
                    rows.append(item)
    return rows


def write_display_metadata(destination, frozen):
    """Record real parent identity and role labels; no endpoint call."""
    import psutil
    route = frozen['economic_identity_candidate']['expected_ft_id']
    started = datetime.now(timezone.utc).isoformat()
    catalog = {'model_display_names': {role: ('SP-Lense ' + route if i >= 2 else 'Qwen3.8-27B base')
                                      for i, role in enumerate(economic.FUTURE_IDS)}}
    metadata = {'comparison_id': 'economic-' + route, 'controller_pid': os.getpid(),
        'controller_created_epoch': psutil.Process().create_time(), 'controller_started_utc': started,
        'started_utc': started, 'total_episodes': 12, 'worlds': 3, 'conditions': 4, 'horizon_hours': 48,
        'scientific_plan_sha256': _sha(destination / 'PLAN.json'), 'input_format': economic.economic.PROTOCOL,
        'display_timezone': 'America/Edmonton'}
    for name, value in (('DISPLAY_CATALOG.json', catalog), ('DISPLAY_METADATA.json', metadata)):
        with (destination / name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False); stream.write('\n')
    return metadata


def write_progress(destination, frozen, cells, *, active_role=None, running=True):
    """Update the local read-only UI. A UI issue never changes science status."""
    complete = sum(row.get('evidence_status') == 'audited_complete' for row in cells)
    try:
        value = {'schema': 'sp-lense.current-run-presentation-progress.v1',
            'updated_utc': datetime.now(timezone.utc).isoformat(), 'updated_display': 'Alberta Mountain time',
            'stage': (f'Fresh economic comparison: {complete}/12 audited; {active_role} running.' if running and active_role
                      else f'Fresh economic comparison: {complete}/12 audited.'),
            'model': frozen['economic_identity_candidate']['expected_ft_id'], 'game_running': running,
            'training_running': False, 'active_model_id': active_role, 'active_model_ids': list(economic.FUTURE_IDS),
            'planned_games': 12, 'audited_games': complete, 'horizon_hours': 48,
            'context_notice': 'Identical public economic summaries for all conditions; prior Game uses a different input protocol.',
            'outcome_notice': 'Only independently replayed completed cases have final outcomes.',
            'steps': [{'id': 'games', 'label': 'Fresh 48-hour matched comparison',
                       'completed': complete, 'total': 12, 'status': 'Running' if running else 'Collection stopped or finished'}]}
        temporary = PROGRESS.with_name(PROGRESS.name + '.economic-' + str(os.getpid()) + '.tmp')
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        os.replace(temporary, PROGRESS)
    except (OSError, ValueError, KeyError) as error:
        evaluation.write_json(Path(destination) / 'PRESENTATION_WARNING.json', {
            'status': 'READ_ONLY_PROGRESS_UNAVAILABLE', 'type': type(error).__name__,
            'scientific_results_changed': False})


def execute_child(destination, *, consumer):
    """One exact child; model dispatch requires a newly admitted consumer."""
    if not isinstance(consumer, _consumer_type()):
        raise ValueError('Fresh admitted economic consumer required')
    admission = consumer._validated_scientific_amendment()
    batch = copy.deepcopy(consumer._validate_game_source_scope())
    destination = Path(destination).resolve()
    expected = Path(consumer.source_scope['plan_path']).parent / f"batch_{consumer.source_scope['batch_index'] + 1:02d}"
    if destination != expected or not economic.source_matches(batch):
        raise ValueError('Economic child destination or frozen source differs')
    started, utc = time.monotonic(), datetime.now(timezone.utc)
    batch.update(started_utc=utc.isoformat(), deadline_utc=(utc + timedelta(seconds=batch['outer_seconds'])).isoformat())
    consumer.validate_child_plan(batch)
    consumer.bind_case_deadline(batch['started_utc'], batch['deadline_utc'])
    destination.mkdir(exist_ok=True)
    with (destination / 'MANIFEST.json').open('x', encoding='utf-8') as stream:
        json.dump(batch, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    records, failure, results = [], None, []

    def event(record):
        stamped = {**copy.deepcopy(record), 'bundle': destination.name}
        with (destination / 'EVENTS.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(stamped, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
        if stamped['type'] in {'simulator_action', 'episode_result'}:
            records.append(stamped)

    try:
        models.assert_not_reserved()
        model_id = batch['models'][0]
        spec = models.MODEL_SPECS[model_id]
        core = consumer._reference(consumer.core_binding['core_binding'])[1]
        if os.environ.get('RESEARCH4_QWEN_BASE_URL', core['base_url']).rstrip('/') != core['base_url']:
            raise ValueError('Client endpoint differs from fresh qualified Core')
        consumer.validate_request(spec['service_model_id'])
        os.environ['RESEARCH4_QWEN_BASE_URL'] = core['base_url']
        os.environ.setdefault('RESEARCH4_QWEN_API_KEY', 'local-unused')
        evaluation.write_json(destination / 'LOCAL_BINDING_READBACK.json', {
            spec['local_binding_group']: consumer.validated_health(spec['service_model_id'])})
        remaining = min(batch['outer_seconds'] - (time.monotonic() - started) - 30,
                        consumer.remaining_global_seconds())
        if remaining <= 0:
            raise TimeoutError('Original case or global budget exhausted')
        client = models._security500_guard_model(models.build_model(spec['model'], 1, 8192), consumer, model_id)
        client.config.timeout = max(1, int(min(900, remaining)))
        task = economic.build_task(batch, destination, remaining, event)
        from inspect_ai import Task
        bound = Task(dataset=task.dataset, solver=task.solver, scorer=task.scorer,
                     model=client, name='economic_' + model_id)
        if not economic.source_matches(batch):
            raise ValueError('Economic source changed before evaluation')
        logs = evaluation.evaluate_parallel([bound], destination / 'inspect', 8192)
        if not evaluation.logs_complete(logs):
            failure = {'type': 'provider_or_harness_incomplete', 'automatic_retry': False}
    except Exception as error:
        failure = {'type': type(error).__name__, 'automatic_retry': False}
    finally:
        try:
            results = evaluation.replay_all(records, batch['samples'])
            evaluation.write_json(destination / 'REPLAYED_RESULT.json', results[0] if results else None)
        except Exception as error:
            failure = {'type': type(error).__name__, 'stage': 'client_replay', 'automatic_retry': False}
        unchanged = economic.source_matches(batch)
        if not unchanged:
            failure = {'type': 'SourceDrift', 'automatic_retry': False}
        evaluation.write_json(destination / 'CLOSEOUT.json', {
            'failure': failure, 'result': results[0] if results else None, 'results': results,
            'elapsed_seconds': time.monotonic() - started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
            'source_unchanged': unchanged, 'transport_plan_sha256': consumer.pins['expected_game_plan_sha256'],
            'scientific_plan_sha256': admission['main_plan_ref']['sha256'],
            'resources': 'Existing Root-owned service only; no clock, quota or model reset.'})
    return 0 if failure is None and results and all(row['status'] == 'complete' for row in results) else 1


def augment_audited_cells(batch, directory, cells):
    """Append observed-prefix metrics only after the existing native replay audit."""
    if len(cells) != 1:
        raise ValueError('One audited case per batch required')
    sample, cell = batch['samples'][0], cells[0]
    if not cell.get('prefix_replay_passed'):
        return cells
    rows = _records(Path(directory) / 'EVENTS.jsonl', sample['case_id'])
    transitions = [row['transition'] for row in rows if row.get('type') == 'simulator_action']
    path = Path(directory) / 'REPLAYED_RESULT.json'
    replayed = json.loads(path.read_bytes()) if path.exists() else None
    # Failed children keep their observed prefix, never a promoted outcome.
    if cell.get('evidence_status') != 'audited_complete':
        replayed = None
    more = observations.observed_metrics(replayed, transitions, sample['model_id'], sample['seed'])
    if cell.get('evidence_status') == 'audited_complete' and more['score'] != cell.get('score'):
        raise ValueError('Observed metrics differ from independently replayed score')
    return [{**cell, 'copy_observation': more['copy_observation'],
             'company_observation': more['company_observation']}]


def execute_cohort(plan_path, *, consumer):
    """One fresh serial twelve-case attempt; no old cases are imported."""
    if not isinstance(consumer, _consumer_type()):
        raise ValueError('Fresh admitted economic consumer required')
    admission = consumer._validated_scientific_amendment()
    expected, frozen = consumer._main_reference(admission['main_plan_ref'])
    path = Path(plan_path).resolve()
    if path != expected:
        raise ValueError('Admitted economic plan path differs')
    economic.validate_cohort(frozen, excluded_seeds=frozenset(admission['excluded_seeds']))
    destination = path.parent
    if (any(destination.glob('batch_*')) or any((destination / name).exists()
            for name in ('RESULTS.json', 'CLOSEOUT.json', 'COLLECTION_STOP.json'))):
        raise ValueError('Existing economic attempt cannot be rerun')
    cells, receipts, attempted = [], [], []
    with (destination / 'RUNNER.lock').open('x', encoding='utf-8') as lock:
        lock.write(json.dumps({'pid': os.getpid(), 'scientific_plan_sha256': _sha(path)}))
        lock.flush(); os.fsync(lock.fileno())
        try:
            consumer.require_plan(_sha(path))
            consumer.validate_request(consumer.contract.BASE)
            comparison.claim_security500_grant(destination, _sha(path), 'all', consumer)
            write_display_metadata(destination, frozen)
            compact._ledger(destination, frozen, cells)
            for index, batch in enumerate(frozen['batches']):
                consumer._validated_scientific_amendment()
                if not economic.source_matches(batch):
                    raise ValueError('Economic source changed before next child')
                write_progress(destination, frozen, cells, active_role=batch['models'][0])
                attempted.append(batch['samples'][0]['case_id'])
                compact._ledger(destination, frozen, cells, attempted)
                directory = destination / f'batch_{index + 1:02d}'
                directory.mkdir()
                context = consumer.private_child_context(_sha(path), source_scope={'plan_path': str(path), 'batch_index': index})
                from .economic_game_consumer import ECONOMIC_CONTEXT_ENV
                env = {**os.environ, ECONOMIC_CONTEXT_ENV: context}
                with (directory / 'operator.log').open('xb') as output:
                    process = subprocess.Popen([sys.executable, '-m', MODULE, '--execute-child', '--output', str(directory)],
                        cwd=REPOSITORY, env=env, stdout=output, stderr=subprocess.STDOUT)
                    try:
                        exit_code = process.wait(timeout=min(14490, max(1, consumer.remaining_global_seconds() + 5)))
                    finally:
                        if process.poll() is None:
                            comparison.stop_owned_worker(process)
                new_cells, receipt = comparison.audit_batch(batch, directory, exit_code, continuity_transfer=True)
                new_cells = augment_audited_cells(batch, directory, new_cells)
                cells.extend(new_cells); receipts.append(receipt)
                evaluation.write_json(destination / 'RESULTS.json', {'cells': cells, 'batches': receipts})
                compact._ledger(destination, frozen, cells, attempted)
                evaluation.write_json(destination / 'SUMMARY.json', summarize(cells, models=frozen['models'], seeds=frozen['seeds'],
                    protocol=frozen['schema'], horizon_hours=48, contrasts=frozen['comparison_contrasts']))
                # A successfully archived incomplete case is still unknown,
                # not a logging failure or a reason to submit another case.
                try:
                    published = economic.publish_closed_batch(directory)
                    evaluation.write_json(directory / 'PUBLISHED_GAME_LOGGING_RECEIPT.json', published)
                except Exception as error:
                    evaluation.write_json(destination / 'LOGGING_EXPORT_FAILURE.json', {
                        'batch': directory.name, 'exception_type': type(error).__name__,
                        'native_science_results_reclassified': False, 'next_child_allowed': False})
                    raise
                # Decoding is CPU-only and never mutates the canonical stream or
                # raw guard records. A decoder problem is recorded separately;
                # the saved model result is not reclassified or discarded.
                from . import native_generation_transcript as transcript
                try:
                    native = transcript.run_decoder_subprocess(directory, published['entries'],
                                                               transcript.CPU_REFERENCE_PYTHON)
                    evaluation.write_json(directory / 'NATIVE_TRANSCRIPT_RECEIPT.json', native)
                except Exception as error:
                    evaluation.write_json(directory / 'NATIVE_TRANSCRIPT_ERROR.json', {
                        'status': 'OFFLINE_NATIVE_TRANSCRIPT_UNAVAILABLE', 'exception_type': type(error).__name__,
                        'raw_guard_evidence_preserved': (directory / 'REQUEST_GUARD_EVIDENCE.jsonl').is_file(),
                        'native_science_results_reclassified': False})
                if (receipt['exit_code'] != 0 or receipt['failure'] is not None
                        or any(row['evidence_status'] != 'audited_complete' for row in new_cells)):
                    raise ValueError('Incomplete child preserved; no next case')
                consumer.validate_request(models.MODEL_SPECS[batch['models'][0]]['service_model_id'])
            evaluation.write_json(destination / 'CLOSEOUT.json', {'consumer_finished': True, 'cases': len(cells),
                'scientific_plan_sha256': _sha(path), 'transport_plan_sha256': consumer.pins['expected_game_plan_sha256'],
                'root_window_release_confirmed': False, 'request_window_released': False})
            write_progress(destination, frozen, cells, running=False)
        except BaseException:
            comparison.stop_security500_collection(destination, consumer)
            compact._ledger(destination, frozen, cells, attempted)
            write_progress(destination, frozen, cells, running=False)
            raise
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-child', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute-plan', type=Path)
    args = parser.parse_args()
    if args.execute_child == (args.execute_plan is not None) or (args.execute_child and args.output is None):
        parser.error('Choose exactly one admitted child or parent entry point')
    from .economic_game_consumer import ECONOMIC_CONTEXT_ENV, consumer_from_context
    consumer = consumer_from_context(os.environ.get(ECONOMIC_CONTEXT_ENV, 'null'))
    return execute_child(args.output, consumer=consumer) if args.execute_child else execute_cohort(args.execute_plan, consumer=consumer)


if __name__ == '__main__':
    raise SystemExit(main())
