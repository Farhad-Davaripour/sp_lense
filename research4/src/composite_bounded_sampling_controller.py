"""One external CPU child deadline and durable closeout; never submit or retry."""
import json
from pathlib import Path
import time
import traceback
import zipfile

from .io_utils import read, sha
from .composite_bounded_sampling_profile import MODE, PHASE, NOTEBOOK, OWNER_THREAD, BOUNDS
from .composite_exports import make_evidence_archive, verify_evidence_archive
from .composite_runtime import run_logged


def rescue_partial(output, commit, child):
    """Archive existing owned bytes only after the child is confirmed gone."""
    output = Path(output)
    if child.get('child_exit_confirmed') is not True:
        return {'attempted': False, 'reason': 'child_exit_not_confirmed'}
    if not output.is_dir() or not (output / 'OUTPUT_OWNER.json').is_file():
        return {'attempted': False, 'reason': 'no_owned_raw_output'}
    expected = {'source_commit': commit, 'version': MODE, 'phase': PHASE,
                'owner_thread_id': OWNER_THREAD, 'path': str(output)}
    if read(output / 'OUTPUT_OWNER.json') != expected:
        raise PermissionError('Partial rescue cannot touch another raw output owner')
    archive = output.with_name(output.name + '_PARTIAL_EVIDENCE.zip')
    result = make_evidence_archive(output, archive, BOUNDS['maximum_raw_export_bytes'])
    return {'attempted': True, 'archive': str(archive), **result,
            'scientific_dataset_admitted': False, 'unique_raw_removed': False}


def run(root, commit, python, environment, *, elapsed_setup_seconds=0,
        _runner=None, _clock=None):
    runner = run_logged if _runner is None else _runner
    clock = time.monotonic if _clock is None else _clock
    from .composite_bounded_sampling_worker import Services, verify_cpu_environment
    profile = Services.preview(root, commit)
    output = Path(profile['destination']); output.parent.mkdir(parents=True, exist_ok=True)
    controller_path = output.parent / 'BOUNDED_SAMPLING_CONTROLLER_RECEIPT.json'
    log = output.parent / 'BOUNDED_SAMPLING_WORKER_LOG.txt'
    if (controller_path.exists() or log.exists() or output.exists()
            or output.with_name(output.name + '_EVIDENCE.zip').exists()
            or output.with_name(output.name + '_PARTIAL_EVIDENCE.zip').exists()):
        raise PermissionError('Preserve this existing CPU attempt; do not start another child')
    started = clock(); child = {'child_started': False, 'child_exit_confirmed': None, 'exit_code': None}
    receipt = {'source_commit': commit, 'notebook': NOTEBOOK, 'mode': MODE, 'phase': PHASE,
        'owner_thread_id': OWNER_THREAD, 'status': 'setting_up', 'model_forwards': 0,
        'optimizer_updates': 0, 'scientific_GPU_count': 0, 'model_weights_downloaded': False,
        'heldout_content_read': False, 'automatic_scientific_retries': 0,
        'provider_compute_release_verified': False, 'public_fixture_test_mode': _runner is not None}
    try:
        if (type(elapsed_setup_seconds) not in [int, float] or elapsed_setup_seconds < 0
                or BOUNDS['hard_timeout_seconds'] - elapsed_setup_seconds - 30 < BOUNDS['worker_seconds']):
            raise TimeoutError('A complete registered CPU worker/export envelope must remain')
        result = runner([str(python), '-m', 'src.composite_bounded_sampling_worker',
                         '--root', str(root), '--commit', commit],
                        root, environment, log,
                        BOUNDS['worker_seconds'], exit_observer=lambda value: child.update(value))
        receipt.update(status='external_hard_timeout' if result['timed_out'] else 'worker_exited',
                       exit_code=result['exit_code'], timed_out=result['timed_out'])
        if (result['timed_out'] is not False or type(result['exit_code']) is not int
                or result['exit_code'] != 0 or child.get('child_started') is not True
                or child.get('child_exit_confirmed') is not True or child.get('exit_code') != 0):
            raise RuntimeError('Preserve this exact failed/unknown child attempt; no automatic retry')
        archive = output.with_name(output.name + '_EVIDENCE.zip')
        verify_evidence_archive(archive)
        with zipfile.ZipFile(archive) as stream:
            worker = json.loads(stream.read('RUN_RECEIPT.json'))
        if (worker.get('source_commit') != commit or worker.get('stage') != MODE
                or worker.get('phase') != PHASE or worker.get('status') != 'complete'
                or worker.get('completed_cases') != 144 or worker.get('metadata_cache_removed') is not True
                or worker.get('new_teacher_or_case_generation') is not False
                or worker.get('original_failure_preserved') is not True
                or worker.get('worker_exiting') is not True
                or worker.get('public_fixture_test_mode') is not False):
            raise PermissionError('Complete actual CPU worker export required before notebook success')
        receipt.update(cpu_runtime=verify_cpu_environment(worker.get('cpu_runtime')),
                       raw_archive_sha256=sha(archive), raw_archive=str(archive), durable_raw_export=True,
                       original_failure_preserved=True)
    except Exception as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error),
                       traceback=traceback.format_exc())
        try: receipt['partial_raw_rescue'] = rescue_partial(output, commit, child)
        except Exception as rescue_error:
            receipt['partial_raw_rescue'] = {'attempted': False, 'error': str(rescue_error)}
    finally:
        receipt.update(child=child, seconds=clock() - started)
        with controller_path.open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, indent=2); stream.write('\n')
    return receipt
