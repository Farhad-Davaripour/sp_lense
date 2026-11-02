"""One owned whole-scene frozen-base partition; no fitting or cloud submit API."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import time
import traceback

from .io_utils import read, sha
from .composite_frozen_base import MODE, BOUNDS, authorized_load_plan, conservative_forwards
from .composite_frozen_base_eval import load_package, evaluate_partition
from .composite_frozen_probe import verify_runtime, guard_frozen_devices
from .composite_frozen_worker import (DISK_POLICY as QUALIFICATION_DISK_POLICY, ForwardBudget,
    output_target, atomic_json, verify_transport, disk_admission as qualification_disk_admission,
    reserve_export_space, release_export_space, protected_alarm)
from .composite_probe_ownership import verify_owned_cache
from .composite_exports import exclusive_output, make_evidence_archive

DISK_POLICY = {**QUALIFICATION_DISK_POLICY, 'additional_frozen_detector_download_bytes': 600_790_696}
ARCHIVE = 'FROZEN_BASE_EVIDENCE.zip'


def disk_admission(cache, target, context):
    proof = qualification_disk_admission(cache, target, context['model_pin'])
    pin = read(Path(context['source_root'])/'inputs/MODERNBERT_PIN.json')
    total = sum(item['size'] for item in pin['files'])
    if total != DISK_POLICY['additional_frozen_detector_download_bytes']:
        raise ValueError('The full original detector download size changed')
    required = proof['required_cache_free_bytes'] + total
    free = shutil.disk_usage(cache).free
    if free < required:
        raise OSError('Insufficient disk for both full frozen checkpoints and protected exports')
    return {**proof, 'required_cache_free_bytes': required, 'observed_cache_free_bytes': free,
            'detector_download_bytes': total, 'disk_policy': DISK_POLICY}


class EvidenceWriter:
    def __init__(self, raw):
        self.raw = Path(raw)

    def bytes(self, name, data):
        from .composite_runtime import contained
        path = contained(self.raw, name)
        maximum = DISK_POLICY['maximum_raw_evidence_bytes']-DISK_POLICY['terminal_metadata_headroom_bytes']
        if sum(file.stat().st_size for file in self.raw.rglob('*') if file.is_file())+len(data) > maximum:
            raise OSError('Complete raw base evidence exceeds its export bound; no truncation')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream: stream.write(data)
        if path.read_bytes() != data: raise OSError('Raw base evidence readback differs')

    def json(self, name, value):
        self.bytes(name, (json.dumps(value, indent=2, allow_nan=False)+'\n').encode())


class Services:
    clock = staticmethod(time.monotonic)
    authorize = staticmethod(authorized_load_plan)
    ownership = staticmethod(verify_owned_cache)
    transport = staticmethod(verify_transport)
    runtime = staticmethod(verify_runtime)
    disk = staticmethod(disk_admission)
    reserve = staticmethod(reserve_export_space)
    release_reserve = staticmethod(release_export_space)
    alarm = staticmethod(protected_alarm)
    load = staticmethod(load_package)
    evaluate = staticmethod(evaluate_partition)
    output_base = '/kaggle/working/research4_campaign/runs'

    @staticmethod
    def resources():
        import torch
        guard_frozen_devices(torch.cuda)

    @staticmethod
    def release():
        gc.collect()
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def joint_counts(model):
    if model is None: return None
    return {key: model.actor.actor.frozen_probe_counts[key]+model.detector_forward_counts[key]
            for key in ['attempted', 'completed']}


def run(root, commit, destination, cache, owner_token, _services=None):
    services = Services() if _services is None else _services
    started = services.clock(); root = Path(root); model = None; verify = None
    context = services.authorize(root, commit, read(root/'config/composite_controller_rank32_v2.json')['qwen'])
    pin = context['pin']; plan = context['evaluation_plan']
    if (pin['destination'] != destination or pin['cache_root'] != cache or pin['disk_policy'] != DISK_POLICY
        or conservative_forwards(len(plan['case_ids'])) > BOUNDS['maximum_backbone_forwards']):
        raise PermissionError('Base worker paths/disk/whole-scene envelope differ')
    services.ownership(cache, commit, owner_token); services.transport(cache)
    target = output_target(destination, commit, owner_token, services.output_base)
    upload = exclusive_output(target, services.output_base)
    upload('OUTPUT_OWNER.json', json.dumps({'source_commit': commit, 'token': owner_token, 'path': str(target)}).encode())
    raw = target/'raw'; raw.mkdir(); writer = EvidenceWriter(raw)
    budget = ForwardBudget(started, services.clock, lambda value: atomic_json(raw/'FORWARD_COUNTS.json', value),
                           services.resources, BOUNDS)
    atomic_json(raw/'FORWARD_COUNTS.json', budget.snapshot())
    receipt = {'status': 'running', 'stage': MODE, 'source_commit': commit, 'partition': plan['partition'],
        'authorization_sha256': context['authorization_sha256'], 'model_load_started': False,
        'inference_started': False, 'backbone_instances': None, 'optimizer_updates': 0,
        'behavior_adapter_fitting': False, 'heldout_content_read': False, 'scientific_retries': 0,
        'full_validation_admitted': False, 'full_system_validation_complete': False}
    def guard():
        budget.check()
        if budget.attempted >= BOUNDS['maximum_backbone_forwards']:
            raise TimeoutError('Joint frozen-backbone forward ceiling')
    def progress(stage, **value):
        print(json.dumps({'stage': stage, 'partition': plan['partition'], 'seconds': services.clock()-started,
                          'forward_counts': budget.snapshot(), **value}), flush=True)
    try:
        writer.json('PRECALL_FREEZE.json', {'source_commit': commit, 'mode': MODE, 'pin': pin,
            'input_manifest_sha256': sha(root/'inputs/MANIFEST.json'), 'bounds': BOUNDS, 'disk_policy': DISK_POLICY})
        remaining = BOUNDS['worker_seconds']-BOUNDS['export_reserve_seconds']-(services.clock()-started)
        with services.alarm(remaining):
            writer.json('HARDWARE_PREFLIGHT.json', services.runtime(context))
            writer.json('DISK_ADMISSION.json', services.disk(cache, target, context))
            services.reserve(target, commit, owner_token)
            guard(); receipt['model_load_started'] = True
            model, proof, verify = services.load(context, Path(cache)/'models', guard, budget.event, writer.bytes, writer.json)
            receipt['backbone_instances'] = proof['backbone_instances']
            guard(); receipt['inference_started'] = True
            report = services.evaluate(model, context, guard, writer.json, progress, verify)
            verify = None  # The evaluator already performed its finally verification.
            if report['partition_complete'] is not True or report['full_validation_admitted'] is not False:
                raise ValueError('Completed base collection lacks its six original whole trajectories')
            receipt.update(status='complete', base_partition_complete=True, trajectories=len(plan['case_ids']))
    except Exception as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error))
        try: writer.json('FAILURE.json', {'traceback': traceback.format_exc()})
        except Exception as export_error: receipt['failure_detail_export_error'] = str(export_error)
    finally:
        owner_counts = {'attempted': budget.attempted, 'completed': budget.completed}
        independent = joint_counts(model)
        if independent is not None and independent != owner_counts:
            receipt.update(status='failed', counter_mismatch=True)
        receipt.update(forward_counts=owner_counts, independent_joint_counts=independent,
            counts_exact_at_controlled_exit=True, worker_exiting=True, seconds=services.clock()-started,
            platform_compute_release='requires_live_control_plane_verification')
        # A failed evaluator may have failed during verification; do not retry it
        # or overwrite its preserved raw proof files in cleanup.
        model = None; verify = None
        try: services.release(); receipt['owned_model_references_released'] = True
        except Exception as error: receipt['model_release_error'] = str(error)
        try: receipt.update(services.release_reserve(target, commit, owner_token))
        except Exception as error: receipt['export_reserve_release_error'] = str(error)
        atomic_json(raw/'FORWARD_COUNTS.json', {**owner_counts, 'sealed_for_exit': True,
                                               'source': 'observed_root_forward_hooks'})
        atomic_json(raw/'RUN_RECEIPT.json', receipt)
    archive = target/'evidence.zip'
    proof = make_evidence_archive(raw, archive, DISK_POLICY['maximum_raw_evidence_bytes'])
    upload(ARCHIVE, archive.read_bytes()); upload('RUN_RECEIPT.json', (raw/'RUN_RECEIPT.json').read_bytes())
    export = {**receipt, **proof, 'destination': destination, 'cache_cleanup': 'parent_only_after_child_exit'}
    upload('EXPORT_RECEIPT.json', json.dumps(export, indent=2).encode())
    print(json.dumps(export), flush=True)
    return export


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ['root', 'commit', 'destination', 'cache', 'owner-token']: parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    if run(args.root, args.commit, args.destination, args.cache, args.owner_token)['status'] != 'complete':
        raise SystemExit(1)
