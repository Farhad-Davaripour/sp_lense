"""Frozen inference for the existing base control; no behavior package or fitting."""
import io
import json
from pathlib import Path
import zipfile

from .io_utils import read
from .composite_controller import Composite, rollout
from .composite_frozen_base import (MODE, DETECTOR_ARCHIVE, DETECTOR_ARCHIVE_SHA256,
    DETECTOR_CHECKPOINT_SHA256, base_report)


class RecordingBaseComposite(Composite):
    def __init__(self, detector, actor, threshold, persist):
        super().__init__(detector, actor, threshold); self.persist = persist

    def observe(self, world):
        value = super().observe(world)
        self.persist('observations/' + world.scene.case_id + '_' + str(world.sequence).zfill(3) + '.json', value)
        return value

    def respond(self, world, condition):
        if condition != 'base':
            raise PermissionError('Frozen-base package cannot execute adapter interventions')
        return super().respond(world, condition)


def selected_detector(store, cfg):
    """Read the exact admitted checkpoint, including its complete evidence seal."""
    import hashlib
    import torch
    from .composite_exports import verify_evidence_archive
    raw = store.get(DETECTOR_ARCHIVE)
    if hashlib.sha256(raw).hexdigest() != DETECTOR_ARCHIVE_SHA256:
        raise ValueError('The complete audited detector archive changed')
    verify_evidence_archive(raw)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        receipt = json.loads(archive.read('RUN_RECEIPT.json'))
        checkpoint = archive.read('detector_best.pt')
        if receipt['status'] != 'complete' or hashlib.sha256(checkpoint).hexdigest() != DETECTOR_CHECKPOINT_SHA256:
            raise ValueError('The complete admitted detector checkpoint changed')
    value = torch.load(io.BytesIO(checkpoint), map_location='cpu', weights_only=True)
    if (value['config'] != cfg['modernbert'] or value['head_mode'] != 'role_pool'
        or value['validation_admitted'] is not True or value['threshold'] != 0.9 or value['updates'] != 600):
        raise ValueError('The exact detector selection/configuration differs')
    return value


def load_package(context, cache, guard, on_forward, persist_bytes, persist_json):
    from .composite_store import Store, download_model
    from .composite_detector import Detector
    from .composite_frozen_probe import load_authorized_actor, memory_record
    from .composite_training import parameter_snapshot, assert_frozen, auxiliary_digest
    from .composite_native_capture import FrozenBaseActor
    import torch
    root = Path(context['source_root']); commit = context['source_commit']; cache = Path(cache)
    cfg = read(root/'config/composite_controller_rank32_v2.json')
    saved = selected_detector(Store(), cfg)
    actor, actor_context, actor_proof = load_authorized_actor(root, commit, cfg['qwen'],
        cache/'qwen', guard, on_forward, persist_json, context, MODE)
    if actor_context['authorization_sha256'] != context['authorization_sha256']:
        raise ValueError('Native actor and base package authority differ')
    guard()
    pin = read(root/'inputs/MODERNBERT_PIN.json')
    path, files = download_model(pin, cache/'bert')
    persist_json('DETECTOR_FILE_VERIFICATION.json', files)
    guard()
    detector = Detector(path, cfg['modernbert'], device='cuda:0', head_mode='role_pool')
    detector.restore(saved); detector.eval().requires_grad_(False); detector.on_forward = on_forward
    detector_parameters = list(detector.named_parameters()); detector_buffers = list(detector.named_buffers())
    snapshot = parameter_snapshot(detector_parameters)
    auxiliary = auxiliary_digest(detector_parameters, detector_buffers)
    before = [{'name': name, 'sha256': fingerprint, 'version': version, 'shape': list(parameter.shape),
               'dtype': str(parameter.dtype), 'device': str(parameter.device), 'requires_grad': parameter.requires_grad}
              for name, parameter, version, fingerprint in snapshot]
    persist_json('DETECTOR_PARAMETERS_BEFORE.json', before)
    proxy = FrozenBaseActor(actor, persist_bytes, persist_json, lambda: memory_record(torch.cuda))
    model = RecordingBaseComposite(detector, proxy, 0.9, persist_json).eval().requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()) or actor.adapters:
        raise PermissionError('Frozen-base inference cannot expose trainable weights or behavior adapters')
    proof = {'package_kind': 'frozen-base-reference', 'source_commit': commit,
        'authorization_sha256': context['authorization_sha256'], 'plan': context['evaluation_plan'],
        'backbone_instances': {'ModernBERT': 1, 'Qwen': 1}, 'actor': actor_proof,
        'detector_files': files, 'detector_checkpoint_sha256': DETECTOR_CHECKPOINT_SHA256,
        'detector_capacity': detector.capacity_proof, 'detector_threshold': 0.9,
        'detector_auxiliary_sha256': auxiliary, 'joint_memory_after_load': memory_record(torch.cuda),
        'trainable_parameters': 0, 'optimizer_updates': 0, 'behavior_adapter': None,
        'adapter_OFF_bypass': None, 'adapter_ON_effect': None,
        'full_validation_admitted': False, 'heldout_content_read': False}
    persist_json('FROZEN_BASE_PACKAGE.json', proof)
    persist_json('QWEN_PARAMETERS_BEFORE.json', actor.frozen_proof())

    def verify():
        actor.verify_frozen(); assert_frozen(snapshot)
        if auxiliary_digest(detector_parameters, detector_buffers) != auxiliary:
            raise ValueError('Frozen detector buffers or packed state changed')
        persist_json('DETECTOR_PARAMETERS_AFTER.json', [
            {'name': name, 'sha256': fingerprint, 'version': version, 'shape': list(parameter.shape),
             'dtype': str(parameter.dtype), 'device': str(parameter.device), 'requires_grad': parameter.requires_grad}
            for name, parameter, version, fingerprint in parameter_snapshot(detector_parameters)])
        persist_json('QWEN_PARAMETERS_AFTER.json', actor.frozen_proof())
    return model, proof, verify


def evaluate_partition(model, context, guard, persist, progress, verify):
    from .composite_world import Scene
    plan = context['evaluation_plan']; lookup = {scene['case_id']: scene for scene in context['scenes']}
    if plan['conditions'] != ['base']:
        raise PermissionError('Only the original base control is registered by this endpoint')
    records = []
    persist('FROZEN_BASE_PLAN.json', plan)
    try:
        for key in plan['case_ids']:
            scene = Scene(**lookup[key]); guard()
            def response_progress(world, trace):
                persist('steps/' + key + '_' + str(len(trace)).zfill(3) + '.json', trace[-1])
                progress('response_complete', case_id=key, responses=len(trace), clock=world.clock,
                         outcome=trace[-1]['outcome'])
            value = rollout(model, scene, 'base', guard, response_progress)
            persist('trajectories/' + key + '.json', value); records.append(value)
            progress('trajectory_complete', case_id=key, trajectories=len(records), outcome=value['outcome'])
    finally:
        persist('FROZEN_BASE_REPORT.json', base_report(records, plan['case_ids']))
        verify()
    return base_report(records, plan['case_ids'])
