"""The existing admitted all22/rank32 detector only; no fit or model forward."""
from pathlib import Path
from .io_utils import read,sha
from .composite_frozen_base import DETECTOR_CHECKPOINT_SHA256,PARENTS

FILES={'DETECTOR_SELECTED.pt','DETECTOR_COMPONENT_CLOSEOUT.json','DETECTOR_COMPONENT_AUDIT.json'}


def selected_detector(inputs,cfg):
    inputs=Path(inputs);manifest=read(inputs/'MANIFEST.json')
    if not FILES<=set(manifest) or any(sha(inputs/n)!=manifest[n] for n in FILES):
        raise PermissionError('Complete fixed-detector checkpoint and admitted component receipts required')
    if (sha(inputs/'DETECTOR_SELECTED.pt')!=DETECTOR_CHECKPOINT_SHA256
            or sha(inputs/'DETECTOR_COMPONENT_CLOSEOUT.json')!=PARENTS['recognition_closeout']
            or sha(inputs/'DETECTOR_COMPONENT_AUDIT.json')!=PARENTS['recognition_audit']):
        raise PermissionError('Original selected fixed-detector evidence changed')
    closed=read(inputs/'DETECTOR_COMPONENT_CLOSEOUT.json')
    if (closed.get('run_id')!=361470090 or closed.get('component_validation_admitted') is not True
            or closed.get('owned_compute_released') is not True or closed.get('checkpoint_sha256')!=DETECTOR_CHECKPOINT_SHA256):
        raise PermissionError('Actual admitted and released recognition component required')
    import torch
    value=torch.load(inputs/'DETECTOR_SELECTED.pt',map_location='cpu',weights_only=True,mmap=True)
    if (value.get('config')!=cfg or value.get('head_mode')!='role_pool' or value.get('threshold')!=0.9
            or value.get('validation_admitted') is not True or value.get('updates')!=600):
        raise PermissionError('The complete selected detector/threshold may not be changed')
    return value
