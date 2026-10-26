"""Exact approval, unchanged study, failed-parent and amended-source authority."""
from copy import deepcopy
import os
from pathlib import Path
from .io_utils import read,sha,digest
from .composite_source_archive import verify_transport
from .composite_bounded_sampling_profile import (MODE,PROFILE,MANIFEST,APPROVAL,PROPOSAL,APPROVAL_SHA,
 PROPOSAL_SHA,FAILED_SOURCE,FAILED_RUN,FAILED_CLOSEOUT_SHA,OWNER_THREAD,NOTEBOOK,expected_profile,notebook_code)
from .composite_bounded_sampling_rule import RULE_SHA,validate_rule
from .composite_kaggle_v2_profile import HUMAN,HUMAN_SHA,identity

_KEY=object()


def approval(root,cfg):
    inputs=Path(root)/'inputs'
    if sha(inputs/APPROVAL)!=APPROVAL_SHA or sha(inputs/PROPOSAL)!=PROPOSAL_SHA:
        raise PermissionError('Exact direct human specific sampler approval and reviewed final proposal required')
    approved,proposal=read(inputs/APPROVAL),read(inputs/PROPOSAL)
    expected={'approved_proposal_sha256':PROPOSAL_SHA,'approved_rule_canonical_sha256':RULE_SHA,
      'original_failed_run_id':FAILED_RUN,'original_failed_closeout_sha256':FAILED_CLOSEOUT_SHA,
      'source_user_message_id':'01a178f2-bc41-7963-a5ab-d8ac14241ba6',
      'scientific_stage_admission_still_required':True,'scientific_allocation_admitted_by_receipt_alone':False}
    if any(digest(approved.get(k))!=digest(v) for k,v in expected.items()):
        raise PermissionError('Verified direct human sampler choice binding changed')
    validate_rule(proposal['proposed_generic_rule'],cfg)
    if (proposal['failure_lineage']['source_commit']!=FAILED_SOURCE
        or proposal['failure_lineage']['old_run_remains_failed_unadmitted'] is not True):
        raise PermissionError('Preserve the exact failed/unadmitted old experiment')
    return approved


class SamplingContext:
    def __init__(self,root,profile,cfg,parent,approved,key=None):
        if key is not _KEY:raise PermissionError('Validate the complete new approved source/failed-data lineage')
        self._key=key;self.root=Path(root).resolve();self.parent=parent
        self._profile=deepcopy(profile);self._cfg=deepcopy(cfg);self._approval=deepcopy(approved)
    def profile(self):return deepcopy(self._profile)
    def cfg(self):return deepcopy(self._cfg)
    def approval(self):return deepcopy(self._approval)
    def require_runtime(self):
        from .composite_runtime import platform,KAGGLE_RUNTIME_DIRECTORIES
        if (platform()!='kaggle' or not all(Path(p).is_dir() for p in KAGGLE_RUNTIME_DIRECTORIES)
            or os.environ.get('CUDA_VISIBLE_DEVICES')!=''
            or not self.root.is_relative_to(Path('/kaggle/temp/research4_sampling_amendment_inputs').resolve())
            or self.root.name!='source' or Path(os.environ.get('RESEARCH4_SAMPLING_SOURCE_ROOT','')).resolve()!=self.root
            or os.environ.get('RESEARCH4_SAMPLING_PROFILE_SHA256')!=sha(self.root/'inputs'/PROFILE)):
            raise PermissionError('Actual amended Kaggle CPU runtime/source required for scientific sampler generation')
        return self


def require_context(value):
    from .composite_bounded_sampling_parent import require_failed_parent
    if not isinstance(value,SamplingContext) or value._key is not _KEY:raise PermissionError('Genuine approved sampler source context required')
    require_failed_parent(value.parent);return value


def validate_source(root,commit,parent):
    from .composite_bounded_sampling_parent import FILES,require_failed_parent
    from .composite_bounded_behavior_profile import validate_preparation
    root=Path(root).resolve();verify_transport(root,commit,MODE,MANIFEST);inputs=root/'inputs'
    manifest=read(inputs/'MANIFEST.json');required=FILES|{PROFILE,APPROVAL,PROPOSAL,HUMAN,'WORKSPACE.json'}
    if set(manifest)!=required or any(sha(inputs/n)!=h for n,h in manifest.items()):
        raise PermissionError('Complete exact new source/approval/failed-reference inputs required')
    cfg=validate_preparation(read(root/'config/composite_bounded_behavior_v4.json'))
    approved=approval(root,cfg);require_failed_parent(parent)
    if digest(cfg)!=digest(parent.cfg()) or sha(inputs/HUMAN)!=HUMAN_SHA:
        raise PermissionError('Preserve original full study and selected Kaggle authority')
    profile=read(inputs/PROFILE)
    if digest(profile)!=digest(expected_profile(commit,cfg,parent,approved)) or read(inputs/'WORKSPACE.json')!=identity(NOTEBOOK):
        raise PermissionError('Exact approved-only sampler mode/profile/account required')
    return SamplingContext(root,profile,cfg,parent,approved,_KEY)
