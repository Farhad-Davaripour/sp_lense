"""Reconstruct the actual new V4 full-fit/data/qualification chain; never a warm start."""
import base64,os
from copy import deepcopy
from pathlib import Path
from .io_utils import read,sha,digest
from .composite_task_skill_source_package import frozen_source_package

FILES={'BOUNDED_FIT_PARENT_SOURCE.r4b64','BOUNDED_FIT_PARENT_EXPECTED.json',
 'BOUNDED_FIT_PARENT_CLOSEOUT.json','BOUNDED_FIT_PARENT_BINDING.json',
 'BOUNDED_FIT_PARENT_CONTROLLER.json','BOUNDED_FIT_PARENT_TERMINAL_AX.txt',
 'BOUNDED_FIT_PARENT_RELEASE_AX.txt','BOUNDED_FIT_PARENT_ACTUAL.ipynb','BOUNDED_FIT_PARENT_PREPARED.ipynb'}
_KEY=object();_CACHE={}


class FrozenBoundedModel:
    def __init__(self,value,data,cfg,inputs,source_release,semantics,key=None):
        if key is not _KEY:raise PermissionError('Read the actual complete V4 full-fit lineage')
        self._key=key;self._value=deepcopy(value);self.data=data
        self.cfg=deepcopy(cfg);self.inputs=deepcopy(inputs);self.source_release=deepcopy(source_release)
        self.semantics=deepcopy(semantics)
    def receipt(self):return deepcopy(self._value)


def require_frozen_model(value):
    if not isinstance(value,FrozenBoundedModel) or value._key is not _KEY:
        raise PermissionError('A summary/old adapter cannot create frozen V4 model authority')
    from .composite_bounded_behavior_fit_audit import require_full_fit
    from .composite_bounded_behavior_data_reader import require_data
    report=value.receipt();require_data(value.data)
    if (report['full_fit_raw_audited'] is not True or report['owned_compute_released'] is not True
            or report['source_data_archive_sha256']!=value.data.receipt()['raw_archive_sha256']
            or report['checkpoint']['adapter_parameters']!=233455616 or report['checkpoint']['tensors']!=992):
        raise PermissionError('Complete new frozen checkpoint/data evidence required')
    return report


def read_parent(root,fit_archive,qualification_archive,data_archive,alias_fit_archive,alias_qualification_archive):
    inputs=Path(root)/'inputs';expected=read(inputs/'BOUNDED_FIT_PARENT_EXPECTED.json')
    paths=[Path(p).resolve() for p in [fit_archive,qualification_archive,data_archive,alias_fit_archive,alias_qualification_archive]]
    actual_hashes={name:sha(path) for name,path in zip(['fit','qualification','data','alias_fit','alias_qualification'],paths)}
    if actual_hashes!=expected.get('dependencies_sha256'):
        raise PermissionError('Every actual data/fit/qualification/alias archive must match its immutable content binding')
    stamp=(digest({n:sha(inputs/n) for n in sorted(FILES)}),digest(actual_hashes))
    if stamp in _CACHE:return _CACHE[stamp]
    if sha(paths[0])!=expected['raw_archive_sha256'] or sha(inputs/'BOUNDED_FIT_PARENT_CLOSEOUT.json')!=expected['closeout_sha256']:
        raise PermissionError('Actual V4 fit raw archive/closeout changed')
    packet=base64.b64decode((inputs/'BOUNDED_FIT_PARENT_SOURCE.r4b64').read_bytes(),validate=True)
    from .composite_bounded_behavior_fit_profile import MANIFEST
    from .composite_bounded_behavior_fit_source import validate_source,notebook_code
    from .composite_bounded_behavior_fit_audit import kaggle_closure,audit_fit,require_full_fit
    code=lambda p:[''.join(c['source']) for c in read(p)['cells'] if c['cell_type']=='code']
    with frozen_source_package(packet,expected['package_sha256'],expected['source_commit'],manifest_name=MANIFEST) as (source,_):
        binding=read(inputs/'BOUNDED_FIT_PARENT_BINDING.json')
        if binding['scriptVersionId']!=expected['run_id'] or binding['phase']!='full_fit':
            raise PermissionError('Exact completed fresh64-update fit identity required')
        prepared=code(inputs/'BOUNDED_FIT_PARENT_PREPARED.ipynb')
        if prepared!=[notebook_code(source,expected['source_commit'],sha(source.parent/MANIFEST))]:
            raise PermissionError('Parent fit saved source differs from reviewed bootstrap')
        closure=kaggle_closure(binding,expected['raw_archive_sha256'],
          (inputs/'BOUNDED_FIT_PARENT_TERMINAL_AX.txt').read_text(encoding='utf-8'),
          (inputs/'BOUNDED_FIT_PARENT_RELEASE_AX.txt').read_text(encoding='utf-8'),
          read(inputs/'BOUNDED_FIT_PARENT_CONTROLLER.json'),code(inputs/'BOUNDED_FIT_PARENT_ACTUAL.ipynb'),prepared)
        stage,cfg,data=validate_source(source,expected['source_commit'],paths[1],data_archive=paths[2],
          alias_fit_archive=paths[3],alias_qualification_archive=paths[4])
        proof=audit_fit(paths[0],expected['raw_archive_sha256'],source,expected['source_commit'],closure,
          required_phase='full_fit',qualification_archive=paths[1],data_archive=paths[2],
          alias_fit_archive=paths[3],alias_qualification_archive=paths[4])
        report=require_full_fit(proof)
        if report['checkpoint_sha256']!=expected['checkpoint_sha256']:
            raise PermissionError('Final registered checkpoint differs from actual raw-audited fit')
        small={name:read(source/'inputs'/name) for name in ['QWEN_PIN.json','MODERNBERT_PIN.json','QWEN_TENSOR_METADATA.json','PARENT_MODEL_INPUT_MANIFEST.json']}
        from .composite_bounded_behavior_evaluation_semantics import fingerprint
        result=FrozenBoundedModel(report,data,cfg,small,read(source/'RELEASE.json'),fingerprint(source),_KEY)
    _CACHE[stamp]=result;return result
