"""Re-audit the exact stateful scratch output before a fresh full fit."""
import base64
import os
from pathlib import Path
from .io_utils import read,sha,digest
from .composite_task_skill_source_package import frozen_source_package

SOURCE='10eae62507d38f1507fb7443cf579ca5b6acc29f'
RUN=361949668
PACKAGE_SHA='6c800d6ffc9f534587e1a137e0e9b4765cb10b5955842990b0dfd815bcff5d89'
FILES={'STATEFUL_PARENT_SOURCE.r4b64','STATEFUL_PARENT_EXPECTED.json','STATEFUL_PARENT_BINDING.json',
       'STATEFUL_PARENT_CLOSEOUT.json',
       'STATEFUL_PARENT_CONTROLLER.json','STATEFUL_PARENT_TERMINAL_AX.txt','STATEFUL_PARENT_RELEASE_AX.txt',
       'STATEFUL_PARENT_ACTUAL.ipynb','STATEFUL_PARENT_PREPARED.ipynb'}
_CACHE={}


def raw_path(override=None):
    if override is not None:return Path(override).resolve()
    path=Path(os.environ['RESEARCH4_STATEFUL_QUALIFICATION_ARCHIVE']).resolve()
    suffix='research4_campaign/runs/'+SOURCE+'/task_skill_training_v1/qualification_01/evidence.zip'
    if not path.is_relative_to(Path('/kaggle/input').resolve()) or not path.as_posix().endswith(suffix):
        raise PermissionError('Original immutable stateful qualification output required')
    return path


def read_stateful_parent(root,override=None):
    inputs=Path(root)/'inputs'
    if any(not (inputs/n).is_file() for n in FILES):raise PermissionError('Complete original stateful parent inputs required')
    expected=read(inputs/'STATEFUL_PARENT_EXPECTED.json')
    if (set(expected)!={'source_commit','run_id','package_sha256','raw_archive_sha256','checkpoint_sha256','initial_adapter_hashes_sha256','closeout_sha256'}
            or expected['source_commit']!=SOURCE or type(expected['run_id']) is not int
            or expected['run_id']!=RUN or expected['package_sha256']!=PACKAGE_SHA):
        raise PermissionError('Exact one-use stateful qualification identity required')
    closed=read(inputs/'STATEFUL_PARENT_CLOSEOUT.json')
    if (sha(inputs/'STATEFUL_PARENT_CLOSEOUT.json')!=expected['closeout_sha256']
            or closed.get('scriptVersionId')!=RUN or closed.get('source_commit')!=SOURCE
            or closed.get('scientific_qualification_admitted') is not True or closed.get('owned_compute_released') is not True
            or closed.get('raw_archive_sha256')!=expected['raw_archive_sha256']):
        raise PermissionError('Exact complete independently audited/released stateful closeout required')
    path=raw_path(override);actual=sha(path)
    if actual!=expected['raw_archive_sha256']:raise PermissionError('Complete unchanged qualification raw archive required')
    stamp=(actual,digest({n:sha(inputs/n) for n in sorted(FILES)}))
    if stamp in _CACHE:return _CACHE[stamp]
    raw=base64.b64decode((inputs/'STATEFUL_PARENT_SOURCE.r4b64').read_bytes(),validate=True)
    from .composite_task_skill_model_audit import kaggle_closure,audit_task_model,require_stateful_qualification
    with frozen_source_package(raw,PACKAGE_SHA,SOURCE) as (source,mf):
        code=lambda v:[''.join(c['source']) for c in v['cells'] if c['cell_type']=='code']
        prepared=code(read(inputs/'STATEFUL_PARENT_PREPARED.ipynb'))
        expected_code=(source/'notebooks/composite_task_skill_qualification.py').read_text().replace(
          'SOURCE_COMMIT_PLACEHOLDER',SOURCE).replace('PACKAGE_MANIFEST_SHA256_PLACEHOLDER',sha(source.parent/'TASK_TRAINING_PACKAGE_FILES.json'))
        if prepared!=[expected_code]:raise PermissionError('Parent source/input bootstrap readback differs')
        binding=read(inputs/'STATEFUL_PARENT_BINDING.json')
        if binding.get('scriptVersionId')!=RUN or binding.get('source_commit')!=SOURCE:
            raise PermissionError('Stateful submission identity differs')
        closure=kaggle_closure(binding,actual,(inputs/'STATEFUL_PARENT_TERMINAL_AX.txt').read_text(encoding='utf-8'),
          (inputs/'STATEFUL_PARENT_RELEASE_AX.txt').read_text(encoding='utf-8'),read(inputs/'STATEFUL_PARENT_CONTROLLER.json'),
          code(read(inputs/'STATEFUL_PARENT_ACTUAL.ipynb')),prepared)
        proof=audit_task_model(path,actual,source,SOURCE,closure,required_phase='qualification')
        receipt=require_stateful_qualification(proof)
        if receipt['checkpoint_sha256']!=expected['checkpoint_sha256'] or digest(receipt['initial_adapter_hashes'])!=expected['initial_adapter_hashes_sha256']:
            raise PermissionError('Actual qualified checkpoint/initial tensor identity differs')
    _CACHE[stamp]=proof
    return proof
