"""Re-audit exact scratch evidence as a full-fit dependency, never a warm start."""
import base64
import json
import os
from pathlib import Path
from .io_utils import read,sha,digest

PARENT_SOURCE='149278382451faa1aceed82f22a4604a9bf99f79'
PARENT_ARCHIVE_SHA='b9c1349cbc43747d3d5b715238a0d46ce67920aab52289c6f478a6d23bd2ee44'
PARENT_PACKAGE_SHA='71518ea00a67dbf11033d4e5049e3d9f862c23855477e33ec5bb31b304a8cc46'
PARENT_CLOSEOUT_SHA='8738f1b090fab3d7548b18dc0cac33d3f134308fdee5ad278abea1e752600a00'
PARENT_AUDIT_FILE='PARENT_QUALIFICATION_AUDIT.json'
PARENT_FILES={'PARENT_QUALIFICATION_SOURCE.r4b64','PARENT_QUALIFICATION_CLOSEOUT.json',PARENT_AUDIT_FILE,
 'PARENT_QUALIFICATION_BINDING.json','PARENT_QUALIFICATION_CONTROLLER.json','PARENT_QUALIFICATION_TERMINAL_AX.txt',
 'PARENT_QUALIFICATION_RELEASE_AX.txt','PARENT_QUALIFICATION_ACTUAL.ipynb','PARENT_QUALIFICATION_PREPARED.ipynb'}
_CACHE={}


def parent_raw_path(root,override=None):
    if override is not None:return Path(override).resolve()
    path=Path(os.environ['RESEARCH4_TASK_QUALIFICATION_ARCHIVE']).resolve()
    if not path.is_relative_to(Path('/kaggle/input').resolve()):raise PermissionError('Parent raw archive must be an immutable Kaggle input')
    expected='research4_campaign/runs/'+PARENT_SOURCE+'/task_skill_training_v1/qualification_01/evidence.zip'
    if not path.as_posix().endswith(expected):raise PermissionError('Another parent archive path was selected')
    return path


def read_parent(root,raw_override=None):
    root=Path(root);inputs=root/'inputs'
    if sha(inputs/'PARENT_QUALIFICATION_CLOSEOUT.json')!=PARENT_CLOSEOUT_SHA:
        raise PermissionError('Exact audited released qualification closeout required')
    path=parent_raw_path(root,raw_override)
    stamp=(str(path),path.stat().st_size,path.stat().st_mtime_ns,
           digest({name:sha(inputs/name) for name in sorted(PARENT_FILES)}))
    if stamp in _CACHE:return _CACHE[stamp]
    if sha(path)!=PARENT_ARCHIVE_SHA:raise PermissionError('Complete original qualification archive differs')
    encoded=base64.b64decode((inputs/'PARENT_QUALIFICATION_SOURCE.r4b64').read_bytes(),validate=True)
    from .composite_task_skill_model_audit import kaggle_closure,audit_qualification,require_qualification
    from .composite_task_skill_source_package import frozen_source_package
    with frozen_source_package(encoded,PARENT_PACKAGE_SHA,PARENT_SOURCE) as (source,mf):
        binding=read(inputs/'PARENT_QUALIFICATION_BINDING.json')
        if binding['scriptVersionId']!=361902084 or binding['source_commit']!=PARENT_SOURCE:
            raise PermissionError('Parent submission identity differs')
        code=lambda value:[''.join(c['source']) for c in value['cells'] if c['cell_type']=='code']
        closure=kaggle_closure(binding,PARENT_ARCHIVE_SHA,(inputs/'PARENT_QUALIFICATION_TERMINAL_AX.txt').read_text(encoding='utf-8'),
            (inputs/'PARENT_QUALIFICATION_RELEASE_AX.txt').read_text(encoding='utf-8'),read(inputs/'PARENT_QUALIFICATION_CONTROLLER.json'),
            code(read(inputs/'PARENT_QUALIFICATION_ACTUAL.ipynb')),code(read(inputs/'PARENT_QUALIFICATION_PREPARED.ipynb')))
        proof=audit_qualification(path,PARENT_ARCHIVE_SHA,source,PARENT_SOURCE,closure)
        receipt=require_qualification(proof)
        supplied=read(inputs/PARENT_AUDIT_FILE)
        for key in ['checkpoint_sha256','raw_archive_sha256','source_data_archive_sha256','provider_run_id']:
            if receipt[key]!=supplied[key]:raise PermissionError('Parent summary differs from the actual raw re-audit')
    _CACHE[stamp]=proof
    return proof
