"""Re-audit an actual complete development-risk data export, never a summary."""
import base64
from pathlib import Path
import re
from .io_utils import read,sha,digest
from .composite_task_skill_source_package import frozen_source_package

FILES={'RISK_PARENT_SOURCE.r4b64','RISK_PARENT_EXPECTED.json','RISK_PARENT_BINDING.json',
       'RISK_PARENT_CONTROLLER.json','RISK_PARENT_TERMINAL_AX.txt','RISK_PARENT_RELEASE_AX.txt',
       'RISK_PARENT_ACTUAL.ipynb','RISK_PARENT_PREPARED.ipynb','RISK_PARENT_AUDIT.json'}
_CACHE={}


def read_risk_parent(inputs,archive):
    inputs=Path(inputs);archive=Path(archive)
    manifest=read(inputs/'MANIFEST.json')
    if not FILES<=set(manifest) or any(sha(inputs/n)!=manifest[n] for n in FILES):
        raise PermissionError('Complete immutable risk-data source/raw/provider inputs required')
    expected=read(inputs/'RISK_PARENT_EXPECTED.json')
    keys={'source_commit','run_id','package_sha256','package_manifest_sha256','raw_archive_sha256','release_sha256','audit_canonical_sha256'}
    if (set(expected)!=keys or type(expected.get('run_id')) is not int or expected['run_id']<=0
            or not re.fullmatch('[0-9a-f]{40}',expected.get('source_commit',''))
            or any(not re.fullmatch('[0-9a-f]{64}',expected.get(k,'')) for k in keys-{'source_commit','run_id'})):
        raise PermissionError('Exact actual one-use development-risk export descriptor required')
    if sha(archive)!=expected['raw_archive_sha256']:raise PermissionError('Complete original risk reference archive required')
    stamp=(expected['raw_archive_sha256'],digest({n:sha(inputs/n) for n in sorted(FILES)}))
    if stamp in _CACHE:return _CACHE[stamp]
    encoded=base64.b64decode((inputs/'RISK_PARENT_SOURCE.r4b64').read_bytes(),validate=True)
    from .composite_task_skill_model_audit import kaggle_closure
    from .composite_frozen_risk_data_reader import audit_data,require_data
    with frozen_source_package(encoded,expected['package_sha256'],expected['source_commit'],
                               manifest_name='RISK_DATA_PACKAGE_FILES.json') as (source,_):
        if sha(source.parent/'RISK_DATA_PACKAGE_FILES.json')!=expected['package_manifest_sha256']:
            raise PermissionError('Risk-data parent package manifest changed')
        code=lambda value:[''.join(c['source']) for c in value['cells'] if c['cell_type']=='code']
        prepared=code(read(inputs/'RISK_PARENT_PREPARED.ipynb'))
        expected_code=(source/'notebooks/composite_task_skill_data.py').read_text(encoding='utf-8-sig').replace(
          'SOURCE_COMMIT_PLACEHOLDER',expected['source_commit']).replace('PACKAGE_MANIFEST_SHA256_PLACEHOLDER',expected['package_manifest_sha256'])
        if prepared!=[expected_code]:raise PermissionError('Risk data parent code differs from its exact source/input freeze')
        binding=read(inputs/'RISK_PARENT_BINDING.json')
        if binding.get('source_commit')!=expected['source_commit'] or binding.get('scriptVersionId')!=expected['run_id']:
            raise PermissionError('Risk reference submission identity differs')
        closure=kaggle_closure(binding,expected['raw_archive_sha256'],
          (inputs/'RISK_PARENT_TERMINAL_AX.txt').read_text(encoding='utf-8'),
          (inputs/'RISK_PARENT_RELEASE_AX.txt').read_text(encoding='utf-8'),read(inputs/'RISK_PARENT_CONTROLLER.json'),
          code(read(inputs/'RISK_PARENT_ACTUAL.ipynb')),prepared)
        proof=audit_data(archive,expected['raw_archive_sha256'],source,expected['source_commit'],closure)
        report=require_data(proof)
        if (digest(report)!=expected['audit_canonical_sha256'] or digest(report)!=digest(read(inputs/'RISK_PARENT_AUDIT.json'))
                or report['release_sha256']!=expected['release_sha256']):
            raise PermissionError('Actual complete risk reference raw audit/release differs')
    _CACHE[stamp]=proof;return proof
