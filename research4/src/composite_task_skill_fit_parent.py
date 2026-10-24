"""Read-only full-fit dependency for future frozen inference; never submit/fit."""
import base64
from pathlib import Path
from .io_utils import read,sha,digest
from .composite_task_skill_source_package import frozen_source_package
from .composite_task_skill_package import FIT_SOURCE,FIT_RUN,full_fit_identity,verify_audit_reference
from .composite_task_skill_parent import PARENT_ARCHIVE_SHA

FIT_PACKAGE_SHA='b18e06971434cda2b5312e670f5942c6b44688826103cb421f0f8e607fe83e5b'
FIT_MANIFEST_SHA='3069515817b1ac3ec599de2becb1e3d0cdf0758b957327d5e1c789f5c91d8ecc'
FIT_FILES={'FIT_PARENT_SOURCE.r4b64','FIT_PARENT_EXPECTED.json','FIT_PARENT_BINDING.json','FIT_PARENT_AUDIT.json',
  'FIT_PARENT_CONTROLLER.json','FIT_PARENT_TERMINAL_AX.txt','FIT_PARENT_RELEASE_AX.txt',
  'FIT_PARENT_ACTUAL.ipynb','FIT_PARENT_PREPARED.ipynb'}
_CACHE={}


def read_full_fit_parent(inputs,fit_archive,qualification_archive):
    inputs=Path(inputs);fit_archive=Path(fit_archive);qualification_archive=Path(qualification_archive)
    if any(not (inputs/name).is_file() for name in FIT_FILES):
        raise PermissionError('Complete full-fit source/raw/provider dependency inputs required')
    manifest=read(inputs/'MANIFEST.json')
    if not FIT_FILES<=set(manifest) or any(sha(inputs/name)!=manifest[name] for name in FIT_FILES):
        raise PermissionError('Frozen full-fit dependency input hashes differ')
    expected=read(inputs/'FIT_PARENT_EXPECTED.json')
    if (set(expected)!={'source_commit','run_id','package_sha256','raw_archive_sha256',
                       'checkpoint_sha256','fit_evidence_identity_sha256','qualification_archive_sha256','raw_audit_canonical_sha256'}
            or expected['source_commit']!=FIT_SOURCE or type(expected['run_id']) is not int
            or expected['run_id']!=FIT_RUN or expected['package_sha256']!=FIT_PACKAGE_SHA
            or expected['qualification_archive_sha256']!=PARENT_ARCHIVE_SHA):
        raise PermissionError('Only the registered exact full-fit lineage is eligible')
    fit_sha,qual_sha=sha(fit_archive),sha(qualification_archive)
    if fit_sha!=expected['raw_archive_sha256'] or qual_sha!=PARENT_ARCHIVE_SHA:
        raise PermissionError('Complete original fit and scratch archives required')
    stamp=(fit_sha,qual_sha,digest({n:sha(inputs/n) for n in sorted(FIT_FILES)}))
    if stamp in _CACHE:return _CACHE[stamp]
    encoded=base64.b64decode((inputs/'FIT_PARENT_SOURCE.r4b64').read_bytes(),validate=True)
    from .composite_task_skill_model_audit import kaggle_closure,audit_task_model,require_full_fit
    with frozen_source_package(encoded,FIT_PACKAGE_SHA,FIT_SOURCE) as (source,mf):
        if sha(source.parent/'TASK_TRAINING_PACKAGE_FILES.json')!=FIT_MANIFEST_SHA:
            raise PermissionError('Original full-fit package manifest changed')
        code=lambda value:[''.join(c['source']) for c in value['cells'] if c['cell_type']=='code']
        prepared=code(read(inputs/'FIT_PARENT_PREPARED.ipynb'))
        expected_code=(source/'notebooks/composite_task_skill_qualification.py').read_text().replace(
          'SOURCE_COMMIT_PLACEHOLDER',FIT_SOURCE).replace('PACKAGE_MANIFEST_SHA256_PLACEHOLDER',FIT_MANIFEST_SHA)
        if prepared!=[expected_code]:raise PermissionError('Prepared parent code differs from its exact source/input package')
        binding=read(inputs/'FIT_PARENT_BINDING.json')
        if binding.get('source_commit')!=FIT_SOURCE or type(binding.get('scriptVersionId')) is not int or binding['scriptVersionId']!=FIT_RUN:
            raise PermissionError('Full-fit submission identity differs')
        closure=kaggle_closure(binding,expected['raw_archive_sha256'],
          (inputs/'FIT_PARENT_TERMINAL_AX.txt').read_text(encoding='utf-8'),
          (inputs/'FIT_PARENT_RELEASE_AX.txt').read_text(encoding='utf-8'),
          read(inputs/'FIT_PARENT_CONTROLLER.json'),code(read(inputs/'FIT_PARENT_ACTUAL.ipynb')),prepared)
        proof=audit_task_model(fit_archive,expected['raw_archive_sha256'],source,FIT_SOURCE,closure,
                              required_phase='full_fit',qualification_archive=qualification_archive)
        report=require_full_fit(proof)
        reference=read(inputs/'FIT_PARENT_AUDIT.json')
        if digest(reference)!=expected['raw_audit_canonical_sha256']:
            raise PermissionError('Original full-fit audit reference changed')
        verify_audit_reference(reference,report)
        if (report['checkpoint_sha256']!=expected['checkpoint_sha256']
                or full_fit_identity(proof)!=expected['fit_evidence_identity_sha256']):
            raise PermissionError('Fitted checkpoint/raw identity differs from its original complete freeze')
    _CACHE[stamp]=proof
    return proof
