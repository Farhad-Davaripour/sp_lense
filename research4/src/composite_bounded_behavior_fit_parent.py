"""Rebuild V4 data and qualification authority from immutable complete raw parents.

Serialized flags never issue permits. Source packages and independently saved
provider/controller/notebook observations are replayed by the current reader.
"""
import base64,os
from pathlib import Path
from .io_utils import read,sha,digest
from .composite_task_skill_source_package import frozen_source_package

DATA_FILES={'BOUNDED_DATA_PARENT_SOURCE.r4b64','BOUNDED_DATA_PARENT_EXPECTED.json',
 'BOUNDED_DATA_PARENT_CLOSEOUT.json','BOUNDED_DATA_PARENT_BINDING.json',
 'BOUNDED_DATA_PARENT_CONTROLLER.json','BOUNDED_DATA_PARENT_TERMINAL_AX.txt',
 'BOUNDED_DATA_PARENT_RELEASE_AX.txt','BOUNDED_DATA_PARENT_ACTUAL.ipynb','BOUNDED_DATA_PARENT_PREPARED.ipynb'}
QUAL_FILES={'BOUNDED_QUAL_PARENT_SOURCE.r4b64','BOUNDED_QUAL_PARENT_EXPECTED.json',
 'BOUNDED_QUAL_PARENT_CLOSEOUT.json','BOUNDED_QUAL_PARENT_BINDING.json',
 'BOUNDED_QUAL_PARENT_CONTROLLER.json','BOUNDED_QUAL_PARENT_TERMINAL_AX.txt',
 'BOUNDED_QUAL_PARENT_RELEASE_AX.txt','BOUNDED_QUAL_PARENT_ACTUAL.ipynb','BOUNDED_QUAL_PARENT_PREPARED.ipynb'}
_CACHE={}


def raw_path(environment,override=None):
    path=Path(override if override is not None else os.environ[environment]).resolve()
    if override is None and not path.is_relative_to(Path('/kaggle/input').resolve()):
        raise PermissionError('Actual parent raw data must be immutable Kaggle input')
    if not path.is_file():raise PermissionError('Complete raw parent file required')
    return path


def _read(root,kind,environment,manifest,reader,require,raw_override=None,alias_fit=None,alias_qualification=None,data_archive=None):
    inputs=Path(root)/'inputs';prefix='BOUNDED_'+kind+'_PARENT_'
    expected=read(inputs/(prefix+'EXPECTED.json'));source=expected['source_commit']
    raw=raw_path(environment,raw_override)
    def dependency(environment,override):
        path=raw_path(environment,override);stat=path.stat()
        return (str(path),stat.st_size,stat.st_mtime_ns)
    deps=[dependency('RESEARCH4_BOUNDED_ALIAS_FIT_ARCHIVE',alias_fit),
      dependency('RESEARCH4_BOUNDED_ALIAS_QUALIFICATION_ARCHIVE',alias_qualification)]
    if kind=='QUAL':deps.append(dependency('RESEARCH4_BOUNDED_DATA_ARCHIVE',data_archive))
    stamp=(kind,str(raw),sha(raw),digest({p.name:sha(p) for p in inputs.iterdir() if p.name.startswith(prefix)}),tuple(deps))
    if stamp in _CACHE:return _CACHE[stamp]
    if stamp[2]!=expected['raw_archive_sha256'] or sha(inputs/(prefix+'CLOSEOUT.json'))!=expected['closeout_sha256']:
        raise PermissionError('Original released raw parent/closeout differs')
    encoded=base64.b64decode((inputs/(prefix+'SOURCE.r4b64')).read_bytes(),validate=True)
    binding=read(inputs/(prefix+'BINDING.json'))
    if binding['scriptVersionId']!=expected['run_id'] or binding['source_commit']!=source:
        raise PermissionError('Exact consumed parent identity required')
    code=lambda p:[''.join(c['source']) for c in read(p)['cells'] if c['cell_type']=='code']
    if kind=='DATA':
        from .composite_bounded_behavior_data_admission import kaggle_closure
    else:
        from .composite_bounded_behavior_fit_audit import kaggle_closure
    with frozen_source_package(encoded,expected['package_sha256'],source,manifest_name=manifest) as (parent,_):
        if kind=='DATA':
            from .composite_bounded_behavior_data_profile import notebook_code
        else:
            from .composite_bounded_behavior_fit_source import notebook_code
        prepared=code(inputs/(prefix+'PREPARED.ipynb'))
        if prepared!=[notebook_code(parent,source,sha(parent.parent/manifest))]:
            raise PermissionError('Parent prepared code differs from its exact reviewed source/manifest')
        closure=kaggle_closure(binding,expected['raw_archive_sha256'],
          (inputs/(prefix+'TERMINAL_AX.txt')).read_text(encoding='utf-8'),
          (inputs/(prefix+'RELEASE_AX.txt')).read_text(encoding='utf-8'),read(inputs/(prefix+'CONTROLLER.json')),
          code(inputs/(prefix+'ACTUAL.ipynb')),prepared)
        if kind=='DATA':
            fit=raw_path('RESEARCH4_BOUNDED_ALIAS_FIT_ARCHIVE',alias_fit)
            qual=raw_path('RESEARCH4_BOUNDED_ALIAS_QUALIFICATION_ARCHIVE',alias_qualification)
            proof=reader(raw,expected['raw_archive_sha256'],parent,source,fit,qual,closure)
        else:
            proof=reader(raw,expected['raw_archive_sha256'],parent,source,closure,
              data_archive=raw_path('RESEARCH4_BOUNDED_DATA_ARCHIVE',data_archive),
              alias_fit_archive=raw_path('RESEARCH4_BOUNDED_ALIAS_FIT_ARCHIVE',alias_fit),
              alias_qualification_archive=raw_path('RESEARCH4_BOUNDED_ALIAS_QUALIFICATION_ARCHIVE',alias_qualification))
        receipt=require(proof)
        for key in ['raw_archive_sha256','source_commit']:
            if receipt[key]!=expected[key]:raise PermissionError('Raw audited parent descriptor differs')
        if kind=='QUAL' and (receipt['checkpoint_sha256']!=expected['checkpoint_sha256']
                or digest(receipt['initial_adapter_hashes'])!=expected['initial_adapter_hashes_sha256']):
            raise PermissionError('Qualification checkpoint/initialization differs')
    _CACHE[stamp]=proof
    return proof


def read_data_parent(root,raw_override=None,alias_fit=None,alias_qualification=None):
    from .composite_bounded_behavior_data_reader import audit_data_from_source,require_data
    return _read(root,'DATA','RESEARCH4_BOUNDED_DATA_ARCHIVE','BOUNDED_BEHAVIOR_DATA_PACKAGE_FILES.json',
      audit_data_from_source,require_data,raw_override,alias_fit,alias_qualification)


def read_qualification_parent(root,raw_override=None,*,data_archive=None,alias_fit=None,alias_qualification=None):
    from .composite_bounded_behavior_fit_audit import audit_fit,require_qualification
    return _read(root,'QUAL','RESEARCH4_BOUNDED_QUALIFICATION_ARCHIVE','BOUNDED_BEHAVIOR_FIT_PACKAGE_FILES.json',
      audit_fit,require_qualification,raw_override,alias_fit,alias_qualification,data_archive)
