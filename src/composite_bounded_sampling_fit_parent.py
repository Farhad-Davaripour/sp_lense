"""Genuine approved-data and scratch-qualification capsule reconstruction."""
import base64,hashlib,json,re,zipfile
import os
from pathlib import Path
from .io_utils import read, sha, digest
from .composite_task_skill_source_package import frozen_source_package

def _files(kind):
    return {'SAMPLING_'+kind+'_PARENT_'+name for name in ['SOURCE.r4b64','EXPECTED.json',
        'CLOSEOUT.json','BINDING.json','CONTROLLER.json','TERMINAL_AX.txt','RELEASE_AX.txt',
        'ACTUAL.ipynb','PREPARED.ipynb']}

DATA_FILES=_files('DATA');QUAL_FILES=_files('QUAL');_CACHE={}

def raw_path(environment,override=None):
    path=Path(override if override is not None else os.environ[environment]).resolve()
    if override is None and not path.is_relative_to(Path('/kaggle/input').resolve()):
        raise PermissionError('Actual approved-sampling parent must remain immutable Kaggle input')
    if not path.is_file():raise PermissionError('Complete actual raw parent required')
    return path

def _read(root,kind,override,alias_fit,alias_qualification,data_archive=None):
    inputs=Path(root)/'inputs';prefix='SAMPLING_'+kind+'_PARENT_'
    expected=read(inputs/(prefix+'EXPECTED.json'))
    raw=raw_path('RESEARCH4_SAMPLING_'+kind+'_ARCHIVE',override)
    aliases=[raw_path('RESEARCH4_BOUNDED_ALIAS_FIT_ARCHIVE',alias_fit),
             raw_path('RESEARCH4_BOUNDED_ALIAS_QUALIFICATION_ARCHIVE',alias_qualification)]
    dependencies=aliases+([raw_path('RESEARCH4_SAMPLING_DATA_ARCHIVE',data_archive)] if kind=='QUAL' else [])
    stamp=(kind,sha(raw),digest({n:sha(inputs/n) for n in sorted(_files(kind))}),
           tuple(sha(path) for path in dependencies))
    if stamp[1]!=expected['raw_archive_sha256'] or sha(inputs/(prefix+'CLOSEOUT.json'))!=expected['closeout_sha256']:
        raise PermissionError('Actual approved-sampling parent raw/closeout changed')
    if stamp in _CACHE:return _CACHE[stamp]
    if kind=='DATA':
        from .composite_bounded_sampling_profile import MANIFEST,notebook_code
        from .composite_bounded_sampling_admission import kaggle_closure
        from .composite_bounded_sampling_reader import audit_data_from_source,require_approved_sampling_data
    else:
        from .composite_bounded_sampling_fit_profile import MANIFEST
        from .composite_bounded_sampling_fit_source import notebook_code
        from .composite_bounded_sampling_fit_audit import kaggle_closure,audit_fit,require_qualification
    encoded=base64.b64decode((inputs/(prefix+'SOURCE.r4b64')).read_bytes(),validate=True)
    binding=read(inputs/(prefix+'BINDING.json'));source=expected['source_commit']
    if binding['source_commit']!=source or binding['scriptVersionId']!=expected['run_id']:
        raise PermissionError('Exact consumed approved-sampling parent identity required')
    code=lambda p:[''.join(c['source']) for c in read(p)['cells'] if c['cell_type']=='code']
    with frozen_source_package(encoded,expected['package_sha256'],source,manifest_name=MANIFEST) as (capsule,_):
        prepared=code(inputs/(prefix+'PREPARED.ipynb'))
        if prepared!=[notebook_code(capsule,source,sha(capsule.parent/MANIFEST))]:
            raise PermissionError('Actual parent prepared code differs from its own source capsule')
        closure=kaggle_closure(binding,expected['raw_archive_sha256'],
            (inputs/(prefix+'TERMINAL_AX.txt')).read_text(encoding='utf-8'),
            (inputs/(prefix+'RELEASE_AX.txt')).read_text(encoding='utf-8'),
            read(inputs/(prefix+'CONTROLLER.json')),code(inputs/(prefix+'ACTUAL.ipynb')),prepared)
        if kind=='DATA':
            proof=audit_data_from_source(raw,expected['raw_archive_sha256'],capsule,source,*aliases,closure)
            receipt=require_approved_sampling_data(proof)
        else:
            proof=audit_fit(raw,expected['raw_archive_sha256'],capsule,source,closure,
                required_phase='qualification',data_archive=dependencies[-1],
                alias_fit_archive=aliases[0],alias_qualification_archive=aliases[1])
            receipt=require_qualification(proof)
            if (receipt['checkpoint_sha256']!=expected['checkpoint_sha256']
                    or digest(receipt['initial_adapter_hashes'])!=expected['initial_adapter_hashes_sha256']):
                raise PermissionError('Actual scratch initial/checkpoint hashes differ')
        if receipt['source_commit']!=source or receipt['raw_archive_sha256']!=expected['raw_archive_sha256']:
            raise PermissionError('Recomputed approved parent receipt differs')
    _CACHE[stamp]=proof;return proof

def read_data_parent(root,raw_override=None,alias_fit=None,alias_qualification=None):
    return _read(root,'DATA',raw_override,alias_fit,alias_qualification)

def read_qualification_parent(root,raw_override=None,*,data_archive=None,alias_fit=None,alias_qualification=None):
    return _read(root,'QUAL',raw_override,alias_fit,alias_qualification,data_archive)


def historical_initialization_reference(data,archive=None):
    """Read only old initial tensor hashes: never weights, optimizer or a new PASS."""
    from .composite_bounded_sampling_reader import require_approved_sampling_data
    require_approved_sampling_data(data)
    failed=data.failed_reference().receipt()
    expected=failed['dependency_content_hashes']['qualification']
    path=raw_path('RESEARCH4_BOUNDED_ALIAS_QUALIFICATION_ARCHIVE',archive)
    if sha(path)!=expected:raise PermissionError('Original independently audited seed-map archive changed')
    with zipfile.ZipFile(path) as stream:
        names=stream.namelist()
        if len(names)!=len(set(names)):raise PermissionError('Duplicate historical hash-map member')
        manifest=json.loads(stream.read('MANIFEST.json'))
        raw=stream.read('ADAPTER_INITIAL_HASHES.json')
        if hashlib.sha256(raw).hexdigest()!=manifest['ADAPTER_INITIAL_HASHES.json']:
            raise PermissionError('Original initialization hash-map bytes changed')
        hashes=json.loads(raw)
    if (len(hashes)!=992 or sum(n.endswith('.A') for n in hashes)!=496
            or sum(n.endswith('.B') for n in hashes)!=496
            or any(not re.fullmatch('[0-9a-f]{64}',value or '') for value in hashes.values())):
        raise PermissionError('All992 original seeded initialization hashes are required')
    descriptor={'profile':'historical_same_seed_bytes_only_not_capacity_qualification_v1',
        'reference_run_id':361949668,'reference_raw_archive_sha256':expected,
        'initial_adapter_hashes_sha256':digest(hashes),'seed':202610212,
        'devices':[0,1],'first_device_layers':22,'second_device_layers':42,
        'rank':32,'alpha':64,'projections':496,'adapter_parameters':233455616,
        'torch':'2.11.0+cu128','current_training_capacity_qualified':False,
        'historical_checkpoint_or_optimizer_loaded':False}
    return descriptor,hashes
