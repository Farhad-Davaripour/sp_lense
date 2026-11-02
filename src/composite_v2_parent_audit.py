"""Re-audit existing scratch evidence before a later fit; never run a model."""
import hashlib,io,json,re,zipfile
from pathlib import Path
from .io_utils import read
from .composite_v2_model_audit import audit_model_archive,provider_closure,require_qualification_evidence


def _locator(value,prefix):
    if not isinstance(value,str) or not re.fullmatch('[A-Za-z0-9_./-]+',value) or not value.startswith(prefix) or '..' in value.split('/'):
        raise PermissionError('Parent evidence must stay in the exact Research4 source/phase namespace')
    return value


def extract_source(raw,expected_sha256,destination):
    """Extract bounded metadata/source as data only; no imported code or hooks."""
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('Qualification source archive changed')
    root=Path(destination).resolve();root.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(raw)) as stream:
        infos=stream.infolist();names=[v.filename for v in infos]
        if len(names)!=len(set(names)) or sum(v.file_size for v in infos)>64*1024**2:
            raise ValueError('Duplicate or unbounded source package')
        if any(n.startswith('/') or any(p in ['','..','.'] for p in n.split('/')) or '\\' in n or ':' in n
               or (n.startswith('inputs/') and ('sealed' in n.lower() or 'heldout' in n.lower()))
               or n=='.env' or n.endswith('/.env') for n in names):
            raise PermissionError('Unsafe or held-out source package member')
        for item in infos:
            if item.is_dir() or (item.external_attr>>16)&0o170000==0o120000 or item.flag_bits&1:
                raise PermissionError('Only ordinary unencrypted source files allowed')
            target=(root/item.filename).resolve()
            if not target.is_relative_to(root):raise PermissionError('Source member escaped its owned directory')
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as output:output.write(stream.read(item))
    release=read(root/'RELEASE.json')
    expected={n for n in names if n.startswith(('src/','config/','ddl/'))}
    if set(release.get('files',{}))!=expected or any(n not in names for n in release.get('files',{})):
        raise PermissionError('Release coverage must name only the complete archived source')
    return root


def validate_parent_descriptor(stage,workspace):
    """Metadata only; the returned paths grant no scientific authority."""
    parent=stage.get('qualification',{});commit=parent.get('source_commit')
    if not isinstance(commit,str) or not re.fullmatch('[0-9a-f]{40}',commit) or commit!=stage['source_commit']:
        raise PermissionError('Keep one reviewed common source freeze across scratch and full fit')
    if parent.get('workspace')!=workspace or parent.get('data_release_sha256')!=stage['data_release_sha256']:
        raise PermissionError('Scratch qualification belongs to another data/workspace freeze')
    descriptor=parent.get('source_archive',{})
    source_path=_locator(descriptor.get('path'),'model_sources_v2/'+commit+'/qualification/')
    if source_path!='model_sources_v2/'+commit+'/qualification/V2_MODEL_SOURCE.zip':raise PermissionError('Exact scratch source package required')
    for value in [descriptor.get('sha256'),parent.get('raw_archive_sha256'),parent.get('source_release_sha256'),parent.get('source_input_manifest_sha256')]:
        if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):raise PermissionError('Exact source/raw archive hashes required')
    if type(parent.get('provider_run_id')) is not int or parent['provider_run_id']<=0:raise PermissionError('Exact scratch provider run identity required')
    raw_path=_locator(parent.get('raw_archive_path'),'model_generalized_v2/'+commit+'/qualification_01/')
    if raw_path!='model_generalized_v2/'+commit+'/qualification_01/evidence.zip':
        raise PermissionError('Exact immutable scratch archive required')
    metadata_prefix='model_generalized_v2/'+commit+'/qualification_01/'
    air_path=_locator(parent.get('AIR_response_path'),metadata_prefix)
    jobs_path=_locator(parent.get('Jobs_response_path'),metadata_prefix)
    return {'source_path':source_path,'raw_path':raw_path,'air_path':air_path,'jobs_path':jobs_path}


def qualification_descriptor(evidence,source_archive,raw_archive_path,AIR_response_path,Jobs_response_path,provider_run_id=None):
    """Construct a later-stage metadata descriptor only from genuine evidence.

    Audit activity counters stay distinct from the scratch's verified one
    optimizer update. The runtime will re-audit this descriptor's raw files.
    """
    from .composite_v2_model_audit import ModelEvidence
    if not isinstance(evidence,ModelEvidence):raise PermissionError('Exact raw qualification reader required')
    proof=evidence.receipt()
    if provider_run_id is None:provider_run_id=proof.get('provider_run_id')
    if type(provider_run_id) is not int or provider_run_id!=proof.get('provider_run_id'):
        raise PermissionError('Qualification descriptor must retain the independently reconciled run ID')
    parent={key:proof.get(key) for key in ['source_commit','workspace','data_release_sha256','raw_archive_sha256',
        'source_release_sha256','source_input_manifest_sha256','complete_training_qualification_audited','owned_compute_released']}
    parent.update(optimizer_updates=proof.get('recorded_update_count_verified'),checkpoint_initializes_full_fit=False,
        source_archive=source_archive,raw_archive_path=raw_archive_path,AIR_response_path=AIR_response_path,
        Jobs_response_path=Jobs_response_path,provider_run_id=provider_run_id)
    require_qualification_evidence(evidence,parent)
    validate_parent_descriptor({'source_commit':proof['source_commit'],'data_release_sha256':proof['data_release_sha256'],
                                'qualification':parent},proof['workspace'])
    return parent


def qualification_from_descriptor(stage,workspace,store,scratch,data_archive):
    paths=validate_parent_descriptor(stage,workspace);parent=stage['qualification'];commit=parent['source_commit']
    # API response snapshots and source archive are written by the controller,
    # not by the model. Missing/ambiguous responses block rather than retry fit.
    closure=provider_closure(workspace['host'],parent.get('provider_run_id'),parent['raw_archive_sha256'],
                             store.json(paths['air_path']),store.json(paths['jobs_path']))
    root=extract_source(store.get(paths['source_path']),parent['source_archive']['sha256'],Path(scratch)/'qualification_source')
    evidence=audit_model_archive(store.get(paths['raw_path']),parent['raw_archive_sha256'],root,commit,data_archive,closure=closure)
    require_qualification_evidence(evidence,parent)
    return evidence
