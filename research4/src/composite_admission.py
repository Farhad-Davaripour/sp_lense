"""No held-out content access until full audited validation and model freeze."""
import hashlib,json,re
from pathlib import Path
from .io_utils import digest,sha,read,dump
from .composite_data import require_databricks,materialize


def validate_admission(report,package,cfg):
    required=['complete','admitted','raw_evidence_audited','component_detector_validation_admitted','runtime_source_matches_audit','runtime_environment_enforced']
    if report.get('role')!='validation' or any(report.get(k) is not True for k in required):raise PermissionError('Full audited validation has not passed')
    if report.get('package_sha256')!=digest(package):raise PermissionError('Validation belongs to another package')
    if report.get('evaluation_source_commits')!=[report.get('source_commit')]:raise PermissionError('Validation runtime source provenance differs')
    expected=cfg['cohorts']['validation_groups']*len(cfg['cohorts']['conditions'])*len(cfg['evaluation']['conditions'])
    if report.get('observed_trajectories')!=expected or report.get('missing'):raise PermissionError('Validation trajectory coverage is incomplete')
    if set(report['families'])!={str(v) for v in cfg['cohorts']['validation_templates']}:raise PermissionError('Validation family coverage differs')
    for family in report['families'].values():
        if family.get('admitted') is not True or set(family['conditions'])!=set(cfg['evaluation']['conditions']):raise PermissionError('Validation controls / family admission incomplete')


def freeze(root,store,cfg,package,report,partitions):
    require_databricks();validate_admission(report,package,cfg)
    release=read(root/'RELEASE.json');data=store.json('data_v1/DATA_RELEASE.json')
    if report['source_commit']!=release['commit']:raise PermissionError('Validation was executed under another runtime source')
    value={'version':'composite-freeze-v1','source_commit':release['commit'],'runtime_files_digest':digest(release['files']),'protocol_sha256':sha(root/'config/composite_controller_v1.json'),'package':package,'package_sha256':digest(package),'validation_report':report,'validation_report_sha256':digest(report),'validation_partitions':partitions,'heldout_seal_sha256':data['heldout_seal']['sha256'],'heldout_opened':False}
    path='freezes/'+digest(value)+'/FREEZE.json';raw=(json.dumps(value,indent=2)+'\n').encode();store.put(path,raw)
    return {'path':path,'sha256':hashlib.sha256(raw).hexdigest(),'package_sha256':digest(package),'validation_pass':True}


def frozen(root,store,descriptor):
    if not re.fullmatch(r'freezes/[0-9a-f]{64}/FREEZE\.json',descriptor['path']) or not re.fullmatch('[0-9a-f]{64}',descriptor['sha256']):raise PermissionError('A dedicated model-freeze descriptor is required')
    raw=store.get(descriptor['path'])
    if hashlib.sha256(raw).hexdigest()!=descriptor['sha256']:raise PermissionError('Model freeze bytes differ')
    value=json.loads(raw);cfg=read(root/'config/composite_controller_v1.json');release=read(root/'RELEASE.json')
    validate_admission(value['validation_report'],value['package'],cfg)
    if digest(value['validation_report'])!=value['validation_report_sha256'] or digest(value['package'])!=value['package_sha256']:raise PermissionError('Model/validation freeze differs')
    if release['commit']!=value['source_commit'] or value['validation_report']['source_commit']!=value['source_commit'] or digest(release['files'])!=value['runtime_files_digest'] or sha(root/'config/composite_controller_v1.json')!=value['protocol_sha256']:raise PermissionError('Runtime changed after model/configuration freeze')
    return value


def open_heldout(root,store,descriptor):
    require_databricks();value=frozen(root,store,descriptor);cfg=read(root/'config/composite_controller_v1.json')
    # This is the first and only authorized secret read. Every full-validation
    # and exact-runtime admission above runs before it; the original is immutable.
    raw=store.get('sealed/HELDOUT_SEAL.json')
    if hashlib.sha256(raw).hexdigest()!=value['heldout_seal_sha256']:raise PermissionError('Original held-out seal differs')
    seal=json.loads(raw);c=cfg['cohorts']
    if seal['opened'] or seal['templates']!=c['heldout_templates'] or seal['domains']!=c['heldout_domains'] or seal['groups']!=c['heldout_groups']:raise PermissionError('Held-out plan changed')
    output=Path(store.path('heldout_v1'))
    manifest=materialize(cfg,'heldout',seal['identity_secret'],output,{'validation_pass':True,'model_freeze_sha256':descriptor['sha256']})
    result={'status':'materialized_after_audited_validation','freeze':descriptor,'package_sha256':value['package_sha256'],'manifest':manifest,'original_seal_sha256':value['heldout_seal_sha256'],'original_seal_modified':False,'heldout_content_read':True,'model_forwards':0,'optimizer_updates':0}
    dump(output/'OPENED.json',result)
    if hashlib.sha256(store.get('sealed/HELDOUT_SEAL.json')).hexdigest()!=value['heldout_seal_sha256']:raise ValueError('Original held-out seal changed during opening')
    return result
