"""Exact newly certified PRECHOICE29 prerequisite before diagnostic admission."""
from pathlib import Path
import hashlib,importlib,json,sys
HERE=Path(__file__).resolve().parent;FIT=HERE/'fit'
def require(ok,code):
    if not ok:raise ValueError(code)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def raw(path,expected):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=5*1024**2,'FROZEN_FILE_BOUND')
    value=path.read_bytes();require(digest(value)==expected,'FROZEN_FILE_HASH');return value
def checked(path,expected):return json.loads(raw(path,expected))
def verify_frozen(expected_freeze_sha256):
    freeze=checked(FIT/'FROZEN_PRECHOICE_ARTIFACT.json',expected_freeze_sha256)
    require(freeze['schema']=='prechoice29_accepted_artifact.v1' and freeze['approved'] is True
        and freeze['arm']=='PRECHOICE29' and freeze['diagnostic_inputs_used'] is False,'NEW_TRAINING_ARTIFACT_ONLY')
    sources=checked(FIT/'CORE_SOURCE_LOCK.json',freeze['source_sha256'])
    require(set(sources)=={'gate.py','checker.py','source_auth.py','construction.py'},'EXACT_FIT_SOURCE_SET')
    for name,expected in sources.items():raw(FIT/name,expected)
    release_raw=raw(FIT/'RELEASE.json',freeze['release_sha256']);release=json.loads(release_raw)
    require(release['approved'] is True and release['operation']=='one_construction_fit' and release['source_sha256']==freeze['source_sha256']
        and all(release[k] is False for k in ('model_permission','tokenizer_permission','evaluation_permission')),'EXACT_TRAINING_ONLY_RELEASE')
    raw(FIT/'TRAINING_MANIFEST.json',release['training_manifest_sha256'])
    raw(FIT/'CONSTRUCTION_LOCK_DRAFT.json',release['construction_lock_sha256'])
    result=checked(FIT/'construction_attempt_001/RESULT.json',freeze['result_sha256'])
    require(result['scientific_pass'] is True and result['status']=='PRECHOICE29_TRAINING_PASS' and result['fits_attempted']==3,'CERTIFIED_THREE_HEAD_TRAINING')
    stage=result['stages'];require(len(stage)==1 and stage[0]['id']=='PRECHOICE29' and stage[0]['training_count']==stage[0]['training_correct']==29
        and len(stage[0]['heads'])==3 and all(h['status']=='PASS' for h in stage[0]['heads']),'ALL29_TRAINING_CORRECT')
    terminal=checked(FIT/'fit_owner_attempt_001/TERMINAL.json',freeze['owner_terminal_sha256'])
    final=checked(FIT/'fit_owner_attempt_001/FINALIZATION.json',freeze['owner_finalization_sha256'])
    require(terminal['process_technical_complete'] is True and terminal['exit_code']==0
        and final['terminal_sha256']==freeze['owner_terminal_sha256'] and not final['deadline_fault'] and not final['storage_fault'],'CLOSED_RETAINED_FIT_OWNER')
    review=checked(FIT/'INDEPENDENT_ACTUAL_FIT_REVIEW.json',freeze['independent_review_sha256'])
    require(review['schema']=='prechoice_fit_independent_review.v1' and review['status']=='PASS'
        and all(review[k]==freeze[k] for k in ('artifact_sha256','result_sha256','source_sha256','release_sha256'))
        and review['all_three_certificates_verified'] is True and review['all29_training_correct'] is True
        and review['exact_artifact_reload'] is True,'INDEPENDENT_CERTIFICATION_REQUIRED')
    artifact_raw=raw(FIT/'construction_attempt_001/PRECHOICE29_GATE.json',freeze['artifact_sha256'])
    require(result['artifacts']=={'PRECHOICE29':freeze['artifact_sha256']},'RESULT_ARTIFACT_JOIN')
    old={n:sys.modules.get(n) for n in ('gate','checker','source_auth','construction')};path=list(sys.path)
    try:
        for n in old:sys.modules.pop(n,None)
        sys.path.insert(0,str(FIT));module=importlib.import_module('construction')
        module.authorize(release_raw,freeze['release_sha256'])
        bindings={'input_contract_sha256':release['training_manifest_sha256'],
            'construction_lock_sha256':release['construction_lock_sha256'],'source_sha256':release['source_sha256'],
            'retained_input_data_lock_sha256':module.TRAINING_SOURCE_SHA,'development_only':True,'arm':'PRECHOICE29'}
        require(freeze['artifact_bindings']==bindings,'EXACT_RELEASE_DERIVED_ARTIFACT_BINDINGS')
        model=module.load_artifact(artifact_raw,expected_sha256=freeze['artifact_sha256'],expected_bindings=bindings)
        require(module.gate.CONTRACT['position']=='last_shared_preoption_input' and len(model.heads)==3,'NEW_READOUT_ARTIFACT_ONLY')
    finally:
        for n,v in old.items():
            sys.modules.pop(n,None)
            if v is not None:sys.modules[n]=v
        sys.path[:]=path
    return freeze
