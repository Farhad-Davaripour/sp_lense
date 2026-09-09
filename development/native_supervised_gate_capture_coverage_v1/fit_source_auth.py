"""Only the new32-row native construction capture may supply fitted features."""
from pathlib import Path
import importlib,json,math,struct,sys,time
from gate import CONTRACT,canonical,decode,digest,require

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
CAPTURE=HERE
ATTEMPT='native_supervised_gate_capture_coverage_attempt_001'
MAX_FILE_BYTES=5*1024**2
CAPTURE_SOURCE_SHA256=digest((CAPTURE/'SOURCE_FREEZE.json').read_bytes())
def keys():
    return tuple(f'{f}_{c}__{o}' for f in ('G05','G06')
        for c in ('self_shutdown','other_shutdown','non_termination_control')
        for o in ('KEEP_then_STOP','STOP_then_KEEP'))
def read_file(path,expected=None,*,deadline):
    require(time.monotonic()<deadline,'CAPTURE_AUTH_DEADLINE')
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=MAX_FILE_BYTES,'CAPTURE_FILE_BOUND')
    raw=path.read_bytes();require(expected is None or digest(raw)==expected,'CAPTURE_FILE_HASH');return raw
def authenticate_capture(execution,*,deadline):
    require(CAPTURE_SOURCE_SHA256==execution['source_freeze_sha256'],'INDEPENDENT_CAPTURE_SOURCE_PIN')
    raw=read_file(CAPTURE/'SOURCE_FREEZE.json',CAPTURE_SOURCE_SHA256,deadline=deadline);frozen=decode(raw)
    require({'authority.py','input_reader.py','production_run.py','owned_production.py','support.py','audit_saved.py','loader.py','receiver.py'}<=set(frozen['source_sha256']),'FINITE_CAPTURE_SOURCE_ALLOWLIST')
    for name,sha in frozen['source_sha256'].items():read_file(CAPTURE/name,sha,deadline=deadline)
    for pin in frozen['external_sources']:read_file(ROOT/pin['path'],pin['sha256'],deadline=deadline)
    require(execution.get('actual_authorized') is not False and not execution.get('test_only',False),'NO_SYNTHETIC_CAPTURE_FOR_REAL_FIT')
    for name in ('support','input_reader','authority'):
        if name in sys.modules:require(Path(sys.modules[name].__file__).resolve()==(CAPTURE/(name+'.py')).resolve(),'CAPTURE_MODULE_COLLISION')
    sys.path.insert(0,str(CAPTURE))
    try:
        authority=importlib.import_module('authority');reader=importlib.import_module('input_reader')
        release=authority.read_release(execution['release_sha256'])
        require(authority.execution(release,execution['release_sha256'])==execution,'EXACT_CAPTURE_RELEASE_EXECUTION')
        return reader.read_bundle(authority.RELEASE.parent,release)
    finally:sys.path.remove(str(CAPTURE))
def build_manifest(*,deadline=None):
    deadline=time.monotonic()+60 if deadline is None else deadline
    base=CAPTURE/'real_evidence'/ATTEMPT
    audit_raw=read_file(base/'AUDIT_RESULT.json',deadline=deadline);audit=decode(audit_raw)
    parent_raw=read_file(base/'PARENT_FINAL.json',deadline=deadline);parent=decode(parent_raw)
    require(audit['audit_completed'] is True and audit['scientific_pass'] is True
        and parent['audit_completed'] is True and parent['scientific_pass'] is True
        and parent['worker_quiescent'] is True and parent['audit_quiescent'] is True,'CLOSED_CAPTURE_PASS')
    execution=audit['execution'];require(parent['execution']==execution and execution['scope']=='ROOT_APPROVED_NATIVE_CONSTRUCTION_CAPTURE_ONLY'
        and execution['attempt']==ATTEMPT,'ONLY_NEW_CONSTRUCTION_CAPTURE')
    require(parent['errors']==[] and parent['classification']=='COMPLETE_NATIVE_CONSTRUCTION_CAPTURE','NO_PARENT_TECHNICAL_FAILURE')
    prepared=authenticate_capture(execution,deadline=deadline)
    closed_raw=read_file(base/'CLOSED_WORKER_BINDING.json',audit['closed_worker_binding_sha256'],deadline=deadline);closed=decode(closed_raw)
    require(closed['good_capture'] is True and closed['execution']==execution,'RETAINED_CLOSED_WORKER')
    read_file(base/'WORKER_RESULT.json',closed['worker_result_sha256'],deadline=deadline)
    frozen=read_file(CAPTURE/'SOURCE_FREEZE.json',execution['source_freeze_sha256'],deadline=deadline)
    lock=decode(frozen)
    for name,sha in lock['source_sha256'].items():read_file(CAPTURE/name,sha,deadline=deadline)
    for pin in lock['external_sources']:read_file(ROOT/pin['path'],pin['sha256'],deadline=deadline)
    manifest=audit['training_manifest']
    require(manifest['schema']=='native_supervised_gate_coverage_training_manifest_v1' and manifest['role']=='CONSTRUCTION'
        and manifest['feature_contract']==CONTRACT and manifest['namespace']=='development/native_supervised_gate_capture_coverage_v1'
        and manifest['attempt']==ATTEMPT and manifest['execution']==execution,'CONSTRUCTION_MANIFEST_SCOPE')
    selection=manifest['selection'];require(tuple(x['case'] for x in selection)==keys(),'EXACT12_FRESH_SELECTION')
    expected=tuple(1 if '_self_shutdown__' in key else -1 for key in keys())
    require(tuple(x['label'] for x in selection)==expected and expected.count(1)==4,'EXACT4_POSITIVE8_NEGATIVE')
    require(all(x['row']=='rows/'+x['case']+'__baseline.json' for x in selection),'ONLY_BASELINE_FEATURE_PATHS')
    require(tuple(p['case_key'] for p in prepared['cases'])==keys(),'RELEASED_PREPARATION32')
    inventory={p['path']:p for p in closed['files']}
    for selected,p in zip(selection,prepared['cases'],strict=True):
        pin=inventory[selected['row']]
        require(pin['sha256']==selected['sha256'] and pin['bytes']==selected['bytes'],'RETAINED_ROW_INVENTORY')
        for values,key in ((p['input']['input_ids'],'input_ids_sha256'),(p['input']['attention_mask'],'mask_sha256')):
            require(digest(struct.pack('<'+'q'*len(values),*values))==selected[key],'RELEASED_PREPARED_INPUT_JOIN')
    return {**manifest,'capture_audit_sha256':digest(audit_raw),'capture_parent_sha256':digest(parent_raw)}
def extract_features(manifest,*,deadline):
    require(canonical(manifest)==canonical(build_manifest(deadline=deadline)),'MANIFEST_REAUTHENTICATION')
    rows=[];labels=[];base=CAPTURE/'real_evidence'/ATTEMPT
    for selected in manifest['selection']:
        raw=read_file(base/selected['row'],selected['sha256'],deadline=deadline)
        require(len(raw)==selected['bytes'],'FEATURE_ROW_BYTES');row=decode(raw)
        require(row['case']==selected['case'] and row['phase']=='baseline' and row['status']=='COMPLETE'
            and row['h']==row['h0'] and len(row['h0'])==1024 and row['offset']==[0.]*1024,'ONLY_UNEDITED_NATIVE_FEATURE')
        require(row['input_ids_sha256']==selected['input_ids_sha256'] and row['mask_sha256']==selected['mask_sha256']
            and row['input_dtype']=='float32' and row['capture']['native_target']==CONTRACT['native_target']
            and row['capture']['parameter_versions_unchanged'] is True,'NATIVE_FEATURE_JOIN')
        values=row['h0']
        require(all(type(x) in (int,float) and math.isfinite(x) and struct.unpack('<f',struct.pack('<f',x))[0]==x for x in values),'EXACT_FINITE_FLOAT32_FEATURE')
        require(digest(canonical(values))==selected['feature_sha256'],'FEATURE_COORDINATE_HASH')
        rows.append(tuple(float(x) for x in values));labels.append(selected['label'])
    require(len(rows)==12 and len(set(x['input_ids_sha256'] for x in manifest['selection']))==12,'NO_DUPLICATE_TRAINING_INPUT')
    return tuple(rows),tuple(labels)
