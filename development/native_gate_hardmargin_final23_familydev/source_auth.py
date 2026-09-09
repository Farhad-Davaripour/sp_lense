"""Only the closed final-block-23 construction capture may supply fitted features."""
from pathlib import Path
import importlib,json,math,struct,sys,time
from gate import CONTRACT,canonical,decode,digest,require

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
CAPTURE=ROOT/'development/native_supervised_gate_capture_final23_v1'
ATTEMPT='native_supervised_gate_capture_final23_attempt_001'
MAX_FILE_BYTES=5*1024**2
CAPTURE_SOURCE_SHA256='a8b0faee7e1fb776937f48508ff932a95aef5976f156c4bd0ffa441fd9d80178'
PREPARED_INPUT_SHA256='b4f1234f2c072dcdf83c5d4c381f3d35f36b3415af27516b2d8075d65e211970'
PREPARATION_RESULT_SHA256='f7bab766a6f2fc4decca3391e119af5e51b1d7cd1898bfe3b75a9c75d812ec84'
PREPARATION_CLOSURE_SHA256='161fd606c5142cc80488b3aa1d8160002eecb683d1be3ae836351cb917acbb3b'
def keys():
    return tuple(f'{f}_{c}__{o}' for f in ('G01','G02','G03','G04')
        for c in ('self_shutdown','other_shutdown','non_termination_control')
        for o in ('KEEP_then_STOP','STOP_then_KEEP'))+tuple('O'+str(i).zfill(2) for i in range(1,9))
def expected_labels():return tuple(1 if '_self_shutdown__' in k else -1 for k in keys())

def read_file(path,expected=None,*,deadline):
    require(time.monotonic()<deadline,'CAPTURE_AUTH_DEADLINE')
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=MAX_FILE_BYTES,'CAPTURE_FILE_BOUND')
    raw=path.read_bytes();require(expected is None or digest(raw)==expected,'CAPTURE_FILE_HASH');return raw
def authenticate_capture(execution,*,deadline):
    require(CAPTURE_SOURCE_SHA256!='0'*64 and CAPTURE_SOURCE_SHA256==execution['source_freeze_sha256'],'INDEPENDENT_CAPTURE_SOURCE_PIN')
    require(execution['inputs_sha256']==PREPARED_INPUT_SHA256
        and execution['preparation_result_sha256']==PREPARATION_RESULT_SHA256
        and execution['preparation_closure_sha256']==PREPARATION_CLOSURE_SHA256,'EXACT_REUSED_PREPARATION')
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
    execution=audit['execution'];require(parent['execution']==execution and execution['scope']=='ROOT_APPROVED_NATIVE_FINAL23_CONSTRUCTION_CAPTURE_ONLY'
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
    require(manifest['schema']=='native_supervised_gate_training_manifest_v2' and manifest['role']=='CONSTRUCTION'
        and manifest['feature_contract']==CONTRACT and manifest['namespace']=='development/native_supervised_gate_capture_final23_v1'
        and manifest['attempt']==ATTEMPT and manifest['execution']==execution,'CONSTRUCTION_MANIFEST_SCOPE')
    selection=manifest['selection'];require(tuple(x['case'] for x in selection)==keys(),'EXACT32_EXPOSED_DEVELOPMENT_SELECTION')
    expected=tuple(1 if '_self_shutdown__' in key else -1 for key in keys())
    require(tuple(x['label'] for x in selection)==expected and expected.count(1)==8,'EXACT8_POSITIVE24_NEGATIVE')
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
            and row['capture']['hook']=='blocks.23.hook_out' and row['capture']['hook_calls']==1
            and row['capture']['final_input_index']==len(row['input_ids'])-1
            and row['capture']['parameter_versions_unchanged'] is True,'NATIVE_FEATURE_JOIN')
        values=row['h0']
        require(all(type(x) in (int,float) and math.isfinite(x) and struct.unpack('<f',struct.pack('<f',x))[0]==x for x in values),'EXACT_FINITE_FLOAT32_FEATURE')
        require(digest(canonical(values))==selected['feature_sha256'],'FEATURE_COORDINATE_HASH')
        rows.append(tuple(float(x) for x in values));labels.append(selected['label'])
    require(len(rows)==32 and len(set(x['input_ids_sha256'] for x in manifest['selection']))==32,'NO_DUPLICATE_TRAINING_INPUT')
    return tuple(rows),tuple(labels)


def metadata(*,deadline=None):
    """Authenticate closed capture provenance and row hashes, without fitting."""
    return build_manifest(deadline=deadline)

def input_binding(*,deadline=None):
    """Future candidate only: bind saved final23 content after complete authentication."""
    deadline=time.monotonic()+60 if deadline is None else deadline
    manifest=metadata(deadline=deadline)
    rows,labels=extract_features(manifest,deadline=deadline)
    require(tuple(labels)==expected_labels() and all(len(r)==1024 for r in rows),'EXACT_NATIVE32_WIDTH')
    return {'manifest_sha256':digest(canonical(manifest)),
        'feature_sha256':digest(canonical({'rows':rows,'labels':labels}))}

def frozen_binding(*,deadline=None):
    """Check frozen candidate against current closed capture; do not read feature rows."""
    deadline=time.monotonic()+60 if deadline is None else deadline
    inputs=decode(read_file(HERE/'TRAINING_MANIFEST.json',deadline=deadline))
    require(inputs.get('schema')=='hardmargin_familydev_inputs.v1'
        and inputs.get('role')=='EXPOSED_DEVELOPMENT_ONLY','FROZEN_INPUT_SCHEMA')
    binding={key:inputs[key] for key in ('manifest_sha256','feature_sha256')}
    require(all(type(value) is str and len(value)==64
        and all(c in '0123456789abcdef' for c in value) for value in binding.values()),'FROZEN_CONTENT_DIGESTS')
    require(binding['manifest_sha256']==digest(canonical(metadata(deadline=deadline))),'FROZEN_CAPTURE_MANIFEST')
    return binding

def load_saved(*,deadline):
    binding=frozen_binding(deadline=deadline)
    manifest=metadata(deadline=deadline)
    rows,labels=extract_features(manifest,deadline=deadline)
    require(digest(canonical({'rows':rows,'labels':labels}))==binding['feature_sha256'],'EXACT32_FEATURE_CONTENT')
    require(tuple(labels)==expected_labels() and all(len(r)==1024 for r in rows),'EXACT_NATIVE32_WIDTH')
    return rows,labels
