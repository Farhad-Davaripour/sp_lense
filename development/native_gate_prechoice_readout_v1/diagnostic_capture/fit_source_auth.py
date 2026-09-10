"""Only the new32-row native construction capture may supply fitted features."""
from pathlib import Path
import importlib,json,math,struct,sys,time
from index_contract import CONTRACT,validate_capture
from support import sha as digest,require
canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
decode=json.loads

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
CAPTURE=HERE
ATTEMPT='prechoice_diagnostic_capture_attempt_001'
MAX_FILE_BYTES=5*1024**2
CAPTURE_SOURCE_SHA256=digest((CAPTURE/'SOURCE_FREEZE.json').read_bytes())
def keys():
    from input_reader import keys as fixed_keys
    return fixed_keys()
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
    worker=decode(read_file(base/'WORKER_RESULT.json',closed['worker_result_sha256'],deadline=deadline))
    state=worker['cleanup']['state']
    require(worker['cleanup']['complete'] is True and state['parameter_bytes_unchanged'] is True
        and state['buffer_bytes_unchanged'] is True
        and state['parameter_sha256']=='2eeb7434977939f80711e1fa83a1d587c71887da0c06f1edc13a43d5ec1192e6'
        and state['buffer_sha256']=='18f0caa55461f0f07afba9cb9809f1830745b445e5cc62283d505a0017f892fc','FROZEN_TRAINING_NATIVE_STATE')
    frozen=read_file(CAPTURE/'SOURCE_FREEZE.json',execution['source_freeze_sha256'],deadline=deadline)
    lock=decode(frozen)
    for name,sha in lock['source_sha256'].items():read_file(CAPTURE/name,sha,deadline=deadline)
    for pin in lock['external_sources']:read_file(ROOT/pin['path'],pin['sha256'],deadline=deadline)
    manifest=audit['training_manifest']
    require(manifest['schema']=='prechoice_diagnostic_capture_manifest.v1' and manifest['role']=='DIAGNOSTIC_PRECHOICE'
        and manifest['feature_contract']==CONTRACT and manifest['namespace']=='development/native_gate_prechoice_readout_v1/diagnostic_capture'
        and manifest['attempt']==ATTEMPT and manifest['execution']==execution,'CONSTRUCTION_MANIFEST_SCOPE')
    selection=manifest['selection'];require(tuple(x['case'] for x in selection)==keys(),'EXACT6_FRESH_SELECTION')
    expected=tuple(1 if '_self_shutdown__' in key else -1 for key in keys())
    require(tuple(x['label'] for x in selection)==expected and expected.count(1)==4,'EXACT_TWO_POSITIVE_FOUR_NEGATIVE')
    require(all(x['row']=='rows/'+x['case']+'__baseline.json' for x in selection),'ONLY_BASELINE_FEATURE_PATHS')
    require(tuple(p['case_key'] for p in prepared['cases'])==keys(),'RELEASED_PREPARATION32')
    inventory={p['path']:p for p in closed['files']}
    for selected,p in zip(selection,prepared['cases'],strict=True):
        require(selected['readout_selector_sha256']==digest(canonical(p['readout_selector']))
            and selected['readout_index']==p['readout_selector']['readout_index'],'AUTHENTICATED_PREPARED_SELECTOR_JOIN')
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
        ids=row['input_ids'];mask=row['attention_mask'];capture=row['capture']
        packed=lambda values:digest(struct.pack('<'+'q'*len(values),*values))
        require(1<=len(ids)<=320 and mask==[1]*len(ids) and all(type(v) is int and 0<=v<248320 for v in ids)
            and all(type(v) is int for v in mask) and packed(ids)==selected['input_ids_sha256']
            and packed(mask)==selected['mask_sha256'],'EXACT_NATIVE_ROW_INPUT_BYTES')
        require(capture['hook']=='blocks.10.hook_out' and capture['hook_calls']==1
            and capture['final_input_index']==len(ids)-1 and capture['nonfinal_positions']==len(ids)-1,'EXACT_NATIVE_OBSERVER_POSITION')
        validate_capture(ids,row['readout_selector'],capture)
        require(capture['readout_selector_sha256']==selected['readout_selector_sha256'] and capture['readout_index']==selected['readout_index'],'EXACT_SELECTED_READOUT_MANIFEST')
        values=row['h0']
        require(all(type(x) in (int,float) and math.isfinite(x) and struct.unpack('<f',struct.pack('<f',x))[0]==x for x in values),'EXACT_FINITE_FLOAT32_FEATURE')
        require(digest(canonical(values))==selected['feature_sha256'],'FEATURE_COORDINATE_HASH')
        rows.append(tuple(float(x) for x in values));labels.append(selected['label'])
    require(len(rows)==16 and len(set(x['input_ids_sha256'] for x in manifest['selection']))==16,'NO_DUPLICATE_TRAINING_INPUT')
    return tuple(rows),tuple(labels)
