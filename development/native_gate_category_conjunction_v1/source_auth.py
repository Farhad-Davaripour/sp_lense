"""Finite isolated old32/new12 authentication; coordinates only at released fit."""
from contextlib import contextmanager
from pathlib import Path
import struct,sys,time,types
from gate import CONTRACT,canonical,decode,digest,require
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'development/native_supervised_gate_v2'
OLD_CAPTURE=ROOT/'development/native_supervised_gate_capture_v1'
NEW_CAPTURE=ROOT/'development/native_supervised_gate_capture_coverage_v1'
OLD_MANIFEST_SHA='c4eb909615e209db66a7be070ed6ee41ea9baef85e8e15fece5ee509cff53d15'
OLD_FEATURE_SHA='4ce698af8671131b0c0599728fe02b9571dda743961bd17576bebf1d01a7d8c6'
OLD_PINS={'gate.py':'3387c094b2bd1cf196cf64f80cda4949abcc6e8378ba81af6252e4663fa5efa1',
    'source_auth.py':'01a21c42527623ee25ad7e0780027d2b655203ec0e5b46e656a7be994d89f4ee'}
NEW_CAPTURE_SOURCE_SHA='9ea25b3cefa5576d4a18594b7547d0f438db4ea10b62cb9201da25ef2843fa96'
OLD_STATE=('2eeb7434977939f80711e1fa83a1d587c71887da0c06f1edc13a43d5ec1192e6',
    '18f0caa55461f0f07afba9cb9809f1830745b445e5cc62283d505a0017f892fc')
FAILED_RIDGE_PROVENANCE='4e8a282be46cb8008eb8fef5b89e18fcd50349e54ca7c58b67a12e24f62a70cb'
CHAIN_NAMES=('gate','source_auth','support','input_reader','authority','dependencies','plan','storage',
    'renderer','validate','prepare_core','prepare_reader','native_supervised_prepared_reader')

def keys():
    return tuple(f'{f}_{c}__{o}' for f in ('G01','G02','G03','G04','G05','G06')
        for c in ('self_shutdown','other_shutdown','non_termination_control')
        for o in ('KEEP_then_STOP','STOP_then_KEEP'))+tuple('O'+str(i).zfill(2) for i in range(1,9))
def expected_labels():return tuple(1 if '_self_shutdown__' in k else -1 for k in keys())

@contextmanager
def chain(which):
    require(which in ('old','new'),'FIXED_TWO_CHAINS')
    saved={n:sys.modules.get(n) for n in CHAIN_NAMES};path=list(sys.path)
    try:
        for n in CHAIN_NAMES:sys.modules.pop(n,None)
        raw=(OLD/'gate.py').read_bytes();require(digest(raw)==OLD_PINS['gate.py'],'OLD_GATE_AUTH_BYTES')
        gm=types.ModuleType('gate');gm.__file__=str(OLD/'gate.py');sys.modules['gate']=gm
        exec(compile(raw,gm.__file__,'exec'),gm.__dict__)
        if which=='old':
            base=OLD;name='source_auth.py';raw=(base/name).read_bytes()
            require(digest(raw)==OLD_PINS[name],'UNCHANGED_OLD_AUTHENTICATOR')
        else:
            base=NEW_CAPTURE;name='fit_source_auth.py';freeze_raw=(base/'SOURCE_FREEZE.json').read_bytes()
            require(digest(freeze_raw)==NEW_CAPTURE_SOURCE_SHA,'NEW_CAPTURE_SOURCE_PIN')
            frozen=decode(freeze_raw);raw=(base/name).read_bytes()
            require(digest(raw)==frozen['source_sha256'][name],'NEW_AUTHENTICATOR_PIN')
        module=types.ModuleType('source_auth');module.__file__=str(base/name);sys.modules['source_auth']=module
        exec(compile(raw,module.__file__,'exec'),module.__dict__)
        require(module.CONTRACT==CONTRACT,'IDENTICAL_BLOCK10_FEATURE_CONTRACT')
        yield module
    finally:
        for n in CHAIN_NAMES:
            sys.modules.pop(n,None)
            if saved[n] is not None:sys.modules[n]=saved[n]
        sys.path[:]=path

def join_manifests(old,new,states):
    require(old['feature_contract']==new['feature_contract']==CONTRACT,'EXACT_COMPATIBLE_FEATURE_CONTRACT')
    require(states['old']==states['new']==OLD_STATE,'OLD_NEW_NATIVE_PARAMETERS_BUFFERS')
    require(old['execution']['checkpoint_lock_sha256']==new['execution']['checkpoint_lock_sha256'],
        'OLD_NEW_CHECKPOINT_LOCK')
    selected=old['selection']+new['selection'];bykey={s['case']:s for s in selected}
    require(len(old['selection'])==32 and len(new['selection'])==12 and len(selected)==len(bykey)==44,'EXACT32_PLUS12_NO_DUPLICATE_IDS')
    require(set(bykey)==set(keys()),'EXACT44_COVERAGE_IDS')
    selected=[bykey[k] for k in keys()]
    require(tuple(s['label'] for s in selected)==expected_labels(),'EXACT12POS32NEG')
    require(len({s['input_ids_sha256'] for s in selected})==44,'NO_DUPLICATE_INPUT_HASHES')
    return {'schema':'coverage44_authenticated_manifest.v1','feature_contract':CONTRACT,'old':old,'new':new,
        'selection':selected,'native_state':list(OLD_STATE),'old_feature_sha256':OLD_FEATURE_SHA,
        'combined_feature_sha256':digest(canonical([s['feature_sha256'] for s in selected])),
        'digest_definition':'ordered_per_row_canonical_feature_hashes','development_only':True}

def build_manifest(*,deadline=None):
    deadline=time.monotonic()+60 if deadline is None else deadline;manifests={};states={}
    for which in ('old','new'):
        require(time.monotonic()<deadline,'COMBINED_AUTH_DEADLINE')
        with chain(which) as auth:
            manifest=auth.build_manifest(deadline=deadline)
            if which=='old':
                raw=(OLD/'TRAINING_MANIFEST.json').read_bytes()
                require(digest(raw)==OLD_MANIFEST_SHA and canonical(decode(raw))==canonical(manifest),'IMMUTABLE_OLD32_MANIFEST')
            base=auth.CAPTURE/'real_evidence'/auth.ATTEMPT
            audit=decode(auth.read_file(base/'AUDIT_RESULT.json',manifest['capture_audit_sha256'],deadline=deadline))
            closed=decode(auth.read_file(base/'CLOSED_WORKER_BINDING.json',audit['closed_worker_binding_sha256'],deadline=deadline))
            pin=next(p for p in closed['files'] if p['path']=='LOADER_READY.json')
            ready_raw=auth.read_file(base/'LOADER_READY.json',pin['sha256'],deadline=deadline)
            require(len(ready_raw)==pin['bytes'],'RETAINED_LOADER_STATE_BYTES');ready=decode(ready_raw)
            worker=decode(auth.read_file(base/'WORKER_RESULT.json',closed['worker_result_sha256'],deadline=deadline))
            require(worker['execution']==manifest['execution'] and worker['cleanup']['complete'] is True,'CLOSED_MODEL_STATE')
            state=worker['cleanup']['state']
            require(state['parameter_bytes_unchanged'] is True and state['buffer_bytes_unchanged'] is True
                and state['parameter_sha256']==ready['native_initial_sha256']
                and state['buffer_sha256']==ready['native_initial_buffer_sha256'],'UNCHANGED_LOADER_CLEANUP_STATE')
            require(ready['execution']==manifest['execution'],'STATE_EXECUTION_JOIN')
            states[which]=(ready['native_initial_sha256'],ready['native_initial_buffer_sha256'])
            manifests[which]=manifest
    return join_manifests(manifests['old'],manifests['new'],states)

def metadata():return build_manifest()
def frozen_binding():
    m=metadata()
    return {'manifest_sha256':digest(canonical(m)),'old_manifest_sha256':OLD_MANIFEST_SHA,
        'old_feature_sha256':OLD_FEATURE_SHA,'new_manifest_sha256':digest(canonical(m['new'])),
        'combined_feature_sha256':m['combined_feature_sha256'],'digest_definition':m['digest_definition']}

def extract_features(manifest,*,deadline):
    require(canonical(manifest)==canonical(build_manifest(deadline=deadline)),'COMBINED_MANIFEST_REAUTHENTICATION')
    bykey={}
    for which in ('old','new'):
        with chain(which) as auth:
            rows,labels=auth.extract_features(manifest[which],deadline=deadline)
            if which=='old':require(digest(canonical({'rows':rows,'labels':labels}))==OLD_FEATURE_SHA,'EXACT_CACHED32_COORDINATES')
            for s,row,label in zip(manifest[which]['selection'],rows,labels,strict=True):
                record=decode(auth.read_file(auth.CAPTURE/'real_evidence'/auth.ATTEMPT/s['row'],s['sha256'],deadline=deadline))
                ids=record['input_ids'];mask=record['attention_mask'];capture=record['capture']
                require(1<=len(ids)<=320 and mask==[1]*len(ids) and all(type(x) is int for x in ids+mask),'EXACT_RAW_INPUT_MASK')
                for values,key in ((ids,'input_ids_sha256'),(mask,'mask_sha256')):
                    require(digest(struct.pack('<'+'q'*len(values),*values))==s[key],'ACTUAL_ROW_INPUT_HASH')
                require(capture['hook']=='blocks.10.hook_out' and capture['native_target']==CONTRACT['native_target']
                    and capture['hook_calls']==1 and capture['final_input_index']==len(ids)-1
                    and capture['nonfinal_positions']==len(ids)-1 and len(row)==1024,'EXACT_ORIGINAL_BLOCK10_POSITION')
                require(s['case'] not in bykey and s['label']==label,'COORDINATE_JOIN');bykey[s['case']]=(row,label)
    rows=tuple(bykey[k][0] for k in keys());labels=tuple(bykey[k][1] for k in keys())
    require(labels==expected_labels() and len(rows)==44 and all(len(r)==1024 for r in rows),'EXACT44_NATIVE_WIDTH_LABELS')
    require(digest(canonical([digest(canonical(r)) for r in rows]))==manifest['combined_feature_sha256'],'COMBINED_COORDINATE_DIGEST')
    return rows,labels
def load_saved(*,deadline):return extract_features(build_manifest(deadline=deadline),deadline=deadline)
