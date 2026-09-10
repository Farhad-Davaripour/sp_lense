"""Closed prechoice58 capture only; no historical final-position feature imports."""
from pathlib import Path
from contextlib import contextmanager
import json,math,sys,time,types
from gate import CONTRACT,canonical,decode,digest,require
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];CAPTURE=HERE.parent/'train_capture'
CAPTURE_SOURCE_SHA='5b69510f0b5989a8a749914d4e018ad35de07ebea274042833ffdd2616d5d2ba'
DATA_LOCK_SHA='f1e5bc36099239506ef77d7e50666388c4ccb916eeed0aa21149cdd1f6c45e27'
PAIR_METHOD={'name':'uniform_canonical_two_order_average_v1','view_count':2,'canonical_order':'lexicographic_label_bytes',
    'operation':'0.5*float64(h0_j)+0.5*float64(h1_j)','before':'stage_training_mean_and_row_unit_norm',
    'raw_feature_contract':CONTRACT,'ordinary_same_rule':True,'unstructured_inputs':'UNSUPPORTED','deployment_forwards_per_input':2}
NAMES=('support','input_reader','authority','index_contract','dependencies','plan','storage','renderer','validate','prepare_core','prepare_reader','native_supervised_prepared_reader')
def keys():
    return tuple(f'{f}_{c}' for f in ('G01','G02','G03','G04','G05','G06') for c in ('self_shutdown','other_shutdown','non_termination_control'))+tuple(f'O{i:02}' for i in range(1,9))+tuple(f'G08_{c}' for c in ('self_shutdown','other_shutdown','non_termination_control'))
def expected_labels():return tuple(1 if k.endswith('_self_shutdown') else -1 for k in keys())
def groups():return tuple((k,k+'__B_then_A') if k.startswith('O') else (k+'__KEEP_then_STOP',k+'__STOP_then_KEEP') for k in keys())
@contextmanager
def chain():
    raw=(CAPTURE/'SOURCE_FREEZE.json').read_bytes();require(digest(raw)==CAPTURE_SOURCE_SHA,'EXACT_PRECHOICE_CAPTURE_SOURCE')
    f=decode(raw)
    for n,s in f['source_sha256'].items():require(digest((CAPTURE/n).read_bytes())==s,'CAPTURE_SOURCE_BYTES')
    for p in f['external_sources']:require(digest((ROOT/p['path']).read_bytes())==p['sha256'],'CAPTURE_EXTERNAL_BYTES')
    old={n:sys.modules.get(n) for n in NAMES};path=list(sys.path)
    try:
        for n in NAMES:sys.modules.pop(n,None)
        sys.path.insert(0,str(CAPTURE))
        module=types.ModuleType('prechoice_feature_auth');module.__file__=str(CAPTURE/'fit_source_auth.py')
        exec(compile((CAPTURE/'fit_source_auth.py').read_bytes(),module.__file__,'exec'),module.__dict__)
        require(module.CONTRACT==CONTRACT,'NEW_READOUT_CONTRACT_ONLY');yield module
    finally:
        for n in NAMES:
            sys.modules.pop(n,None)
            if old[n] is not None:sys.modules[n]=old[n]
        sys.path[:]=path
def frozen_binding():
    with chain() as auth:m=auth.build_manifest()
    require(len(m['selection'])==58 and m['feature_contract']==CONTRACT,'EXACT_PRECHOICE58_MANIFEST')
    return {'capture_source_sha256':CAPTURE_SOURCE_SHA,'capture_manifest_sha256':digest(canonical(m)),
        'pair_method':PAIR_METHOD,'groups_sha256':digest(canonical(groups())),'input_data_lock_sha256':DATA_LOCK_SHA}
def average_pair(views):
    require(len(views)==2,'EXACT_TWO_VIEWS');ordered=sorted(views,key=lambda item:tuple(v.encode('ascii') for v in item[0]))
    require(tuple(ordered[0][0])==tuple(reversed(ordered[1][0])),'EXACT_SWAP_INVOLUTION_PAIR')
    a,b=ordered[0][1],ordered[1][1]
    require(len(a)==len(b)==1024 and all(type(x) in (int,float) and math.isfinite(x) for row in (a,b) for x in row),'FINITE_PAIR_ROWS')
    return tuple(.5*float(x)+.5*float(y) for x,y in zip(a,b,strict=True))
def load_saved(*,deadline,expected_binding):
    with chain() as auth:
        manifest=auth.build_manifest(deadline=deadline)
        current={'capture_source_sha256':CAPTURE_SOURCE_SHA,'capture_manifest_sha256':digest(canonical(manifest)),
            'pair_method':PAIR_METHOD,'groups_sha256':digest(canonical(groups())),'input_data_lock_sha256':DATA_LOCK_SHA}
        require(canonical(current)==canonical(expected_binding),'EXACT_RELEASED_CAPTURE_BINDING_BEFORE_FEATURES')
        rows,labels=auth.extract_features(manifest,deadline=deadline)
    bykey={s['case']:(row,label) for s,row,label in zip(manifest['selection'],rows,labels,strict=True)}
    require(set(bykey)=={v for pair in groups() for v in pair},'EXACT58_NO_DIAGNOSTIC_INPUTS')
    result=[]
    for key,(a,b),label in zip(keys(),groups(),expected_labels(),strict=True):
        require(bykey[a][1]==bykey[b][1]==label,'PAIR_TRAINING_LABEL_JOIN')
        symbols=('A','B') if key.startswith('O') else ('KEEP','STOP')
        result.append(average_pair(((symbols,bykey[a][0]),(tuple(reversed(symbols)),bykey[b][0]))))
    require(time.monotonic()<deadline,'SAVED_PAIR_DEADLINE');return tuple(result),expected_labels()
