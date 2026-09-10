"""Exact closed52-view authentication and uniform canonical pair averaging."""
from contextlib import contextmanager
from pathlib import Path
import math,struct,sys,time,types
from gate import CONTRACT,canonical,decode,digest,require
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'development/native_gate_hardmargin_coverage_v1'
NEW=ROOT/'development/native_supervised_gate_capture_order_v1'
PAIR_MANIFEST_SHA='6ec27d5f97a97c6ce26a5277285a36d2ee406eb9a4828b21bf67a6d3f78d83ea'
OLD_INPUT_SHA='4c0443844f096df7ffb6e23cb2bb236d5c146d66e339494d27d66f78e099df70'
OLD_AUTH_SHA='19a30fe016ad6f351723cbef1c42093c89834f6f2213d03b35eb52c066bfc885'
GATE_SHA='b9c439049bde05c8e1139f8afa8b754cce9a65506daed167e7c59e90550d108d'
NEW_SOURCE_SHA='6897ad6bfd9d6a38fced133cbe7667c67ee255d721af775a1981e7064fd0b274'
OLD_MANIFEST_SHA='2b1e4e5c4c099059f1d1564a9cb71e27451f4459002b5df741f2b7295af76612'
NEW_MANIFEST_SHA='fae3b4594e1a43151ebb14212edc899190b03d6ef4bcf57338d05072b7711766'
STATE=('2eeb7434977939f80711e1fa83a1d587c71887da0c06f1edc13a43d5ec1192e6','18f0caa55461f0f07afba9cb9809f1830745b445e5cc62283d505a0017f892fc')
PAIR_METHOD={'name':'uniform_canonical_two_order_average_v1','view_count':2,
    'canonical_order':'lexicographic_label_bytes','operation':'0.5*float64(h0_j)+0.5*float64(h1_j)',
    'before':'stage_training_mean_and_row_unit_norm','raw_feature_contract':CONTRACT,
    'ordinary_same_rule':True,'unstructured_inputs':'UNSUPPORTED','deployment_forwards_per_input':2}
CHAIN_NAMES=('gate','source_auth','support','input_reader','authority','dependencies','plan','storage',
    'renderer','validate','prepare_core','prepare_reader','native_supervised_prepared_reader')
def keys():
    return tuple(f'{f}_{c}' for f in ('G01','G02','G03','G04','G05','G06')
        for c in ('self_shutdown','other_shutdown','non_termination_control'))+tuple(f'O{i:02}' for i in range(1,9))
def expected_labels():return tuple(1 if k.endswith('_self_shutdown') else -1 for k in keys())
def pair_groups():
    return tuple({'scenario':k,'view_keys':[k+'__KEEP_then_STOP',k+'__STOP_then_KEEP'],
        'label':1 if k.endswith('_self_shutdown') else -1} for k in keys()[:18])+tuple(
        {'scenario':k,'view_keys':[k,k+'__B_then_A'],'label':-1} for k in keys()[18:])
def checked(path,expected=None):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=5*1024**2,'SAVED_FILE_BOUND')
    raw=path.read_bytes();require(expected is None or digest(raw)==expected,'SAVED_FILE_HASH');return raw
@contextmanager
def chain(which):
    require(which in ('old','new'),'FIXED_TWO_CHAINS');saved={n:sys.modules.get(n) for n in CHAIN_NAMES};path=list(sys.path)
    try:
        for n in CHAIN_NAMES:sys.modules.pop(n,None)
        raw=checked(OLD/'gate.py',GATE_SHA);gm=types.ModuleType('gate');gm.__file__=str(OLD/'gate.py');sys.modules['gate']=gm
        exec(compile(raw,gm.__file__,'exec'),gm.__dict__)
        if which=='old':base=OLD;name='source_auth.py';raw=checked(base/name,OLD_AUTH_SHA)
        else:
            base=NEW;name='fit_source_auth.py';frozen=decode(checked(base/'SOURCE_FREEZE.json',NEW_SOURCE_SHA))
            raw=checked(base/name,frozen['source_sha256'][name])
        module=types.ModuleType('source_auth');module.__file__=str(base/name);sys.modules['source_auth']=module
        exec(compile(raw,module.__file__,'exec'),module.__dict__)
        require(module.CONTRACT==CONTRACT,'UNCHANGED_FEATURE_CONTRACT');yield module
    finally:
        for n in CHAIN_NAMES:
            sys.modules.pop(n,None)
            if saved[n] is not None:sys.modules[n]=saved[n]
        sys.path[:]=path
def validate_pair_manifest(m):
    require(m['schema']=='closed_order_views52_metadata.v1' and m['feature_contract']==CONTRACT
        and tuple(m['native_state'])==STATE and m['view_count']==52 and m['scenario_count']==26
        and m['averaging_performed'] is False and m['fitting_authorized'] is False,'ADMITTED_PAIR_MANIFEST')
    require(canonical(m['groups'])==canonical(pair_groups()),'EXACT_FIXED_SCENARIO_PAIRS')
    s=m['selection'];require(len(s)==52 and len({v['case'] for v in s})==len({v['input_ids_sha256'] for v in s})==52,'EXACT52_DISTINCT_VIEWS')
    expected={k:g['label'] for g in pair_groups() for k in g['view_keys']}
    require({v['case']:v['label'] for v in s}==expected,'EXACT52_PAIR_LABELS')
    return m
def build_manifest(*,deadline=None):
    deadline=time.monotonic()+60 if deadline is None else deadline
    m=validate_pair_manifest(decode(checked(HERE/'CLOSED_52_VIEW_MANIFEST.json',PAIR_MANIFEST_SHA)))
    require(digest(checked(OLD/'TRAINING_MANIFEST.json'))==OLD_INPUT_SHA,'ADMITTED_OLD44_INPUT')
    with chain('old') as auth:
        old=auth.build_manifest(deadline=deadline)
        require(digest(canonical(old))==OLD_MANIFEST_SHA and tuple(old['native_state'])==STATE,'EXACT_OLD44_REAUTHENTICATION')
    with chain('new') as auth:
        new=auth.build_manifest(deadline=deadline)
        require(digest(canonical(new))==NEW_MANIFEST_SHA,'EXACT_NEW8_REAUTHENTICATION')
        base=NEW/'real_evidence'/auth.ATTEMPT
        audit=decode(auth.read_file(base/'AUDIT_RESULT.json',new['capture_audit_sha256'],deadline=deadline))
        closed=decode(auth.read_file(base/'CLOSED_WORKER_BINDING.json',audit['closed_worker_binding_sha256'],deadline=deadline))
        pin=next(v for v in closed['files'] if v['path']=='LOADER_READY.json')
        ready=decode(auth.read_file(base/'LOADER_READY.json',pin['sha256'],deadline=deadline))
        worker=decode(auth.read_file(base/'WORKER_RESULT.json',closed['worker_result_sha256'],deadline=deadline))
        state=worker['cleanup']['state'];require(worker['cleanup']['complete'] and state['parameter_bytes_unchanged']
            and state['buffer_bytes_unchanged'] and (state['parameter_sha256'],state['buffer_sha256'])==STATE
            and (ready['native_initial_sha256'],ready['native_initial_buffer_sha256'])==STATE,'EXACT_NEW8_NATIVE_STATE')
    require(old['old']['execution']['checkpoint_lock_sha256']==new['execution']['checkpoint_lock_sha256'],'SAME_CHECKPOINT_LOCK')
    selected=[]
    for source in (old['old'],old['new'],new):
        selected.extend({**s,'source_namespace':source['namespace']} for s in source['selection'])
    bykey={s['case']:s for s in selected}
    require(canonical([bykey[s['case']] for s in m['selection']])==canonical(m['selection']),'EXACT52_AUTHENTICATED_SELECTION')
    require(time.monotonic()<deadline,'PAIR_AUTH_DEADLINE');return m
def frozen_binding():
    m=build_manifest()
    return {'closed52_manifest_sha256':PAIR_MANIFEST_SHA,'old44_manifest_sha256':OLD_MANIFEST_SHA,
        'new8_manifest_sha256':NEW_MANIFEST_SHA,'ordered52_feature_hashes_sha256':digest(canonical([s['feature_sha256'] for s in m['selection']])),
        'pair_method':PAIR_METHOD,'scenario_groups_sha256':digest(canonical(pair_groups()))}
def average_pair(views):
    require(type(views) in (tuple,list) and len(views)==2,'EXACT_TWO_VIEWS')
    orders=[];rows=[]
    for order,row in views:
        require(type(order) in (tuple,list) and len(order)==2 and all(type(v) is str and v and v.isascii() for v in order)
            and order[0]!=order[1],'TWO_DISTINCT_LABEL_ATTACHED_RECORDS')
        orders.append(tuple(order));rows.append(tuple(row))
    require(orders[0]==tuple(reversed(orders[1])),'EXACT_SWAP_INVOLUTION_PAIR')
    require(len(rows[0])==len(rows[1]) and 1<=len(rows[0])<=1024
        and all(type(v) in (int,float) and math.isfinite(v) for r in rows for v in r),'FINITE_PAIR_ROWS')
    ordered=sorted(zip(orders,rows),key=lambda v:tuple(s.encode('ascii') for s in v[0]))
    result=tuple(.5*float(a)+.5*float(b) for a,b in zip(ordered[0][1],ordered[1][1],strict=True))
    require(all(math.isfinite(v) for v in result),'FINITE_AVERAGE');return result
def average_groups(m,bykey):
    validate_pair_manifest(m);require(set(bykey)=={s['case'] for s in m['selection']},'EXACT_PAIR_COORDINATE_KEYS')
    for selected in m['selection']:
        require(digest(canonical(bykey[selected['case']]))==selected['feature_sha256'],'EXACT_PAIR_COORDINATE_PROVENANCE')
    rows=[]
    for group in m['groups']:
        first,second=group['view_keys'];labels=('A','B') if first==group['scenario'] else ('KEEP','STOP')
        rows.append(average_pair(((labels,bykey[first]),(tuple(reversed(labels)),bykey[second]))))
    return tuple(rows),expected_labels()
def load_saved(*,deadline):
    m=build_manifest(deadline=deadline);bykey={}
    with chain('old') as auth:
        om=auth.build_manifest(deadline=deadline);rows,labels=auth.extract_features(om,deadline=deadline)
        for s,row,label in zip(om['selection'],rows,labels,strict=True):require(label==s['label'],'OLD_LABEL');bykey[s['case']]=row
    with chain('new') as auth:
        nm=auth.build_manifest(deadline=deadline);rows,labels=auth.extract_features(nm,deadline=deadline)
        for s,row,label in zip(nm['selection'],rows,labels,strict=True):require(label==s['label'],'NEW_LABEL');bykey[s['case']]=row
    require(all(len(r)==1024 for r in bykey.values()),'EXACT_NATIVE_WIDTH')
    for s in m['selection']:
        require(digest(canonical(bykey[s['case']]))==s['feature_sha256'],'AUTHENTICATED_PAIR_COORDINATES')
    require(time.monotonic()<deadline,'AVERAGING_WITHIN_SHARED_DEADLINE');return average_groups(m,bykey)
