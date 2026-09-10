"""Finite training-only old52/newG08 bridge; never opens HELD_SUBMISSION."""
from contextlib import contextmanager
from pathlib import Path
import sys,time,types
from gate import CONTRACT,canonical,decode,digest,require
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'development/native_gate_order_invariant_v1'
TRAIN=ROOT/'development/native_gate_coverage_increment_train_capture_v1'
OLD_SOURCE_SHA='59440ab93a0201b05dc0a3115cc8de7774e8fd03591de1783142d58df2524c3d'
OLD_GATE_SHA='b9c439049bde05c8e1139f8afa8b754cce9a65506daed167e7c59e90550d108d'
PAIR_MANIFEST_SHA='6ec27d5f97a97c6ce26a5277285a36d2ee406eb9a4828b21bf67a6d3f78d83ea'
TRAIN_SUBMISSION_SHA='32ca6cc2f1eaf871633823bef0d4cfdac4b2dd01ff213933b6537afec03dbe00'
STATE=('2eeb7434977939f80711e1fa83a1d587c71887da0c06f1edc13a43d5ec1192e6','18f0caa55461f0f07afba9cb9809f1830745b445e5cc62283d505a0017f892fc')
PAIR_METHOD={'name':'uniform_canonical_two_order_average_v1','view_count':2,'canonical_order':'lexicographic_label_bytes',
    'operation':'0.5*float64(h0_j)+0.5*float64(h1_j)','before':'stage_training_mean_and_row_unit_norm',
    'raw_feature_contract':CONTRACT,'ordinary_same_rule':True,'unstructured_inputs':'UNSUPPORTED','deployment_forwards_per_input':2}
CHAIN_NAMES=('gate','source_auth','support','input_reader','authority','dependencies','plan','storage','renderer','validate','prepare_core','prepare_reader','native_supervised_prepared_reader')
def keys():
    return tuple(f'{f}_{c}' for f in ('G01','G02','G03','G04','G05','G06') for c in ('self_shutdown','other_shutdown','non_termination_control'))+tuple(f'O{i:02}' for i in range(1,9))+tuple('G08_'+c for c in ('self_shutdown','other_shutdown','non_termination_control'))
def expected_labels():return tuple(1 if k.endswith('_self_shutdown') else -1 for k in keys())
def read(path,sha=None):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=5*1024**2,'FINITE_SOURCE_FILE')
    raw=path.read_bytes();require(sha is None or digest(raw)==sha,'EXACT_SOURCE_FILE');return raw
def train_binding():
    b=decode(read(HERE/'root_release/TRAIN_CAPTURE_BINDINGS.json'))
    require(b['schema']=='coverage_increment_train_capture_binding.v1' and b['approved'] is True and b['role']=='TRAIN'
        and b['namespace']=='development/native_gate_coverage_increment_train_capture_v1'
        and b['submission_sha256']==TRAIN_SUBMISSION_SHA,'EXACT_TRAIN_ONLY_CAPTURE_BINDING')
    read(HERE/'TRAIN_SUBMISSION.json',TRAIN_SUBMISSION_SHA);return b
@contextmanager
def chain(which,binding=None):
    require(which in ('old','train'),'FINITE_TRAIN_CHAINS');saved={n:sys.modules.get(n) for n in CHAIN_NAMES};path=list(sys.path)
    try:
        for n in CHAIN_NAMES:sys.modules.pop(n,None)
        gm=types.ModuleType('gate');gm.__file__=str(OLD/'gate.py');sys.modules['gate']=gm
        exec(compile(read(OLD/'gate.py',OLD_GATE_SHA),gm.__file__,'exec'),gm.__dict__)
        if which=='old':base=OLD;name='source_auth.py';raw=read(base/name,OLD_SOURCE_SHA)
        else:
            require(binding is not None,'TRAIN_BINDING_REQUIRED');base=TRAIN;name='fit_source_auth.py'
            f=decode(read(base/'SOURCE_FREEZE.json',binding['source_freeze_sha256']))
            raw=read(base/name,f['source_sha256'][name])
        m=types.ModuleType('source_auth');m.__file__=str(base/name);sys.modules['source_auth']=m
        exec(compile(raw,m.__file__,'exec'),m.__dict__);require(m.CONTRACT==CONTRACT,'FIXED_NATIVE_FEATURE_CONTRACT');yield m
    finally:
        for n in CHAIN_NAMES:
            sys.modules.pop(n,None)
            if saved[n] is not None:sys.modules[n]=saved[n]
        sys.path[:]=path
def average_pair(views):
    with chain('old') as auth:return auth.average_pair(views)
def validate_separation(old,new):
    require(old['feature_contract']==new['feature_contract']==CONTRACT,'EXACT_OLD_NEW_FEATURE_CONTRACT')
    a=old['selection'];b=new['selection']
    require(len(a)==52 and len(b)==6,'EXACT52_PLUS6_VIEWS')
    require(len({s['case'] for s in a+b})==58,'DISTINCT58_CASE_IDENTITIES')
    require(len({s['input_ids_sha256'] for s in a+b})==58,'DISTINCT58_PREPARED_INPUTS')
def build_manifest(*,deadline=None):
    deadline=time.monotonic()+60 if deadline is None else deadline;b=train_binding()
    with chain('old') as auth:
        old=auth.build_manifest(deadline=deadline);require(digest(canonical(old))==digest(canonical(decode(read(OLD/'CLOSED_52_VIEW_MANIFEST.json',PAIR_MANIFEST_SHA)))),'OLD52_EXACT')
    with chain('train',b) as auth:
        new=auth.build_manifest(deadline=deadline);require(digest(canonical(new))==b['manifest_sha256'],'TRAIN_MANIFEST_PIN')
        expected=tuple(k+'__'+o for k in keys()[-3:] for o in ('KEEP_then_STOP','STOP_then_KEEP'))
        require(tuple(s['case'] for s in new['selection'])==expected and tuple(s['label'] for s in new['selection'])==(1,1,-1,-1,-1,-1),'EXACT_G08_SIX_VIEWS')
        require(new['execution']['training_submission_sha256']==TRAIN_SUBMISSION_SHA,'TRAIN_SUBMISSION_CAPTURE_JOIN')
        base=auth.CAPTURE/'real_evidence'/auth.ATTEMPT
        audit=decode(auth.read_file(base/'AUDIT_RESULT.json',new['capture_audit_sha256'],deadline=deadline))
        closed=decode(auth.read_file(base/'CLOSED_WORKER_BINDING.json',audit['closed_worker_binding_sha256'],deadline=deadline))
        worker=decode(auth.read_file(base/'WORKER_RESULT.json',closed['worker_result_sha256'],deadline=deadline));state=worker['cleanup']['state']
        require(worker['cleanup']['complete'] and state['parameter_bytes_unchanged'] and state['buffer_bytes_unchanged']
            and (state['parameter_sha256'],state['buffer_sha256'])==STATE,'UNCHANGED_TRAIN_NATIVE_STATE')
    validate_separation(old,new)
    require(time.monotonic()<deadline,'TRAIN_AUTH_DEADLINE')
    return {'schema':'coverage_increment_training_manifest.v1','role':'TRAIN_ONLY','old52_sha256':PAIR_MANIFEST_SHA,
        'train_binding':b,'new6':new,'keys':keys(),'labels':expected_labels(),'pair_method':PAIR_METHOD,'held_inputs_used':False}
def frozen_binding():return {'manifest_sha256':digest(canonical(build_manifest())),'old52_sha256':PAIR_MANIFEST_SHA,'training_submission_sha256':TRAIN_SUBMISSION_SHA}
def load_saved(*,deadline):
    m=build_manifest(deadline=deadline)
    with chain('old') as auth:old,labels=auth.load_saved(deadline=deadline)
    require(len(old)==26 and tuple(labels)==expected_labels()[:26],'EXACT_BASELINE26')
    with chain('train',m['train_binding']) as auth:
        new,ys=auth.extract_features(m['new6'],deadline=deadline)
    require(len(new)==6 and tuple(ys)==(1,1,-1,-1,-1,-1),'EXACT_NEW_TRAINING_ROWS')
    added=tuple(average_pair(((('KEEP','STOP'),new[2*i]),(('STOP','KEEP'),new[2*i+1]))) for i in range(3))
    require(time.monotonic()<deadline,'SHARED_AVERAGING_DEADLINE');return tuple(old)+added,expected_labels()
