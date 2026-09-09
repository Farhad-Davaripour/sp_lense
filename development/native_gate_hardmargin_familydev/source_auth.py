"""Pinned delegation to the existing accepted32-capture authenticator; no fit."""
from pathlib import Path
import sys,time,types
from gate import canonical,decode,digest,require
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
LEGACY=ROOT/'development/native_supervised_gate_v2'
PINS={'gate.py':'3387c094b2bd1cf196cf64f80cda4949abcc6e8378ba81af6252e4663fa5efa1',
    'source_auth.py':'01a21c42527623ee25ad7e0780027d2b655203ec0e5b46e656a7be994d89f4ee'}
MANIFEST_SHA='c4eb909615e209db66a7be070ed6ee41ea9baef85e8e15fece5ee509cff53d15'
FEATURE_SHA='4ce698af8671131b0c0599728fe02b9571dda743961bd17576bebf1d01a7d8c6'
FAILED_RIDGE_PROVENANCE='4e8a282be46cb8008eb8fef5b89e18fcd50349e54ca7c58b67a12e24f62a70cb'
def keys():
    return tuple(f'{f}_{c}__{o}' for f in ('G01','G02','G03','G04')
        for c in ('self_shutdown','other_shutdown','non_termination_control')
        for o in ('KEEP_then_STOP','STOP_then_KEEP'))+tuple('O'+str(i).zfill(2) for i in range(1,9))
def expected_labels():return tuple(1 if '_self_shutdown__' in k else -1 for k in keys())
def metadata():
    raw=(LEGACY/'TRAINING_MANIFEST.json').read_bytes();require(digest(raw)==MANIFEST_SHA,'EXACT_ACCEPTED_CAPTURE_MANIFEST')
    m=decode(raw);require(tuple(s['case'] for s in m['selection'])==keys()
        and tuple(s['label'] for s in m['selection'])==expected_labels(),'EXACT32_SOURCE_ASSIGNMENT')
    for name,sha in PINS.items():require(digest((LEGACY/name).read_bytes())==sha,'UNCHANGED_CAPTURE_AUTHENTICATOR')
    return m
def library():
    modules={}
    for name in ('gate','source_auth'):
        raw=(LEGACY/(name+'.py')).read_bytes();require(digest(raw)==PINS[name+'.py'],'PINNED_LEGACY_SOURCE')
        unique='hardmargin_reused_'+name;module=types.ModuleType(unique);module.__file__=str(LEGACY/(name+'.py'))
        sys.modules[unique]=module
        if name=='source_auth':
            previous=sys.modules.get('gate');sys.modules['gate']=modules['gate']
            try:exec(compile(raw,module.__file__,'exec'),module.__dict__)
            finally:
                if previous is None:sys.modules.pop('gate',None)
                else:sys.modules['gate']=previous
        else:exec(compile(raw,module.__file__,'exec'),module.__dict__)
        modules[name]=module
    return modules['source_auth']
def load_saved(*,deadline):
    require(time.monotonic()<deadline,'SOURCE_DEADLINE');m=metadata()
    rows,labels=library().extract_features(m,deadline=deadline)
    require(digest(canonical({'rows':rows,'labels':labels}))==FEATURE_SHA,'EXACT32_FEATURE_CONTENT')
    require(tuple(labels)==expected_labels() and all(len(r)==1024 for r in rows),'EXACT_NATIVE32_WIDTH')
    return rows,labels
