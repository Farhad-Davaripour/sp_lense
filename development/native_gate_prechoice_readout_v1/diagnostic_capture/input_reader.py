"""Five immutable prepared chains; TRAIN uses four, no tokenizer/provider imports."""
from contextlib import contextmanager
import importlib.util,json,os,sys
from pathlib import Path
from support import HERE,ROOT,require,sha,json_bytes,check_freeze
from index_contract import bind,int_hash
NAMES=('dependencies','plan','storage','renderer','validate','prepare_core','prepare_reader','native_supervised_prepared_reader','frozen_training')
def keys():
    return tuple(f'{f}_{c}__{o}' for f in ('G10','G11') for c in ('self_shutdown','other_shutdown','non_termination_control') for o in ('KEEP_then_STOP','STOP_then_KEEP'))+tuple(f'{k}__{o}' for k in ('O09','O10') for o in ('A_then_B','B_then_A'))
def checked(path,expected):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=5*1024**2,'RETAINED_FILE_BOUND')
    raw=path.read_bytes();require(sha(raw)==expected,'RETAINED_FILE_HASH');return raw
def data():
    lock=json.loads((HERE/'DATA_LOCK.json').read_bytes())
    require(lock['schema']=='prechoice_retained_diagnostic_inputs.v1' and lock['role']=='DIAGNOSTIC_PRECHOICE'
        and lock['training_chain_indexes']==[0,1,2,3] and lock['diagnostic_chain_indexes']==[4],'EXACT_DATA_ROLES')
    cert=json.loads(checked(ROOT/lock['certificate_path'],lock['certificate_sha256']))
    require(cert['training_pairs']==29 and cert['diagnostic_pairs']==8 and cert['views']==74
        and len(cert['pins'])==89 and len(cert['pairs'])==37,'EXACT_BOUNDARY_CENSUS')
    for pin in cert['pins']:
        raw=checked(ROOT/pin['path'],pin['sha256']);require(len(raw)==pin['bytes'],'RETAINED_PIN_BYTES')
    return lock,cert
@contextmanager
def chain(pin):
    base=ROOT/pin['namespace'];frozen=json.loads(checked(base/'SOURCE_FREEZE.json',pin['source_freeze_sha256']))
    for name,digest in frozen['source_sha256'].items():checked(base/name,digest)
    for p in frozen['external_sources']:checked(ROOT/p['path'],p['sha256'])
    raw=checked(base/'input_reader.py',pin['reader_sha256'])
    previous={n:sys.modules.get(n) for n in NAMES};path=list(sys.path)
    try:
        for n in NAMES:sys.modules.pop(n,None)
        if 'frozen_training.py' in frozen['source_sha256']:
            ft=checked(base/'frozen_training.py',frozen['source_sha256']['frozen_training.py'])
            fspec=importlib.util.spec_from_file_location('frozen_training',base/'frozen_training.py')
            fm=importlib.util.module_from_spec(fspec);sys.modules['frozen_training']=fm
            exec(compile(ft,str(base/'frozen_training.py'),'exec'),fm.__dict__)
        spec=importlib.util.spec_from_file_location('prechoice_immutable_prepared_reader',base/'input_reader.py')
        module=importlib.util.module_from_spec(spec);exec(compile(raw,str(base/'input_reader.py'),'exec'),module.__dict__)
        yield module
    finally:
        for n in NAMES:
            sys.modules.pop(n,None)
            if previous[n] is not None:sys.modules[n]=previous[n]
        sys.path[:]=path
def build_inputs():
    lock,cert=data();cases={};witnesses={k:w for w in cert['pairs'] if w['role']=='EXPOSED_DIAGNOSTIC' for k in w['view_keys']}
    require(set(witnesses)==set(keys()) and len(witnesses)==16,'EXACT_TRAINING_SELECTOR_KEYS')
    for index in lock['diagnostic_chain_indexes']:
        pin=lock['chains'][index];old=ROOT/pin['namespace']/'root_release'
        r=json.loads(checked(old/'RELEASE.json',pin['release_sha256']))
        require(r['approved'] is True,'OLD_PREPARATION_ACCEPTED')
        with chain(pin) as reader:bundle=reader.read_bundle(old,r)
        for case in bundle['cases']:
            key=case['case_key'];require(key not in cases and key in witnesses,'UNIQUE_TRAINING_CASE')
            ids=case['input']['input_ids'];expected=case['input_binding']['derived_input_int64_le_sha256']
            selector=bind(ids,witnesses[key],key,lock['certificate_sha256'],expected_input_ids_sha256=expected)
            cases[key]={**case,'readout_selector':selector}
    require(set(cases)==set(keys()),'COMPLETE58_TRAINING_INPUTS')
    result={'schema':'prechoice_diagnostic_inputs.v1','role':'DIAGNOSTIC_PRECHOICE','cases':[cases[k] for k in keys()],
        'certificate_sha256':lock['certificate_sha256'],'data_lock_sha256':sha((HERE/'DATA_LOCK.json').read_bytes())}
    return result
def read_bundle(base,release):
    require(release['input_data_lock_sha256']==sha((HERE/'DATA_LOCK.json').read_bytes()),'RELEASE_DATA_LOCK')
    result=build_inputs()
    require(result['certificate_sha256']==release['certificate_sha256'],'RELEASE_CERTIFICATE_JOIN')
    require(sha(json_bytes(result))==release['inputs_sha256'],'RELEASE_EXACT58_INPUTS')
    return result
def read():
    from authority import read_release,RELEASE
    release=read_release(os.environ.get('SP_NATIVE_RELEASE_SHA',''))
    return read_bundle(RELEASE.parent,release)
