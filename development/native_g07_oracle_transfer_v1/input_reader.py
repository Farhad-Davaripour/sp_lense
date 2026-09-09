"""Authenticated preparation adapter; zero tokenizer/model imports."""
import importlib,importlib.util,json,os,sys
from support import ROOT,require,sha

PREP=ROOT/'development/native_gate_score_preparation_g07_v1'
PREPARATION_SOURCE_SHA256='34bbaf0518d49773072515976dbc048b8d00bf58125283a92e72b101d7a59094'
def adapter():
    raw=(PREP/'SOURCE_FREEZE.json').read_bytes()
    require(sha(raw)==PREPARATION_SOURCE_SHA256,'PREPARATION_MANIFEST_HASH')
    frozen=json.loads(raw)
    for name,digest in frozen['source_sha256'].items():
        require(sha((PREP/name).read_bytes())==digest,'PREPARATION_SOURCE_BYTES')
    for pin in frozen['external_sources']:
        require(sha((ROOT/pin['path']).read_bytes())==pin['sha256'],'PREPARATION_EXTERNAL_BYTES')
    names=('dependencies','plan','storage','renderer','validate','prepare_core')
    for name in names:
        if name in sys.modules:
            require(getattr(sys.modules[name],'__file__',None)==str(PREP/(name+'.py')),'PREPARATION_MODULE_COLLISION')
    sys.path.insert(0,str(PREP))
    try:
        for name in names:importlib.import_module(name)
        spec=importlib.util.spec_from_file_location('native_supervised_prepared_reader',PREP/'prepare_reader.py')
        value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
        return value
    finally:sys.path.remove(str(PREP))

OLD_BUNDLE=ROOT/'development/native_gate_score_capture_g07_v1/root_release'
OLD_RELEASE_SHA256='bc6904a23e403ff1733ea65cf319c3d3068c91ef6f685b89946c6e2d5579a2c7'
INPUTS_SHA256='d451243ad6401f39607103676cbbf4ddc800be1fdab0c20fb484d8ac80e2b202'
SUBMISSION_SHA256='3deac7c4c66d04498364ab52db07d06ebbdd3f741b1b5427a2db1375c669d92a'
def schema_cases():return adapter().schema_cases()
def oracle():
    return {'schema':'g07_opposed_choice_oracle.v1','applicability':{p['case_key']:p['audit_only']['category']=='self_shutdown' for p in schema_cases()},
        'request_rule':'OPPOSITE_UNIQUE_ELIGIBLE_BASELINE_CHOICE','development_only':True}
def require_oracle(value):
    require(value==oracle() and all(type(v) is bool for v in value['applicability'].values()),'EXACT_G07_ORACLE_AUTHORITY')
def read_bundle(base,release,*,synthetic=False):
    require(not synthetic,'NO_PRODUCTION_SYNTHETIC_BYPASS')
    raw=(OLD_BUNDLE/'RELEASE.json').read_bytes();require(sha(raw)==OLD_RELEASE_SHA256,'ACCEPTED_PREPARATION_RELEASE_BYTES')
    old=json.loads(raw)
    require(old['approved'] is True and old['inputs_sha256']==INPUTS_SHA256 and old['admitted_submission_sha256']==SUBMISSION_SHA256,'FIXED_EXISTING_G07_INPUTS')
    require(release['reused_preparation_release_sha256']==OLD_RELEASE_SHA256
        and release['inputs_sha256']==old['inputs_sha256'] and release['text_lock_sha256']==old['text_lock_sha256']
        and release['admitted_submission_sha256']==old['admitted_submission_sha256']
        and release['preparation_files']==old['preparation_files']
        and release['preparation_closure_sha256']==old['preparation_closure_sha256'],'EXACT_REUSED_PREPARATION_JOIN')
    data=adapter().read_bundle(OLD_BUNDLE,old)
    require_oracle(release['oracle_authority']);data['oracle_authority']=release['oracle_authority']
    return data
def read():
    from authority import read_release,RELEASE
    release=read_release(os.environ.get('SP_NATIVE_RELEASE_SHA',''))
    return read_bundle(RELEASE.parent,release)
