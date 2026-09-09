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
def schema_cases():return adapter().schema_cases()
def read_bundle(base,release,*,synthetic=False):return adapter().read_bundle(base,release,synthetic=synthetic)
def read():
    from authority import read_release,RELEASE
    release=read_release(os.environ.get('SP_NATIVE_RELEASE_SHA',''))
    return read_bundle(RELEASE.parent,release)
