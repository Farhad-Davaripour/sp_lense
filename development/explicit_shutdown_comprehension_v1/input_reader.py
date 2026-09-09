"""Authenticated preparation adapter; zero tokenizer/model imports."""
import importlib,importlib.util,json,os,sys
from support import ROOT,require,sha

PREP=ROOT/'development/explicit_shutdown_comprehension_v1/preparation'
PREPARATION_SOURCE_SHA256='0c09d9558d5189a01f2e3772b671b18c5fcc9c9a5a68c697eb0b92e75b5522ce'
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
