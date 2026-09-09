"""Read-only reuse of the authenticated original417-operation bundle, never its authority."""
import importlib.util,json,os
from pathlib import Path
from support import HERE,ROOT,require,sha

RECORD=HERE/'REUSED_INPUTS.json'
ORIGINAL_READER=ROOT/'development/native_supervised_gate_capture_v1/input_reader.py'
ORIGINAL_READER_SHA='5e5a2d8d269c2f3953c9d3fede9bde3262087dd6a04c6577f21d8aebbfff2031'
ORIGINAL_BASE=ROOT/'development/native_supervised_gate_capture_v1/root_release'
INPUT_SHA='b4f1234f2c072dcdf83c5d4c381f3d35f36b3415af27516b2d8075d65e211970'
RESULT_SHA='f7bab766a6f2fc4decca3391e119af5e51b1d7cd1898bfe3b75a9c75d812ec84'
CLOSURE_SHA='161fd606c5142cc80488b3aa1d8160002eecb683d1be3ae836351cb917acbb3b'

def adapter():
    require(sha(ORIGINAL_READER.read_bytes())==ORIGINAL_READER_SHA,'UNCHANGED_INPUT_PROOF_ADAPTER')
    spec=importlib.util.spec_from_file_location('final23_immutable_input_reader',ORIGINAL_READER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.adapter()

def schema_cases():return adapter().schema_cases()

def read_bundle(base,release,*,synthetic=False):
    require(synthetic is False,'ACTUAL_REUSE_ONLY')
    raw=RECORD.read_bytes();require(sha(raw)==release['reused_inputs_sha256'],'REUSED_INPUT_RECORD_HASH')
    record=json.loads(raw)
    require(record['schema']=='final23_readonly_input_reuse.v1' and record['approved'] is False
        and record['scope']=='INPUT_PROOF_REUSE_ONLY' and record['retokenize'] is False
        and Path(record['base']).resolve()==ORIGINAL_BASE.resolve(),'FIXED_READONLY_REUSE')
    bindings=record['bundle_bindings']
    require(bindings['approved'] is False and bindings['inputs_sha256']==INPUT_SHA
        and bindings['preparation_files']['RESULT.json']==RESULT_SHA
        and bindings['preparation_closure_sha256']==CLOSURE_SHA,'EXACT_ACCEPTED_INPUT_PROOF')
    for name in ('inputs_sha256','text_lock_sha256','preparation_files','preparation_closure_sha256','admitted_submission_sha256'):
        require(release[name]==bindings[name],'NEW_RELEASE_REUSE_JOIN_'+name)
    return adapter().read_bundle(ORIGINAL_BASE,bindings)

def read():
    from authority import read_release,RELEASE
    release=read_release(os.environ.get('SP_NATIVE_RELEASE_SHA',''))
    return read_bundle(RELEASE.parent,release)
