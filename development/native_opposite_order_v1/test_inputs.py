"""Compact provider-free successor delta and exact public-input checks."""
import json
from support import HERE,sha,require,json_bytes
from select_inputs import build,blob,BASE_COMMIT,BASE_PATH

def main():
    inputs,provenance=build()
    require((HERE/'inputs.json').read_bytes()==json_bytes(inputs),'FROZEN_INPUTS_EXACT')
    require((HERE/'INPUT_PROVENANCE.json').read_bytes()==json_bytes(provenance),'PUBLIC_PROVENANCE_EXACT')
    from input_reader import read
    require(read()==inputs,'UNCHANGED_INPUT_READER_ADMITS')
    same=[];bound=[]
    for path in sorted(HERE.glob('*.py')):
        if path.name in ('select_inputs.py','test_inputs.py','prepare.py'):continue
        raw=blob(BASE_COMMIT,BASE_PATH+path.name);expected=raw
        if path.name in ('support.py','loader.py'):
            expected=raw.replace(b'native_development_baseline_attempt_001',b'native_opposite_order_attempt_001');bound.append(path.name)
        else:same.append(path.name)
        require(path.read_bytes()==expected,'ONLY_DECLARED_IMPLEMENTATION_DELTA')
    require((HERE/'CHECKPOINT.json').read_bytes()==blob(BASE_COMMIT,BASE_PATH+'CHECKPOINT.json'),'CHECKPOINT_LOCK_UNCHANGED')
    from authority import LIMITS
    from support import ATTEMPT,output
    require(ATTEMPT=='native_opposite_order_attempt_001' and output().parent==HERE/'real_evidence','SEPARATE_ATTEMPT_PATH')
    require(not (HERE/'root_release').exists() and not (HERE/'real_evidence').exists(),'NO_REAL_RELEASE_OR_ATTEMPT')
    cases=['exact_committed_public_prompt_and_saved_token_recovery','only_answer_display_lines_reversed',
        'ordinary_case_unchanged','unchanged_reader_accepts_fixed_lengths_and_hashes',
        'only_two_implementation_attempt_path_deltas','checkpoint_lock_byte_identity','separate_unreleased_namespace']
    result={'status':'PASS','cases':cases,'unchanged_python_files':same,'path_binding_only_files':bound,
        'base_commit':BASE_COMMIT,'inputs_sha256':sha(json_bytes(inputs)),'provider_imports':0,
        'new_encoding_calls':0,'actual_qwen_loads':0,'real_authorized':False,'limits':LIMITS}
    (HERE/'TEST_INPUTS_RESULT.json').write_bytes(json_bytes(result));print(json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
