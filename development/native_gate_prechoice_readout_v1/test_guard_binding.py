"""Final guard delta, unpatched functions with only FIT path fixture binding."""
import importlib.util,json,sys
from pathlib import Path
from test_owned_pipeline import HERE,sha,jb
def main():
    base=Path(sys.argv[1]).resolve();fit=base/'fit'
    path=fit/'FROZEN_PRECHOICE_ARTIFACT.json';original=path.read_bytes();freeze=json.loads(original)
    spec=importlib.util.spec_from_file_location('final_prechoice_guard',HERE/'diagnostic_gate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.FIT=fit
    assert module.verify_frozen(sha(original))==freeze
    bad={**freeze,'artifact_bindings':{**freeze['artifact_bindings'],'source_sha256':'0'*64}};path.write_bytes(jb(bad))
    try:
        try:module.verify_frozen(sha(jb(bad)))
        except ValueError as error:assert str(error)=='EXACT_RELEASE_DERIVED_ARTIFACT_BINDINGS'
        else:raise AssertionError('coherent foreign artifact binding admitted')
    finally:path.write_bytes(original)
    receipt={'status':'PASS_FINAL_GUARD_BINDING','source_sha256':sha((HERE/'diagnostic_gate.py').read_bytes()),
        'fixture':str(base),'positive':True,'coherent_wrong_binding_rejected':True,'FIT_constant_substitution_only':True,
        'model_calls':0,'optimizer_calls':0,'function_mocks':False,'frozen_fixture_restored':sha(path.read_bytes())==sha(original)}
    (base/'FINAL_GUARD_BINDING_TEST.json').write_text(json.dumps(receipt,sort_keys=True),encoding='utf-8');print(json.dumps(receipt))
if __name__=='__main__':main()
