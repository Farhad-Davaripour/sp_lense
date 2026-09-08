"""Pure explicit-import join checks; no library/model imports or old-suite rerun."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import types
HERE=Path(__file__).resolve().parent
class Block:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in {'torch','transformers','transformer_lens','numpy','tokenizers','safetensors','sp_lense'}:
            raise RuntimeError('PURE_IMPORT_BOUNDARY')
sys.meta_path.insert(0,Block())
import constructor_operands as operands
import helper_binding
def need(ok,code):
    if not ok:raise RuntimeError(code)
def reject(call):
    try:call()
    except (ValueError,RuntimeError):return
    raise RuntimeError('EXPECTED_REJECTION')
def run():
    started=time.monotonic();events=[];result=[]
    name='inert_hf_definitions';path=(HERE/'PUBLIC_INPUT.json').resolve();digest=hashlib.sha256(path.read_bytes()).hexdigest()
    source=types.SimpleNamespace(PINS={'hf':(str(path),digest)})
    source.authenticate=lambda root:events.append('authenticate_source')
    package={'source_contract':source,'compat':types.SimpleNamespace(HF_MODULE=name)}
    module=types.ModuleType(name);module.__file__=str(path)
    prior=(helper_binding.ensure_package,operands.importlib.util.find_spec,operands.importlib.import_module)
    original=sys.modules.get(name)
    helper_binding.ensure_package=lambda:package
    def specification(value):
        need(value==name,'EXACT_IMPORT_NAME');events.append('source_origin');return types.SimpleNamespace(origin=str(path))
    def importing(value):
        need(value==name,'EXACT_IMPORT_NAME');events.append('import_definitions');sys.modules[name]=module;return module
    operands.importlib.util.find_spec=specification;operands.importlib.import_module=importing
    guard=types.SimpleNamespace(installed=True,forwards=0,derivatives=0,latch=types.SimpleNamespace(admit=lambda:events.append('latch_admit')))
    try:
        need(operands.import_hf_definitions(guard) is module and events==['latch_admit','authenticate_source','source_origin','import_definitions'],'SOURCE_BEFORE_IMPORT_EXACT_MODULE')
        result.append({'case':'clean_authenticated_explicit_import','status':'PASS','order':list(events)})
        for bad in ('guard_not_installed','guard_already_dispatched','terminal_latch','wrong_spec_origin','wrong_returned_module_path','changed_source'):
            events.clear();guard.installed=True;guard.forwards=0;guard.latch.admit=lambda:events.append('latch_admit')
            operands.importlib.util.find_spec=specification;module.__file__=str(path);source.PINS['hf']=(str(path),digest)
            if bad=='guard_not_installed':guard.installed=False
            elif bad=='guard_already_dispatched':guard.forwards=1
            elif bad=='terminal_latch':
                def stopped():raise RuntimeError('ORIGINAL_TERMINAL_REFUSAL')
                guard.latch.admit=stopped
            elif bad=='wrong_spec_origin':operands.importlib.util.find_spec=lambda value:types.SimpleNamespace(origin=str(HERE/'README.md'))
            elif bad=='wrong_returned_module_path':module.__file__=str(HERE/'README.md')
            elif bad=='changed_source':source.PINS['hf']=(str(path),'0'*64)
            reject(lambda:operands.import_hf_definitions(guard))
            if bad in ('guard_not_installed','guard_already_dispatched','terminal_latch','wrong_spec_origin'):need('import_definitions' not in events,'DENIED_BEFORE_IMPORT')
            result.append({'case':bad,'status':'PASS','order':list(events)})
    finally:
        helper_binding.ensure_package,operands.importlib.util.find_spec,operands.importlib.import_module=prior
        if original is None:sys.modules.pop(name,None)
        else:sys.modules[name]=original
    tree=ast.parse((HERE/'candidate_loader.py').read_bytes());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_diagnostic_load_adapter')
    body=next(x for x in fn.body if isinstance(x,ast.Try)).body
    start=next(i for i,x in enumerate(body) if isinstance(x,ast.ImportFrom) and x.module=='constructor_operands')
    sequence=body[start:start+5]
    need(sequence[1].value.func.id=='import_hf_definitions' and sequence[2].value.args[0].value=='BEFORE' and
        isinstance(sequence[3],ast.Assign) and sequence[3].value.func.attr=='load' and sequence[4].value.args[0].value=='AFTER','ACTUAL_CORRECTED_IMPORT_BEFORE_CAPTURE_BEFORE_LOAD')
    result.append({'case':'actual_candidate_source_order','status':'PASS'})
    need(time.monotonic()-started<45,'PURE_TEST_CEILING')
    return {'status':'PASS_PURE_IMPORT_JOIN_ONLY','checks':result,'elapsed_seconds':time.monotonic()-started,
        'library_imports':0,'model_loads':0,'parameter_accesses':0,'tensor_calls':0,'forwards':0,'derivatives':0,'encoding':0,
        'actual_prefix_provider_verified':False,'real_authority':False}
if __name__=='__main__':print(json.dumps(run(),sort_keys=True))
