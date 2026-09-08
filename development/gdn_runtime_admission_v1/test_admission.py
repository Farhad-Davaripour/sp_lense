"""Focused source-admission tests. No package imports or model constructors."""
import ast,copy,hashlib,json,pathlib,sys,types
from . import admission,compat_candidate as candidate
from .source_contract import Rejected,need
ROOT=pathlib.Path(__file__).resolve().parents[2]
def cell(x):return (lambda:x).__closure__[0]
def inert(a=None):return a
def reject(call,code):
    try:call()
    except Rejected as exc:
        need(exc.code==code,"WRONG_REJECTION:"+exc.code)
        return code
    raise ValueError("EXPECTED_REJECTION")
def clone(fn,**kwargs):
    return types.FunctionType(kwargs.get("code",fn.__code__),kwargs.get("globals",fn.__globals__),fn.__name__,
      kwargs.get("defaults",fn.__defaults__),kwargs.get("closure",fn.__closure__))
def main():
    records=[]
    path=pathlib.Path(inert.__code__.co_filename).resolve();code=inert.__code__
    need(all(admission.code_conditions(inert,code,path)[k] for k in ("function_type","code_equal","resolved_path_equal")),"ORIGINAL_ACCEPTANCE")
    candidate._code(inert,code,path);records.append("exact_original_code")
    records.append(reject(lambda:candidate._code(object(),code,path),"CALLABLE_FUNCTION_TYPE"))
    altered=clone(inert,code=code.replace(co_consts=code.co_consts+(99,)))
    records.append(reject(lambda:candidate._code(altered,code,path),"CALLABLE_CODE_IDENTITY"))
    moved=clone(inert,code=code.replace(co_filename=str(ROOT/"development/untrusted.py")))
    need(moved.__code__==code,"PATH_NEGATIVE_REACHES_PATH_OPERAND")
    records.append(reject(lambda:candidate._code(moved,code,path),"CALLABLE_SOURCE_PATH"))
    # Instantiate function objects from authenticated code, not model/tensor objects.
    wrap_path,wrap_code=admission.wrapper_code(ROOT,need)
    mod=types.ModuleType(admission.ACCEL_MODULE);mod.__file__=str(wrap_path)
    hf=types.ModuleType("INERT_HF");base=clone(inert,globals=vars(hf),defaults=(None,None))
    wrapped=types.FunctionType(wrap_code,vars(mod),"wrapped",None,(cell(["conv1d"]),cell(base)))
    prior=sys.modules.get(admission.ACCEL_MODULE);sys.modules[admission.ACCEL_MODULE]=mod
    try:
        value=admission.admit_accelerate_forward(wrapped,hf,code,path,ROOT,need)
        need(value["wrapper_preserved"] and not value["unwrapped_function_executed"],"WRAPPER_PRESERVED")
        records.append("authenticated_wrapper_chain_without_execution")
        wrong=clone(wrapped,code=wrap_code.replace(co_consts=wrap_code.co_consts+("changed",)))
        records.append(reject(lambda:admission.admit_accelerate_forward(wrong,hf,code,path,ROOT,need),"CALLABLE_CODE_IDENTITY"))
        wrong=clone(wrapped,closure=(cell(["changed"]),cell(base)))
        records.append(reject(lambda:admission.admit_accelerate_forward(wrong,hf,code,path,ROOT,need),"ACCEL_CHILDREN"))
        wrongbase=clone(base,code=code.replace(co_consts=code.co_consts+(88,)))
        wrong=clone(wrapped,closure=(cell(["conv1d"]),cell(wrongbase)))
        records.append(reject(lambda:admission.admit_accelerate_forward(wrong,hf,code,path,ROOT,need),"CALLABLE_CODE_IDENTITY"))
        wrong=clone(wrapped,defaults=(None,))
        records.append(reject(lambda:admission.admit_accelerate_forward(wrong,hf,code,path,ROOT,need),"ACCEL_WRAPPER_DEFAULTS"))
        wrongbase=clone(base,defaults=(None,False));wrong=clone(wrapped,closure=(cell(["conv1d"]),cell(wrongbase)))
        records.append(reject(lambda:admission.admit_accelerate_forward(wrong,hf,code,path,ROOT,need),"ACCEL_BASE_BINDING"))
        wrong=clone(wrapped,globals={})
        records.append(reject(lambda:admission.admit_accelerate_forward(wrong,hf,code,path,ROOT,need),"ACCEL_GLOBALS"))
    finally:
        if prior is None:sys.modules.pop(admission.ACCEL_MODULE,None)
        else:sys.modules[admission.ACCEL_MODULE]=prior
    # Exact old source contract; rest of compat AST is identical after two narrow edits.
    original=(ROOT/"diagnostics/fresh_confirmation_first_forward_helper_diagnostic_v3/helper_pkg/compat.py").read_bytes()
    a=ast.parse(original);b=ast.parse(pathlib.Path(candidate.__file__).read_bytes())
    b.body=[x for x in b.body if not(isinstance(x,ast.ImportFrom) and x.module=="admission")]
    oldfn={x.name:x for x in a.body if isinstance(x,ast.FunctionDef)}
    for fn in b.body:
        if isinstance(fn,ast.FunctionDef) and fn.name=="_code":fn.body=copy.deepcopy(oldfn["_code"].body)
        if isinstance(fn,ast.FunctionDef) and fn.name=="_live_sources":
            for i,x in enumerate(fn.body):
                if isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and isinstance(x.value.func,ast.Name) and x.value.func.id=="admit_accelerate_forward":
                    fn.body[i]=copy.deepcopy(oldfn["_live_sources"].body[7])
    # Locate the old forward-code statement by its call expression, independent of line numbering.
    oldforward=next(x for x in oldfn["_live_sources"].body if isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and
      isinstance(x.value.func,ast.Name) and x.value.func.id=="_code" and "unwrap" in ast.dump(x))
    newlive=next(x for x in b.body if isinstance(x,ast.FunctionDef) and x.name=="_live_sources")
    # The replaced call occupies the same original slot (index 7).
    newlive.body[7]=copy.deepcopy(oldforward)
    need(ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),"EXACT_NARROW_PARITY")
    need((ROOT/"development/gdn_runtime_admission_v1/source_contract.py").read_bytes()==
      (ROOT/"diagnostics/fresh_confirmation_first_forward_helper_diagnostic_v3/helper_pkg/source_contract.py").read_bytes(),"SOURCE_CONTRACT_UNCHANGED")
    print(json.dumps({"status":"PASS_FOCUSED_ENGINEERING","cases":records,"case_count":len(records),
      "old_constructor_acceptance_unchanged":True,"rest_of_compat_AST_equal":True,"source_contract_byte_identical":True,
      "imports_torch_hf_tl":False,"model_constructors":0,"forwards":0,"derivatives":0,"encoding":0}))
if __name__=="__main__":main()
