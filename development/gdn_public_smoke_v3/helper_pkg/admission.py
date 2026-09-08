"""Ordinary development: exact code operands and the pinned accelerate wrapper.
No model imports, invocation, constructor, dispatch, or authorization.
"""
import hashlib
import pathlib
import sys
import types
ACCEL_MODULE="transformers.integrations.accelerate"
ACCEL_PATH=".venv/Lib/site-packages/transformers/integrations/accelerate.py"
ACCEL_SHA="72de9bd04d7827ac83aee81593d0d253768eb7196f01a5f2067aef80a4e80bc5"
WRAPPER_QUALNAME="force_accelerate_hooks.<locals>.decorator.<locals>.wrapped"
FIELDS=("co_argcount","co_posonlyargcount","co_kwonlyargcount","co_nlocals","co_stacksize",
 "co_flags","co_code","co_consts","co_names","co_varnames","co_freevars","co_cellvars",
 "co_name","co_qualname","co_firstlineno","co_linetable","co_exceptiontable")
def code_conditions(fn,expected,path):
    # Finite observations of the SAME three original predicate operands.
    function_type=type(fn) is types.FunctionType
    code=fn.__code__ if function_type else None
    return {"function_type":function_type,"code_equal":bool(code is not None and code==expected),
      "resolved_path_equal":bool(code is not None and pathlib.Path(code.co_filename).resolve()==path),
      "different_code_fields":[k for k in FIELDS if code is not None and getattr(code,k)!=getattr(expected,k)]}
def require_code(fn,expected,path,need):
    value=code_conditions(fn,expected,path)
    need(value["function_type"],"CALLABLE_FUNCTION_TYPE")
    need(value["code_equal"],"CALLABLE_CODE_IDENTITY")
    need(value["resolved_path_equal"],"CALLABLE_SOURCE_PATH")
    return value
def wrapper_code(root,need):
    path=(pathlib.Path(root)/ACCEL_PATH).resolve();raw=path.read_bytes()
    need(hashlib.sha256(raw).hexdigest()==ACCEL_SHA,"ACCEL_SOURCE_SHA")
    top=compile(raw,str(path),"exec",dont_inherit=True)
    matches=[]
    def visit(code):
        if code.co_qualname==WRAPPER_QUALNAME:matches.append(code)
        for child in code.co_consts:
            if type(child) is types.CodeType:visit(child)
    visit(top)
    need(len(matches)==1,"ACCEL_WRAPPER_SOURCE")
    return path,matches[0]
def admit_accelerate_forward(fn,hf_module,expected_base,hf_path,root,need):
    path,expected_wrapper=wrapper_code(root,need)
    require_code(fn,expected_wrapper,path,need)
    module=sys.modules.get(ACCEL_MODULE)
    need(type(module) is types.ModuleType and fn.__globals__ is vars(module) and
         pathlib.Path(module.__file__).resolve()==path,"ACCEL_GLOBALS")
    need(fn.__defaults__ is None and fn.__kwdefaults__ is None and
         "__wrapped__" not in vars(fn),"ACCEL_WRAPPER_DEFAULTS")
    need(fn.__code__.co_freevars==("child_module_names","forward_func") and
         fn.__closure__ is not None and len(fn.__closure__)==2,"ACCEL_CLOSURE")
    children,base=(cell.cell_contents for cell in fn.__closure__)
    need(type(children) is list and children==["conv1d"],"ACCEL_CHILDREN")
    require_code(base,expected_base,hf_path,need)
    need(base.__globals__ is vars(hf_module) and base.__defaults__==(None,None) and
         base.__kwdefaults__ is None and base.__closure__ is None,"ACCEL_BASE_BINDING")
    # Return metadata only. The installed wrapper remains the dispatch target.
    return {"schema":"exact_accelerate_forward_admission.v1","source_sha256":ACCEL_SHA,
      "wrapper_identity":id(fn),"base_identity":id(base),"child_module_names":["conv1d"],
      "wrapper_preserved":True,"unwrapped_function_executed":False}
