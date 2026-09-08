"""Prospective source parity and emitted candidate only; no candidate execution."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
from source_binding import HERE,source,instrumented
from loader_diagnostics import PREDICATES


def sha(raw): return hashlib.sha256(raw).hexdigest()
def init(text):
    cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=="RealAdapter")
    return next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="__init__")
def requires(tree,instrumented=False):
    result=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and ((isinstance(n.func,ast.Name) and n.func.id=="require") or
            (instrumented and isinstance(n.func,ast.Attribute) and n.func.attr=="check")):
            args=n.args[1:] if isinstance(n.func,ast.Attribute) else n.args
            result.append((ast.dump(args[0]),ast.literal_eval(args[1])))
    return result
def main():
    loader,adapter=instrumented();old_loader=source("loader.py");old_adapter=source("real_adapter.py")
    one=requires(ast.parse(old_loader));two=requires(ast.parse(loader),True)
    three=requires(init(old_adapter));four=requires(init(adapter),True)
    assert one==two and three==four and {m for _,m in one+three}==set(PREDICATES)
    old_digest=next(n for n in ast.parse(old_adapter).body if isinstance(n,ast.FunctionDef) and n.name=="parameter_digest")
    new_digest=next(n for n in ast.parse(adapter).body if isinstance(n,ast.FunctionDef) and n.name=="parameter_digest")
    assert ast.dump(old_digest)==ast.dump(new_digest)
    old_exception=next(n for n in ast.walk(ast.parse(old_loader)) if isinstance(n,ast.ExceptHandler))
    new_exception=next(n for n in ast.walk(ast.parse(loader)) if isinstance(n,ast.ExceptHandler))
    assert ast.dump(old_exception)==ast.dump(new_exception)
    for path in HERE.glob("*.py"): ast.parse(path.read_bytes())
    changes=[]
    for name,before,after in (("loader.py",old_loader,loader),("real_adapter.py",old_adapter,adapter)):
        changes.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),"pinned_production_v2/"+name,"diagnostic_candidate/"+name))
    # Outputs are returned to the caller for apply_patch, not silently written.
    print(json.dumps({"candidate_loader.py":loader,"candidate_real_adapter.py":adapter,"NARROW_DIFF.patch":"".join(changes),
        "SOURCE_PARITY.json":json.dumps({"status":"SOURCE_PARITY_VERIFIED_NOT_EXECUTED","loader_predicates":len(one),"constructor_predicates":len(three),
            "predicate_conditions_and_messages_identical":True,"original_loader_exception_cleanup_ast_identical":True,
            "parameter_digest_function_source_sha256":sha(ast.get_source_segment(old_adapter,old_digest).encode()),
            "parameter_digest_ast_identical":True,"model_backend_imports":0,"production_authorized":False},indent=2)+"\n"}))


if __name__=="__main__":main()
