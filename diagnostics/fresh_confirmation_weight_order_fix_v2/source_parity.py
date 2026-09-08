"""Read-only exact assertion-to-check and inherited raw-source parity."""
import ast
import json
from support import HERE,SOURCES,require,sha

OLD_COMMIT="e7a689b9c795573c58c72e693579c9acb10ac953"
OLD_PREFIX="diagnostics/fresh_confirmation_weight_order_fix_v1/"
def dump(node):return ast.dump(node,include_attributes=False)
def check():
    old=ast.parse(SOURCES.read(OLD_COMMIT,OLD_PREFIX+"weight_reader.py").decode())
    new=ast.parse((HERE/"weight_reader.py").read_text())
    conditions=[dump(n.test) for n in ast.walk(old) if isinstance(n,ast.Assert)]
    checks=[dump(n.args[0]) for n in ast.walk(new) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="need"]
    require(conditions==checks and conditions,"identical ordered mandatory condition ASTs")
    require(not any(isinstance(n,ast.Assert) for n in ast.walk(new)),"reader has no removable assertions")
    class Convert(ast.NodeTransformer):
        def visit_Assert(self,node):
            require(node.msg is None,"original assertions have no message semantics")
            return ast.Expr(value=ast.Call(func=ast.Name(id="need",ctx=ast.Load()),args=[node.test],keywords=[]))
    expected=Convert().visit(old)
    helper=ast.parse('def need(value):\n    if not value: raise ValueError("SAVED_WEIGHT_BINDING_CHECK_FAILED")\n').body[0]
    index=next(i for i,n in enumerate(expected.body) if isinstance(n,ast.FunctionDef) and n.name=="interpret")
    expected.body.insert(index,helper)
    require(dump(expected)==dump(new),"no other reader branch/classification change")
    for row in json.loads((HERE/"INFRASTRUCTURE_PARITY.json").read_bytes())["files"]:
        require((HERE/row["namespace_file"]).read_bytes()==SOURCES.read(row["commit"],row["path"]),"identical inherited candidate/infrastructure bytes")
    for name in ("support.py","real_boundary.py"):
        raw=SOURCES.read(OLD_COMMIT,OLD_PREFIX+name).decode()
        expected=raw.replace("fresh_confirmation_weight_order_fix_v1","fresh_confirmation_weight_order_fix_v2").replace("fresh_confirmation_weight_order_fix_attempt_001","fresh_confirmation_weight_order_fix_attempt_002")
        require((HERE/name).read_text()==expected,"successor identity only")
    # The production-reader change is not weakened by optimized test assertions.
    for name in ("reader_tests.py","source_parity.py"):
        require(not any(isinstance(n,ast.Assert) for n in ast.walk(ast.parse((HERE/name).read_text()))),"test expectations survive optimization")
    return {"status":"EXACT_READER_DELTA_VERIFIED","mandatory_condition_count":len(conditions),
        "condition_ASTs_identical":True,"classifications_and_branches_identical":True,
        "inherited_raw_files_identical":len(json.loads((HERE/"INFRASTRUCTURE_PARITY.json").read_bytes())["files"]),
        "loader_adapter_14_predicates_digest_current_registry_cleanup_unchanged":True,
        "test_assertions":0,"scientific_change":False}
