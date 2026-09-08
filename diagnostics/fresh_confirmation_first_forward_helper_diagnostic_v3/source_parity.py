"""Read-only source/AST parity, not an integration execution."""
import ast,copy,difflib,hashlib,json,pathlib,subprocess
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE="diagnostics/fresh_confirmation_first_forward_helper_diagnostic_v2"
COMMIT="1c5049b1a984b03870fd9d1682ae33d2a8253263"
OLD_ATTEMPT="fresh_confirmation_first_forward_helper_diagnostic_attempt_001"
NEW_ATTEMPT="fresh_confirmation_first_forward_helper_diagnostic_attempt_002"
def need(ok,code):
    if not ok:raise ValueError(code)
def raw(name):return subprocess.check_output(["git","show",COMMIT+":"+BASE+"/"+name],cwd=ROOT)
def sha(value):return hashlib.sha256(value).hexdigest()
def tree(value):return ast.parse(value)
def dump(value):return ast.dump(value,include_attributes=False)
def fixture_erasure(value):
    value=tree(value)
    out=[]
    for n in value.body:
        if isinstance(n,ast.Import) and n.names[0].name in {"forward_trace","trace_reader"}:continue
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {"ACTIVE_FIXTURE","TRACE_RECEIPTS"}:continue
        if isinstance(n,ast.FunctionDef):
            if n.name in {"bind_trace","finish_trace","loader_splice"}:continue
            if n.name=="untraced_loader_splice":n.name="loader_splice"
            elif n.name=="fixture":
                n.body=[s for s in n.body if not (isinstance(s,ast.Expr) and isinstance(s.value,ast.Call) and isinstance(s.value.func,ast.Name) and s.value.func.id=="bind_trace")]
            elif n.name=="cleanup":n.body=n.body[2:]
            elif n.name=="outer":
                n.body=[s for s in n.body if not (isinstance(s,ast.Expr) and isinstance(s.value,ast.Call) and isinstance(s.value.func,ast.Name) and s.value.func.id=="finish_trace")]
            elif n.name=="run_group":
                t=next(s for s in n.body if isinstance(s,ast.Try))
                need(isinstance(t.finalbody[-1],ast.Try),"declared exception-safe trace close")
                t.finalbody=t.finalbody[-1].finalbody
                class RemoveTraceExpectations(ast.NodeTransformer):
                    def visit_Expr(self,node):
                        if isinstance(node.value,ast.Call) and node.value.args and isinstance(node.value.args[-1],ast.Constant) and node.value.args[-1].value in {
                            "real finite trace wraps original clean callbacks","original terminal refusal remains first and restoration is traced"}:return None
                        return self.generic_visit(node)
                n=RemoveTraceExpectations().visit(n)
            elif n.name=="main":
                t=next(s for s in n.body if isinstance(s,ast.Try))
                kept=[]
                for s in t.body:
                    if isinstance(s,ast.Expr) and isinstance(s.value,ast.Call):
                        call=s.value
                        if call.args and isinstance(call.args[-1],ast.Constant) and call.args[-1].value in {"NO_TRACE_LEAK_AFTER_BATCH","EXACT_NINE_EXISTING_FIXTURES"}:continue
                        if isinstance(call.func,ast.Name) and call.func.id=="save" and len(call.args)>1 and isinstance(call.args[1],ast.Name) and call.args[1].id=="TRACE_RECEIPTS":continue
                    kept.append(s)
                t.body=kept
        out.append(n)
    value.body=out
    return value
def main():
    freeze=json.loads(raw("SOURCE_FREEZE.json"))
    need(sha(raw("SOURCE_FREEZE.json"))=="32f202c6af1e2780a8ce53db6ebf3410b6eac5dbb2d98b9cdc9596b019f722f0","base freeze hash")
    changed={"support.py","real_boundary.py","BINDINGS.json","integration_tests.py","PROTOCOL.md","TEST_PROTOCOL.json","FUTURE_ROOT_RELEASE.md"}
    replaced={"NARROW_DIFF.patch","PREPARATION_PARITY.json","source_parity.py"}
    identical=[]
    for name,digest in freeze["source_sha256"].items():
        base=raw(name);need(sha(base)==digest,"immutable v2 source")
        if name not in changed|replaced:
            need((HERE/name).read_bytes()==base,"unchanged member "+name)
            identical.append({"path":name,"sha256":digest})
    for name in ("support.py","real_boundary.py"):
        need((HERE/name).read_bytes().replace(NEW_ATTEMPT.encode(),OLD_ATTEMPT.encode())==raw(name),"attempt-only "+name)
    old=json.loads(raw("BINDINGS.json"));new=json.loads((HERE/"BINDINGS.json").read_bytes())
    new["attempt_id"]=old["attempt_id"];new["output_relative"]=old["output_relative"]
    need(new==old,"only binding identity changed")
    need(dump(fixture_erasure((HERE/"integration_tests.py").read_bytes()))==dump(tree(raw("integration_tests.py"))),"trace-only fixture AST erasure")
    oldp=json.loads(raw("TEST_PROTOCOL.json"));newp=json.loads((HERE/"TEST_PROTOCOL.json").read_bytes())
    newp.pop("fixture_only_successor");newp["schema"]=oldp["schema"]
    need(newp==oldp,"same exact cases expectations caps")
    parsed=[]
    for p in sorted(HERE.rglob("*.py")):ast.parse(p.read_bytes());parsed.append(p.relative_to(HERE).as_posix())
    differences=[]
    for name in sorted(changed):
        differences.extend(difflib.unified_diff(raw(name).decode().splitlines(True),(HERE/name).read_text().splitlines(True),
            fromfile=BASE+"/"+name,tofile=HERE.name+"/"+name))
    report={"status":"PASS_SOURCE_PARITY_ONLY","base_commit":COMMIT,"base_inventory_sha256":"5a94d122089dd1c218db189308e573c9121ecf4e43c2e751ba9da0d99caed51b",
      "byte_identical_members":identical,"count":len(identical),"runtime_identity_only":["support.py","real_boundary.py"],
      "binding_identity_only":True,"fixture_AST_erasure_equal":True,"same_cases_and_expectations":True,
      "all_python_parsed_without_import":parsed,"model_imports":0,"test_execution":False,"real_authority":False}
    print(json.dumps({"report":report,"diff":"".join(differences)}))
if __name__=="__main__":main()
