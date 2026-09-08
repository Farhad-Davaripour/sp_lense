"""Source-only preparation parity; no candidate/helper/fixture imports."""
import ast,copy,difflib,hashlib,json,pathlib
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/"diagnostics/fresh_confirmation_first_forward_diagnostic_v1"
SELECTION=ROOT/"diagnostics/fresh_confirmation_gdn_helper_selection_v2"
COMPAT=ROOT/"diagnostics/fresh_confirmation_gdn_helper_compat_v1"
def need(ok,code):
    if not ok:raise ValueError(code)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def tree(raw):return ast.parse(raw)
def form(node):return ast.dump(node,include_attributes=False)
class Erase(ast.NodeTransformer):
    def visit_ImportFrom(self,node):
        if node.module in ("helper_binding","helper_reader"):return None
        return node
    def visit_Global(self,node):
        node.names=[x for x in node.names if x!="LAST_HELPER_STATUS"];return node
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id in ("abort_load","reserve_helpers"):
            return None
        return self.generic_visit(node)
    def visit_Assign(self,node):
        ids=[x.id for x in node.targets if isinstance(x,ast.Name)]
        if any(x in ("helper","helper_proof","LAST_HELPER_STATUS") for x in ids):return None
        if ids==["clean"] and any(isinstance(x,ast.Name) and x.id=="helper_proof" for x in ast.walk(node.value)):return None
        if any(isinstance(x,ast.Subscript) and isinstance(x.slice,ast.Constant) and x.slice.value=="helper_binding" for x in node.targets):return None
        return self.generic_visit(node)
    def visit_If(self,node):
        if any(isinstance(x,ast.Name) and x.id=="LAST_HELPER_STATUS" for x in ast.walk(node.test)):return None
        return self.generic_visit(node)
    def visit_Call(self,node):
        if isinstance(node.func,ast.Name) and node.func.id=="setup" and len(node.args)==2 and isinstance(node.args[1],ast.Lambda):
            return self.visit(node.args[1].body)
        if isinstance(node.func,ast.Name) and node.func.id=="helper_checkpoint":
            return ast.Call(func=node.args[1],args=[],keywords=[])
        if isinstance(node.func,ast.Name) and node.func.id=="helper_close":
            body=node.args[1].body
            need(isinstance(body,ast.Call) and isinstance(body.func,ast.Name) and body.func.id=="diagnostic_cleanup","exact cleanup wrapper")
            body.args[-1]=ast.Attribute(value=ast.Attribute(value=ast.Name(id="adapter",ctx=ast.Load()),attr="guard",ctx=ast.Load()),attr="restore",ctx=ast.Load())
            return self.visit(body)
        return self.generic_visit(node)
    def visit_Dict(self,node):
        keep=[i for i,x in enumerate(node.keys) if not (isinstance(x,ast.Constant) and x.value=="helper_status")]
        node.keys=[node.keys[i] for i in keep];node.values=[node.values[i] for i in keep]
        return self.generic_visit(node)
def normalized(name,raw):
    node=tree(raw)
    if name in ("loader.py","loader_handoff.py"):
        for item in ast.walk(node):
            if isinstance(item,ast.ExceptHandler):item.body=[x for x in item.body if not (isinstance(x,ast.Import) and any(a.name=="sys" for a in x.names))]
    if name=="production_run.py":
        fn=next(x for x in node.body if isinstance(x,ast.FunctionDef) and x.name=="worker")
        tr=next(x for x in fn.body if isinstance(x,ast.Try))
        tr.finalbody=[x for x in tr.finalbody if not (isinstance(x,ast.Import) and any(a.name=="sys" for a in x.names)) and
                      not (isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ("core","helper_status") for t in x.targets))]
    return Erase().visit(node)
def main():
    output={"source_only":True,"candidate_imports":0,"fixture_executions":0,"parity":{},"package":{}}
    for name in ("candidate_real_adapter.py","loader_diagnostics.py","weight_reader.py","diagnostic_counter.py","trace_operations.py",
                 "forward_trace.py","trace_reader.py","diagnostic_support.py","diagnostic_io.py","RUNTIME_SPEC.json","setup_budget.py",
                 "entry.py","owned_production.py","pinned.py","authority.py","admission.py","plan.py","hook_binding.py","HOOK_BINDING.json","OWNED_IDENTITY.json"):
        a=(OLD/name).read_bytes();b=(HERE/name).read_bytes();need(a==b,"UNCHANGED_"+name)
        output["parity"][name]={"byte_identical":True,"sha256":sha(b)}
    for name in ("candidate_loader.py","loader_handoff.py","loader.py","diagnostic_core.py","production_run.py","diagnostic_reader.py"):
        need(form(tree((OLD/name).read_bytes()))==form(normalized(name,(HERE/name).read_bytes())),"HELPER_ERASURE_"+name)
        output["parity"][name]={"helper_only_ast_erasure_equal":True}
    for name in ("real_boundary.py","launch.py"):
        expected=(OLD/name).read_text().replace("fresh_confirmation_first_forward_diagnostic_attempt_001","fresh_confirmation_first_forward_helper_diagnostic_attempt_001")
        need((HERE/name).read_text()==expected,"ATTEMPT_ONLY_"+name)
        output["parity"][name]={"attempt_identity_only":True}
    actual=(HERE/"support.py").read_text()
    actual=actual.replace('    from helper_limits import FILE_CAPS,OTHER_REMAINDER\n','')
    actual=actual.replace('    if not preparation and name in FILE_CAPS:\n        require(len(data)<=FILE_CAPS[name],"reserved helper native file cap")\n','')
    actual=actual.replace(' and name not in FILE_CAPS:',':').replace('other_cap=OTHER_REMAINDER','other_cap=3*MIB//4')
    actual=actual.replace(' and p.name not in FILE_CAPS','')
    expected=(OLD/"support.py").read_text().replace("fresh_confirmation_first_forward_diagnostic_attempt_001","fresh_confirmation_first_forward_helper_diagnostic_attempt_001")
    need(actual==expected,"SUPPORT_ONLY_NATIVE_PARTITION")
    output["parity"]["support.py"]={"attempt_and_existing_native_partition_only":True}
    for name in ("support.py","selection.py","selection_reader.py","saved_reader.py","handoff.py","fingerprint.py","compat.py","source_contract.py"):
        source=(COMPAT if name in ("compat.py","source_contract.py") else SELECTION)/name
        original=tree(source.read_bytes());bound=tree((HERE/"helper_pkg"/name).read_bytes())
        for item in ast.walk(bound):
            if isinstance(item,ast.ImportFrom):item.level=0
        if name=="support.py":
            original.body=[x for x in original.body if not (isinstance(x,ast.FunctionDef) and x.name=="component") and
                           not (isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="ROOT" for t in x.targets))]
            bound.body=[x for x in bound.body if not (isinstance(x,ast.FunctionDef) and x.name=="component") and
                        not (isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="ROOT" for t in x.targets))]
            raw=(HERE/"helper_pkg"/name).read_text()
            need(raw.endswith('def component():\n    authenticate()\n    from . import compat\n    return compat\n'),"PRIVATE_FACTORY_EXACT")
            need("ROOT=HERE.parents[2]" in raw,"PRIVATE_ROOT_EXACT")
        need(form(original)==form(bound),"PACKAGE_IMPORT_BINDING_"+name)
        output["package"][name]={"upstream_sha256":sha(source.read_bytes()),"bound_sha256":sha((HERE/"helper_pkg"/name).read_bytes()),
            "relative_import_binding_only":name!="support.py","support_factory_and_root_only":name=="support.py"}
    t=(HERE/"test_topology.py").read_text().replace("from helper_pkg.support import","from support import").replace("from helper_pkg.selection import","from selection import")
    need(t==(SELECTION/"inert_topology.py").read_text(),"EXACT_INERT_TOPOLOGY_REUSE")
    output["inert_topology_import_rebinding_only"]=True
    before=json.loads((OLD/"BINDINGS.json").read_bytes());after=json.loads((HERE/"BINDINGS.json").read_bytes())
    for key in ("input_binding","input_lock_sha256","runtime_spec_sha256","frozen_runtime_reference","resource_contract_sha256",
                "production_seconds","production_ceiling","candidate_adapter","diagnostic_scope"):
        need(before[key]==after[key],"EXACT_INPUT_SCIENCE_SCOPE_"+key)
    need(after["loader"]["path"]=="diagnostics/fresh_confirmation_first_forward_helper_diagnostic_v2/candidate_loader.py" and
         after["loader"]["sha256"]==sha((HERE/"candidate_loader.py").read_bytes()),"NEW_LOADER_TARGET")
    pins=json.loads((HERE/"SOURCE_PINS.json").read_bytes())["files"]
    need(pins[after["loader"]["commit"]+":"+after["loader"]["path"]]["sha256"]==after["loader"]["sha256"],"ACYCLIC_LOADER_SOURCE_PIN")
    package=json.loads((HERE/"HELPER_PACKAGE_BINDING.json").read_bytes())
    need(after["helper_integration"]["package_binding_sha256"]==sha((HERE/"HELPER_PACKAGE_BINDING.json").read_bytes()),"PACKAGE_BINDING_SHA")
    for name,record in package["files"].items():
        raw=(HERE/name).read_bytes();need(len(raw)==record["bytes"] and sha(raw)==record["sha256"],"PACKAGE_SOURCE_SHA")
    caps=ast.literal_eval(tree((HERE/"helper_limits.py").read_bytes()).body[1].value)
    need(caps=={"HELPER_SETUP.json":65536,"HELPER_TERMINAL.json":65536,"HELPER_ADMISSION.json":8192,"HELPER_OUTER_STATUS.json":32768}
         and sum(caps.values())==172032 and 786432-sum(caps.values())==614400,"EXISTING_RESERVE_PARTITION")
    output["source_input_scope_and_reserves"]={"unchanged_original_fields":True,"candidate_source_pin":after["loader"],
         "helper_reserved_bytes":172032,"other_closeout_remainder":614400,"original_other_closeout":786432}
    for p in HERE.rglob("*.py"):
        node=tree(p.read_bytes());compile(node,str(p),"exec",dont_inherit=True)
        if p.name not in ("candidate_real_adapter.py",):
            need(not any(isinstance(x,ast.Assert) for x in ast.walk(node)),"NO_REMOVABLE_ASSERT_"+p.name)
    changed=("candidate_loader.py","loader_handoff.py","loader.py","support.py","diagnostic_core.py","production_run.py","diagnostic_reader.py")
    output["narrow_diff"]="".join("".join(difflib.unified_diff((OLD/n).read_text().splitlines(True),(HERE/n).read_text().splitlines(True),
                              fromfile="first_forward_v1/"+n,tofile="helper_diagnostic_v2/"+n)) for n in changed)
    print(json.dumps(output))
if __name__=="__main__":main()
