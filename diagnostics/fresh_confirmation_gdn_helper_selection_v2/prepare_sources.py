"""Preparation-only source comparison; never imports candidate or fixtures."""
import ast,copy,difflib,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[2]
HERE=pathlib.Path(__file__).resolve().parent
OLD=ROOT/"diagnostics/fresh_confirmation_gdn_helper_handoff_v1"
def need(ok,code):
    if not ok:raise RuntimeError(code)
def digest(raw):return hashlib.sha256(raw).hexdigest()
class Erase(ast.NodeTransformer):
    def visit_ImportFrom(self,node):
        if node.module=="selection_reader":return None
        if node.module=="selection":node.names=[x for x in node.names if x.name not in ("saved_graph","graph_sha")]
        return node
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call):
            call=node.value
            if isinstance(call.func,ast.Name):
                if call.func.id=="verify_graph":return None
                if call.func.id=="need" and len(call.args)>1 and isinstance(call.args[1],ast.Constant) and call.args[1].value in (
                    "COMPLETE_ALIAS_GRAPH","SELECTION_GRAPH_TERMINAL"):return None
        return self.generic_visit(node)
    def visit_Assign(self,node):
        if any(isinstance(t,ast.Name) and t.id=="graph_digest" for t in node.targets):return None
        return self.generic_visit(node)
    def visit_Dict(self,node):
        keep=[i for i,k in enumerate(node.keys) if not (isinstance(k,ast.Constant) and k.value in ("selection_graph","selection_graph_sha256"))]
        node.keys=[node.keys[i] for i in keep];node.values=[node.values[i] for i in keep]
        return self.generic_visit(node)
def main():
    facts={"source_only":True,"candidate_imports":0,"fixture_executions":0,"parity":{},"files":{}}
    for name in ("support.py","fingerprint.py"):
        raw=(HERE/name).read_bytes();need(raw==(OLD/name).read_bytes(),"UNCHANGED_BYTES_"+name)
        facts["parity"][name]={"byte_identical":True,"sha256":digest(raw)}
    for name in ("handoff.py","saved_reader.py"):
        old=ast.parse((OLD/name).read_bytes());new=ast.parse((HERE/name).read_bytes())
        erased=Erase().visit(copy.deepcopy(new))
        need(ast.dump(old,include_attributes=False)==ast.dump(erased,include_attributes=False),"AST_ERASURE_"+name)
        facts["parity"][name]={"graph_only_ast_erasure_equal":True,"old_sha256":digest((OLD/name).read_bytes()),
                                "new_sha256":digest((HERE/name).read_bytes())}
    diffs=[]
    for name in ("selection.py","handoff.py","saved_reader.py"):
        diffs.extend(difflib.unified_diff((OLD/name).read_text().splitlines(True),(HERE/name).read_text().splitlines(True),
            fromfile="handoff_v1/"+name,tofile="selection_v2/"+name))
    for p in sorted(HERE.glob("*")):
        if p.is_file():
            raw=p.read_bytes()
            if p.suffix==".py":
                tree=ast.parse(raw);compile(tree,str(p),"exec",dont_inherit=True)
                need(not any(isinstance(x,ast.Assert) for x in ast.walk(tree)),"NO_REMOVABLE_ASSERTS")
            facts["files"][p.name]={"bytes":len(raw),"sha256":digest(raw)}
    facts["narrow_diff"]="".join(diffs)
    print(json.dumps(facts))
if __name__=="__main__":main()
