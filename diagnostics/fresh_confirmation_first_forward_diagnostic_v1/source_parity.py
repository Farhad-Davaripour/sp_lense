"""Model-free AST erasure proof for only the added observational wrappers."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def need(ok,message):
    if not ok:raise ValueError(message)

def digest(raw):return hashlib.sha256(raw).hexdigest()

class RemoveObservations(ast.NodeTransformer):
    def visit_ImportFrom(self,node):
        return None if node.module=="forward_trace" else node
    def visit_With(self,node):
        node=self.generic_visit(node)
        if len(node.items)==1 and isinstance(node.items[0].context_expr,ast.Call) and isinstance(node.items[0].context_expr.func,ast.Name) and node.items[0].context_expr.func.id=="span":
            return node.body
        return node
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id=="traced_context":
            need(len(node.args)==2 and not node.keywords,"context wrapper exact arity");return node.args[1]
        return node

def check():
    old=(ROOT/'diagnostics/fresh_confirmation_real_release_v3/candidate_real_adapter.py').read_bytes()
    need(digest(old)=="992acc44dfaa4bb67591e4eb956ea149ac200a5e3484dc70a422e0f33be8e01c","checked corrected adapter bytes")
    new=(HERE/'candidate_real_adapter.py').read_bytes()
    stripped=RemoveObservations().visit(ast.parse(new));original=ast.parse(old)
    need(ast.dump(stripped,include_attributes=False)==ast.dump(original,include_attributes=False),"all original computation and predicates identical after erasing observations")
    bindings=json.loads((HERE/'BINDINGS.json').read_bytes())
    need(bindings['candidate_adapter']['sha256']==digest(new),"instrumented adapter identity")
    need((HERE/'candidate_loader.py').read_bytes()==(ROOT/'diagnostics/fresh_confirmation_real_release_v3/candidate_loader.py').read_bytes(),"exact corrected loader")
    need((HERE/'loader_diagnostics.py').read_bytes()==(ROOT/'diagnostics/fresh_confirmation_real_release_v3/loader_diagnostics.py').read_bytes(),"retained-order diagnostic helper unchanged")
    need((HERE/'weight_reader.py').read_bytes()==(ROOT/'diagnostics/fresh_confirmation_real_release_v3/weight_reader.py').read_bytes(),"explicit saved weight proof checks unchanged")
    for name in ('hook_binding.py','HOOK_BINDING.json','entry.py','owned_production.py','pinned.py','authority.py','admission.py','plan.py','loader_handoff.py'):
        need((HERE/name).read_bytes()==(ROOT/'diagnostics/fresh_confirmation_real_release_v3'/name).read_bytes(),'inherited '+name)
    oldbindings=json.loads((ROOT/'diagnostics/fresh_confirmation_real_release_v3/BINDINGS.json').read_bytes())
    for key in ('input_binding','input_lock_sha256','runtime_spec_sha256','frozen_runtime_reference','resource_contract_sha256','production_seconds','production_ceiling','loader'):
        need(bindings[key]==oldbindings[key],'unchanged upper science/input binding '+key)
    for path in HERE.glob('*.py'):
        if path.name in ('candidate_real_adapter.py','candidate_loader.py'):continue
        # New saved predicates and expectations must not disappear under -O.
        if path.name in ('forward_trace.py','trace_reader.py','diagnostic_core.py','diagnostic_reader.py','diagnostic_tests.py'):
            need(not any(isinstance(node,ast.Assert) for node in ast.walk(ast.parse(path.read_bytes()))),'no removable mandatory assertions')
    return {'status':'OBSERVATIONAL_AST_PARITY','base_adapter_sha256':digest(old),'candidate_adapter_sha256':digest(new),
        'all_original_numeric_statements_and_predicates_unchanged':True,'model_imports':0,'scientific_method_changes':False}
