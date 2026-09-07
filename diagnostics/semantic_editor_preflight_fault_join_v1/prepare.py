"""Authenticate only parent metadata/sources and freeze repair before tests."""
import ast,difflib,json,subprocess
from core import HERE,ROOT,Budget,read,sha,require

PARENT=ROOT/"diagnostics/semantic_editor_final_pipeline_v1"
COMMIT="db7c3c05b9021273ba0af4fe0fe08fab24e5951e"
INVENTORY="cf9ac8da94ee7fbe8254022f9b3ded557be182b61b5004446c21ff7e77cf7548"

def prepare():
    require(not (HERE/"freeze.json").exists(),"no repeated preparation")
    raw=(PARENT/"FINAL_INVENTORY.json").read_bytes()
    committed=subprocess.check_output(["git","-C",str(ROOT),"show",COMMIT+":diagnostics/semantic_editor_final_pipeline_v1/FINAL_INVENTORY.json"])
    require(sha(raw)==sha(committed)==INVENTORY,"parent inventory and commit binding")
    entries={r["path"]:r for r in json.loads(raw)["files"]};require(len(entries)==693,"parent693 inventory")
    copied=["core.py","editor.py","judge.py","run.py","numeric_audit.py","synthetic_backend.py","learned_gate.py","gate_reference.py","gate_reload.py","guard_candidate.py","hook_record.py","word_scoring.py","word_reference.py","word_boundary.py","mixed_scoring.py","mixed_boundary.py","fitted_parameters.json","source_bindings.json"]
    bindings={}
    for name in copied:
        before=(PARENT/name).read_bytes();after=(HERE/name).read_bytes();entry=entries[name]
        require(len(before)==entry["bytes"] and sha(before)==entry["sha256"],"authenticated parent source "+name)
        bindings[name]={"old_sha256":sha(before),"new_sha256":sha(after),"unchanged":before==after}
        if name not in ("editor.py","core.py","synthetic_backend.py"):require(before==after,"unchanged helper "+name)
    before=(PARENT/"editor.py").read_bytes();after=(HERE/"editor.py").read_bytes()
    removed=b'            raise RoutingMismatch("preflight route failure; no requests performed")\n'
    anchor=b'        if ineligible:raise EligibilityError("finite self applicability failure; no requests performed")\n'
    added=b'        if wrong:raise RoutingMismatch("preflight route failure; no requests performed")\n'
    require(before.count(removed)==before.count(anchor)==1,"unique minimal patch anchors")
    require(after==before.replace(removed,b"",1).replace(anchor,added+anchor,1),"only delay route raise past eligibility collection")
    old_ast=ast.parse(before);new_ast=ast.parse(after)
    unchanged=[n.name for n in old_ast.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name!="evaluate"]
    for name in unchanged:require(ast.dump(next(n for n in old_ast.body if getattr(n,"name",None)==name))==ast.dump(next(n for n in new_ast.body if getattr(n,"name",None)==name)),"unchanged scientific helper "+name)
    budget=Budget(HERE)
    budget.write_bytes("editor.patch","".join(difflib.unified_diff(before.decode().splitlines(True),after.decode().splitlines(True),fromfile="parent/editor.py",tofile="repair/editor.py")).encode())
    budget.write("repair_source_bindings.json",{"parent_commit":COMMIT,"parent_inventory_sha256":INVENTORY,"parent_entries":693,"sources":bindings,
        "unchanged_editor_helpers":unchanged,"unchanged_independent_judge":bindings["judge.py"]["unchanged"],"real_dataset_or_final_inputs_read":False,
        "permitted_changes":{"editor":"delay route raise until eligibility failures recorded","core":"64MiB namespace/5MiB file, tighter local reserves","synthetic_backend":"combined existing route/eligibility injection","inputs":"two fabricated prompts only"}})
    source_names=[p.name for p in HERE.iterdir() if p.is_file() and (p.suffix==".py" or p.name in ("source_bindings.json","fitted_parameters.json","repair_source_bindings.json","README.md","editor.patch"))]
    budget.write("freeze.json",{"schema":"preflight_fault_join_frozen_sources.v1","source_sha256":{name:sha((HERE/name).read_bytes()) for name in sorted(source_names)},
        "fixtures":["combined_once","route_only","eligibility_only"],"invoked_seconds_ceiling":180,"namespace_bytes":64*1024**2,"per_file_bytes":5*1024**2,
        "real_model_loads":0,"tokenizer_calls":0,"sealed_or_selected_final_input_reads":0,"full_matrix_runs":0,"usage_used_percent_before":88})
    print(json.dumps({"status":"FROZEN","old_editor_sha256":sha(before),"new_editor_sha256":sha(after),"source_files":len(source_names)}))

if __name__=="__main__":prepare()
