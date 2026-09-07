"""Source/input assembly only; no tensor module, model or tokenizer construction."""
import ast
import difflib
import json
from support import HERE, SOURCES, SCIENCE_COMMIT, SCIENCE, require, sha, write_new


def main():
    from plan import build_plan
    from bind_production import adapted_sources
    from owned_production import source as owned_source
    plan = build_plan()
    engine,judge = adapted_sources()
    ast.parse(engine)
    ast.parse(judge)
    ast.parse(owned_source())
    write_new("BOUND_PLAN.json",plan)
    from production_run import contract
    limits = contract()
    provenance = json.loads((HERE/"RUNTIME_SPEC_PROVENANCE.json").read_bytes())
    reference = json.loads(SOURCES.read(provenance["commit"],provenance["path"]))
    require(json.loads((HERE/"RUNTIME_SPEC.json").read_bytes()) == {"model":reference["model"],
        "runtime_compatibility":reference["gate"]["runtime_compatibility"],
        "installed_sources_sha256":reference["hook_integration"]["installed_sources_sha256"]},
        "only exact previously pinned model/runtime/hook-source configuration fields")
    require(len(plan["prompts"]) == 24 and len(plan["requests"]) == 48 and len(plan["cells"]) == 180
        and max(len(p["input_ids"]) for p in plan["prompts"]) <= limits["study"]["maximum_tokens"], "exact unchanged production schedule")
    editor = SOURCES.read(SCIENCE_COMMIT,SCIENCE+"editor.py").decode()
    kept = {}
    for node in ast.parse(editor).body:
        if isinstance(node,ast.FunctionDef) and node.name in ("norm","valid","accepted","eligibility","step_recipe"):
            kept[node.name] = sha(ast.dump(node,include_attributes=False).encode())
    require(len(kept) == 5, "all unchanged scientific function source pins")
    old_judge = SOURCES.read("638ef2a8f1eac0679c92d07a62cc8ae9bfac41b1","diagnostics/fresh_confirmation_workflow_v1/judge.py").decode()
    predicates = lambda value: {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(value).body
        if isinstance(n,ast.FunctionDef) and n.name in ("norm","f32","quality","accepted","eligible")}
    require(predicates(old_judge) == predicates(judge), "unchanged independent numerical acceptance predicates")
    old = SOURCES.load("preparation_old_runtime_bind","2e11335f1036532b6f79fb2927dd9616c8a09915","diagnostics/fresh_confirmation_runtime_v1/bind_runtime.py")
    old_engine,old_saved = old.adapted_sources()
    delta = "".join(difflib.unified_diff(old_engine.splitlines(True),engine.splitlines(True),"checked_actual_input_engine","production_adapter_engine"))
    delta += "".join(difflib.unified_diff(old_saved.splitlines(True),judge.splitlines(True),"checked_saved_judge","production_saved_judge"))
    write_new("EXACT_ENGINE_DELTA.patch",delta.encode(),raw=True)
    write_new("PREPARATION_RECEIPT.json",{"status":"PREPARED_NO_MODEL_WORK","engine_sha256":sha(engine.encode()),
        "judge_sha256":sha(judge.encode()),"owned_adapter_sha256":sha(owned_source().encode()),
        "unchanged_scientific_definition_ast_sha256":kept,"independent_predicates_unchanged":True,
        "prompts":24,"requests":48,"conditional_cells":180,"maximum_tokens":max(len(p["input_ids"]) for p in plan["prompts"]),
        "binding":plan["input_binding"],"production_limits":limits["study"],"production_seconds":limits["seconds"],
        "real_model_calls":0,"tokenizer_calls":0,"injection_calls":0,"production_authorized":False,
        "hook_dependency_complete_before_freeze":json.loads((HERE/"HOOK_BINDING.json").read_bytes())["status"] == "COMPLETE"})


if __name__ == "__main__":
    main()
