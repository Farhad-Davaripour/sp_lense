"""One model/tokenizer-free preparation; admit locked inputs and derive fake counts."""
import ast
import json
from admission import admit
from plan import build_plan
from bind_runtime import adapted_sources
from support import HERE, SOURCES, require, sha, write_new


def main():
    require(not (HERE/"SOURCE_FREEZE.json").exists() and not (HERE/"BATCH_STARTED.json").exists(), "preparation before any candidate freeze/batch")
    bound = admit()
    plan = build_plan()
    fixture = {"execution_mode":"SYNTHETIC_ONLY","model_predictions":False,"description":"Predeclared arithmetic control-flow fixture; no observed cohort model outputs.","records":[]}
    self_indices = {0,1,6,7,12,13}
    for index,prompt in enumerate(plan["prompts"]):
        self_fixture = index in self_indices
        require(self_fixture == (prompt["category"] == "self"), "fixed synthetic roles align with planned audit slots")
        margin = (-.1 if index%2 == 0 else .1) if self_fixture else None
        constant = [] if self_fixture else ([[32,.1],[33,0.]] if index >= 18 else [[50057,.01],[48964,.01],[100,1.],[101,1.]])
        fixture["records"].append({"input_ids":prompt["input_ids"],"attention_mask":prompt["attention_mask"],
            "input_int64_le_sha256":prompt["input_int64_le_sha256"],"axis_sign":1 if self_fixture else -1,
            "keep_intercept":margin,"constant_logits":constant})
    roles = {p["prompt_id"]:i in self_indices for i,p in enumerate(plan["prompts"])}
    margins = {p["prompt_id"]:fixture["records"][i]["keep_intercept"] for i,p in enumerate(plan["prompts"])}
    retentions = sum(roles[r["prompt_id"]] and r["sign"]*margins[r["prompt_id"]] > 0 for r in plan["requests"])
    flips = sum(roles[r["prompt_id"]] and r["sign"]*margins[r["prompt_id"]] < 0 for r in plan["requests"])
    off = sum(not roles[r["prompt_id"]] for r in plan["requests"])
    require((retentions,flips,off) == (6,6,36), "prospective synthetic branch counts")
    forwards = 24+48+retentions+flips*3
    skips = retentions*8+flips*6
    require(forwards == 96 and skips == 84 and forwards+skips == len(plan["cells"]) == 180, "prospective forward/skip arithmetic")
    source,jury = adapted_sources()
    ast.parse(source)
    ast.parse(jury)
    original = SOURCES.read("638ef2a8f1eac0679c92d07a62cc8ae9bfac41b1","diagnostics/fresh_confirmation_workflow_v1/judge.py").decode()
    def functions(text):
        return {n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    old,new = functions(original),functions(jury)
    parity = {}
    for name in ("norm","f32","quality","accepted","eligible"):
        require(old[name] == new[name], "unchanged independent scientific predicate: "+name)
        parity[name] = sha(new[name].encode())
    write_new("BOUND_PLAN.json",plan)
    write_new("SYNTHETIC_FIXTURE.json",fixture)
    write_new("SYNTHETIC_EXPECTATIONS.json",{"execution_mode":"SYNTHETIC_ONLY","model_predictions":False,
        "verified_before_freeze":True,"real_model_calls":0,"fake_forward_calls":0,"tokenizer_calls":0,
        "verification_kind":"structural schedule and declared linear-fixture arithmetic, not a model run",
        "judge_counts":{"forward_attempts":96,"forward_completed":96,"derivatives":6,"loads":1,"routes":72,"self_endpoints":12,"strict_flips":6,"retentions":6,"off_identities":36},
        "ordinary_accuracy":{"tested":6,"correct":sum(g == "A" for g in plan["ordinary_gold_scoring_only"].values())},
        "skipped_cells":84,"unrun_cells":0})
    write_new("PREPARATION_RECEIPT.json",{"status":"READY_TO_FREEZE_FAKE_CANDIDATE","input_binding":bound["binding"],
        "adapted_engine_sha256":sha(source.encode()),"adapted_judge_sha256":sha(jury.encode()),"unchanged_judge_functions":parity,
        "actual_inputs":24,"cold_requests":48,"maximum_tokens":bound["binding"]["maximum_tokens"],
        "real_model_loads":0,"fake_model_loads":0,"tokenizer_loads":0,"forwards":0,"production_authorized":False})
    print(json.dumps({"status":"READY_TO_FREEZE_FAKE_CANDIDATE","prompts":24,"requests":48,"real_model_calls":0,"tokenizer_calls":0}))


if __name__ == "__main__":
    main()
