"""Narrow, fixed real-source adapter; lower-level selection is tested on fixtures."""
import json
import sys
from core import ROOT,blob,load_module,require,sha

PACKET_COMMIT="a2850dfc024575685edd9d1b29461d9c9c0d77cb"
PACKET="diagnostics/semantic_editor_final_study_plan_v1"
PACKET_INVENTORY="879f0f5d7a5dae167c18c204782d64cc591aa190ddfed92f65a96a82dc67d84a"
SOURCE_COMMIT="297a78a2eb9958cc4adeda6ef4f82e6a0a49c584"
SOURCE_PATH="data/conditional_gate_pilot_cases.json"
SOURCE_SHA="0f18c04ae4420883d01c7cff0fbc7e6c688b8c33f404a0d8f687b5bc7489b9da"
FAMILIES=("cg_f08_ferry_schedule","cg_f09_observatory_tools","cg_f10_greenhouse_budget")
CATEGORIES=("self_shutdown","other_shutdown","control")
CASE_IDS=tuple(f"{f}__v1__{c}" for f in FAMILIES for c in CATEGORIES)
PRIMITIVES_PATH="diagnostics/semantic_editor_final_pipeline_v1/selective_input.py"
PRIMITIVES_SHA="8f8d7f4fe1fa0ded756afe722e8298cdc27d4c45d040647f6706c12157ca2a32"
RENDERER_PATH="src/sp_lense/conditional_gate_data.py"
RENDERER_SHA="6accf3e7c47ede747a12eb76d852bb3fb2f1dedca9365cfe58ff8261fc0a9654"

def primitives():
    require(sha((ROOT/PRIMITIVES_PATH).read_bytes())==PRIMITIVES_SHA,"frozen transform/span source")
    return load_module("final_input_frozen_spans",PRIMITIVES_PATH)

def packet():
    invraw=blob(PACKET_COMMIT,f"{PACKET}/FINAL_INVENTORY.json")
    require(sha(invraw)==PACKET_INVENTORY,"packet inventory")
    inv={x["path"]:x for x in json.loads(invraw)["files"]};result={}
    for name in ("cohort.json","plan.json","source_bindings.json"):
        raw=blob(PACKET_COMMIT,f"{PACKET}/{name}")
        require(sha(raw)==inv[name]["sha256"] and len(raw)==inv[name]["bytes"],"packet artifact binding")
        result[name]=json.loads(raw)
    semantic=result["cohort.json"]["prompts"][:18]
    require([p["source_case_id"] for p in semantic]==[i for i in CASE_IDS for _ in range(2)],"exact packet case universe")
    require([p["source_preserve_first"] for p in semantic]==[True,False]*9,"exact packet display order")
    require(len(result["cohort.json"]["prompts"])==24,"packet count")
    return result

def _select_exact_spans(raw):
    """Internal fixture-testable parser; no configurable families, variants or IDs.

    Structural bytes and object keys/ID metadata are scanned throughout. Only the
    nine selected scenario-bearing case values are JSON-decoded. Unselected bytes
    are necessarily read/hashed, NOT claimed unread.
    """
    p=primitives();root=p.members(raw);selected=[];scan_ids=[]
    for a,b in p.elements(raw,root["families"][0]):
        family=p.members(raw,a);fid=json.loads(raw[slice(*family["id"])]);scan_ids.append(fid)
        if fid not in FAMILIES:continue
        split=json.loads(raw[slice(*family["split"])])
        control=json.loads(raw[slice(*family["control_kind"])])
        require(split=="sealed_test","fixed manifest-sealed family metadata")
        for x,y in p.elements(raw,family["variants"][0]):
            variant=p.members(raw,x);vid=json.loads(raw[slice(*variant["id"])])
            if vid!="v1":continue
            cases=p.members(raw,variant["cases"][0])
            require(set(cases)==set(CATEGORIES),"fixed case fields/count; category order is packet-declared")
            for category in CATEGORIES:
                fragment=raw[slice(*cases[category])];case=json.loads(fragment)
                require(case["category"]==category and case["id"]==f"{fid}__v1__{category}","selected exact identity")
                selected.append({**case,"family_id":fid,"variant_id":"v1","split":split,
                    "control_kind":control,"source_case_bytes_sha256":sha(fragment)})
    require([c["id"] for c in selected]==list(CASE_IDS),"exact nine selected cases/source order/no duplicates")
    return selected,{"scanned_family_ids":scan_ids,"scenario_values_decoded_case_ids":[c["id"] for c in selected],
        "scenario_value_decode_count":9,"unselected_scenario_values_decoded":False,
        "whole_file_bytes_read_and_hashed":True,"selection":"fixed nine cases only"}

def extract_locked_source():
    """Sole real extraction entry: no path/hash/family arguments or overrides."""
    raw=blob(SOURCE_COMMIT,SOURCE_PATH)
    require(sha(raw)==SOURCE_SHA,"ONLY authorized whole dataset SHA admitted")
    return _select_exact_spans(raw)

def render_selected(cases,cohort):
    require(sha((ROOT/RENDERER_PATH).read_bytes())==RENDERER_SHA,"frozen renderer")
    sys.path.insert(0,str(ROOT/"src"))
    from sp_lense.conditional_gate_data import PilotCase,render_choice_prompt
    p=primitives();result=[]
    for case,base in ((c,b) for c in cases for b in (True,False)):
        declaration=cohort["prompts"][len(result)]
        require(declaration["source_case_id"]==case["id"] and declaration["source_preserve_first"]==base,"no selected substitution")
        pilot=PilotCase(case_id=case["id"],**{k:case[k] for k in ("family_id","variant_id","split","category","current_assistant_status","scenario","preserve_action","comply_action","control_kind")})
        original=render_choice_prompt(pilot,preserve_first=base)
        value,original_hash=p.render_transform(case,base)
        require(original_hash==sha(original["prompt"].encode()),"real renderer byte parity")
        require(value.endswith("\nAnswer:") and "requested outcome for this test" not in value,"no policy/suffix change")
        if case["category"]!="self_shutdown":
            require(p.NEW_P not in value and p.NEW_C not in value and p.OLD_P in value and p.OLD_C in value,"nonself actions unchanged")
        result.append({**declaration,"text_status":"SELECTIVELY_REVEALED_INPUT_ONLY",
            "prompt":value,"prompt_sha256":sha(value.encode()),"source_rendered_sha256":original_hash,
            "source_case_bytes_sha256":case["source_case_bytes_sha256"],"inverse_transform_exact":True,
            "token_map":declaration["token_map_expected"],"source_preserve_label":original["preserve_label"],
            "source_comply_label":original["comply_label"],"source_split":case["split"],"control_kind":case["control_kind"]})
    for ordinary in cohort["prompts"][18:]:
        require(ordinary["category"]=="ordinary_accuracy" and ordinary["token_map_expected"]=={"A":32,"B":33},"ordinary fixed contract")
        result.append({**ordinary,"prompt_sha256":sha(ordinary["prompt"].encode()),"token_map":ordinary["token_map_expected"],
                       "prompt_and_truth_byte_semantics_unchanged":True})
    require(len(result)==24 and len({p["prompt_id"] for p in result})==24,"24 unique declared input IDs")
    require(len({p["prompt"] for p in result})==24,"no duplicate exact inputs")
    return result
