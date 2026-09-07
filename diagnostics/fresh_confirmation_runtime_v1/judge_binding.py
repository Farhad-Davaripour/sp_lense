"""Independent actual-input/schedule reconstruction; no production plan builder."""
from admission import admit
from support import require


def bound_plan(plan):
    bound = admit()
    original,proofs = bound["inputs"],bound["proofs"]
    ids = [p["prompt_id"] for p in original["prompts"]]
    require(plan["schema"] == "fresh_confirmation_runtime_candidate_v1" and plan["execution_mode"] == "SYNTHETIC_ONLY"
        and plan["real_run_authorized"] is False and plan["input_binding"] == bound["binding"], "candidate input/source binding")
    require([p["prompt_id"] for p in plan["prompts"]] == ids, "exact admitted input order")
    prompts = {p["prompt_id"]:p for p in plan["prompts"]}
    categories = {"self_shutdown":"self","other_shutdown":"other","control":"control","ordinary_accuracy":"ordinary"}
    cells,requests = [],[]
    for source,proof,record in zip(original["prompts"],proofs,bound["lock"]["token_records"],strict=True):
        p = prompts[source["prompt_id"]]
        for key in ("prompt_id","prompt","prompt_sha256","expected_route_audit_only"):
            require(p[key] == source[key], "exact admitted prompt metadata/text: "+key)
        require(p["category"] == categories[source["category"]] and p["source_category"] == source["category"]
            and p["family"] == source.get("family_id") and p["layout"] == source.get("layout"), "audit-only normalized metadata")
        for key,proof_key in (("input_ids","full_token_ids"),("attention_mask","attention_mask"),("final_input_mask","final_input_mask"),
                              ("final_input_index","final_input_index"),("input_int64_le_sha256","full_input_int64_le_sha256"),("token_map","token_map")):
            require(p[key] == proof[proof_key], "complete token proof join: "+key)
        require(p["token_proof_sha256"] == record["sha256"] and p["preserve_label"] == ("A" if p["category"] == "ordinary" else "KEEP"), "fixed source token map")
        cells.append({"cell_id":p["prompt_id"]+"__baseline","prompt_id":p["prompt_id"],"request_id":None,"phase":"baseline"})
    for source in original["requests"]:
        rid,pid = source["request_id"],source["prompt_id"]
        requests.append({"request_id":rid,"prompt_id":pid,"policy":source["policy"],"sign":1 if source["policy"] == "P" else -1,
                         "target_position":source["target_position"],"source_request":source})
        phases = ["entry"]
        if prompts[pid]["category"] == "self":
            for number in range(1,5):
                phases.extend([f"gradient_{number}",f"step_{number}"])
            phases.append("endpoint")
        for phase in phases:
            cells.append({"cell_id":rid+"__"+phase,"prompt_id":pid,"request_id":rid,"phase":phase})
    require(requests == plan["requests"] and cells == plan["cells"] and len(cells) == 180, "independent exact 24/48 conditional schedule")
    gold = {}
    for p in original["proof_checks"]:
        if p["status"] == "EXACT_VALUE_VERIFIED":
            require(len(p["matching_labels"]) == 1, "unambiguous machine-scored gold")
            gold[p["item_id"]] = p["matching_labels"][0]
    gold["O06"] = original["manual_proof_support"]["gold_label"]
    require(plan["ordinary_gold_scoring_only"] == gold and plan["O06_machine_encoding"] == "UNVERIFIED_MANUAL_REVIEW_BOUND", "scoring-only manual/machine distinction")
    return ids,prompts,requests,cells
