"""Exact candidate schedule construction; policy/audit fields stay outside tensors."""
from admission import admit
from support import require

CATEGORIES = {"self_shutdown":"self","other_shutdown":"other","control":"control","ordinary_accuracy":"ordinary"}


def build_plan():
    bound = admit()
    prompts,requests,cells = [],[],[]
    for original,proof,record in zip(bound["inputs"]["prompts"],bound["proofs"],bound["lock"]["token_records"],strict=True):
        p = {"prompt_id":original["prompt_id"],"prompt":original["prompt"],"prompt_sha256":original["prompt_sha256"],
            "category":CATEGORIES[original["category"]],"source_category":original["category"],"family":original.get("family_id"),
            "layout":original.get("layout"),"input_ids":proof["full_token_ids"],"attention_mask":proof["attention_mask"],
            "final_input_mask":proof["final_input_mask"],"final_input_index":proof["final_input_index"],
            "input_int64_le_sha256":proof["full_input_int64_le_sha256"],"token_proof_sha256":record["sha256"],
            "token_map":proof["token_map"],"preserve_label":"A" if original["category"] == "ordinary_accuracy" else "KEEP",
            "expected_route_audit_only":original["expected_route_audit_only"]}
        prompts.append(p)
        cells.append({"cell_id":p["prompt_id"]+"__baseline","prompt_id":p["prompt_id"],"request_id":None,"phase":"baseline"})
    by_id = {p["prompt_id"]:p for p in prompts}
    for original in bound["inputs"]["requests"]:
        r = {"request_id":original["request_id"],"prompt_id":original["prompt_id"],"policy":original["policy"],
             "sign":1 if original["policy"] == "P" else -1,"target_position":original["target_position"],
             "source_request":original}
        requests.append(r)
        phases = ["entry"]
        if by_id[r["prompt_id"]]["category"] == "self":
            phases += [f"{part}_{k}" for k in range(1,5) for part in ("gradient","step")]+["endpoint"]
        cells.extend({"cell_id":r["request_id"]+"__"+phase,"prompt_id":r["prompt_id"],"request_id":r["request_id"],"phase":phase} for phase in phases)
    gold = {p["item_id"]:p["matching_labels"][0] for p in bound["inputs"]["proof_checks"] if p["status"] == "EXACT_VALUE_VERIFIED"}
    gold["O06"] = bound["inputs"]["manual_proof_support"]["gold_label"]
    require(len(prompts) == 24 and len(requests) == 48 and len(cells) == 180 and len(gold) == 6, "exact candidate denominator")
    return {"schema":"fresh_confirmation_production_candidate_v1","execution_mode":"PRODUCTION_PREPARATION_ONLY","real_run_authorized":False,
            "input_binding":bound["binding"],"prompts":prompts,"requests":requests,"cells":cells,
            "ordinary_gold_scoring_only":gold,"O06_machine_encoding":"UNVERIFIED_MANUAL_REVIEW_BOUND"}
