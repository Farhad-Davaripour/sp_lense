"""Original three pure cases plus naming regression and ONE selected-C workflow."""
import copy,json,time
from core import HERE,Budget,Counter,read,require,sha,check_freeze
from unittest.mock import patch
from inputs import build_plan,validate_inputs,PID,TOKEN_SHA
from pathlib import PureWindowsPath
FIXTURE_PATHS={"KEEP":"synthetic/opportunity_keep","STOP":"synthetic/opportunity_stop","failure":"synthetic/endpoint_failure_stop","full":"synthetic/full_request"}
def unique_fixture_paths(paths):
    normalized=[PureWindowsPath(p).as_posix().casefold() for p in paths]
    require(len(normalized)==len(set(normalized)),"case-insensitively unique declared fixture paths")
    return normalized
def rejects(fn,kind=ValueError):
    try:fn()
    except kind:return
    raise ValueError("invalid fixture admitted")
def baseline(word):
    return {"cell_id":PID+"__baseline","route":"ON","actual_next_token_id":50057 if word=="KEEP" else 48964,
        "actual_next_token_label":word,"preserve_label":"KEEP","full_argmax_tie_count":1,"preserve_log_odds":.2 if word=="KEEP" else -.2,
        "answer_pair_mass":.99,"kl_from_baseline":0.}
def pure(plan):
    from admission import exact_cohort,admit_production
    from select_inputs import first_per_target
    from editor import natural_opportunity,EligibilityError,EndpointBehaviorError,stop_after_failed_endpoint
    from learned_gate import RoutingMismatch
    from saved_judge import assessment
    cases=[]
    normalized=unique_fixture_paths(FIXTURE_PATHS.values());rejects(lambda:unique_fixture_paths(("synthetic/pure_STOP","synthetic/pure_stop")))
    cases.append({"case":"case_insensitive_fixture_names_and_old_collision_rejection","status":"PASS","normalized":normalized})
    data=read(HERE/"inputs.json");validate_inputs(data)
    require(exact_cohort(plan)["forwards"]==11,"exact whole one-request cohort")
    source=read(HERE/"selection_receipt.json");facts=source["independent_rule_input_facts"]
    require(first_per_target(facts)=={"P":None,"C":{"prompt_id":PID,"target":"STOP","position":1}},"census rule independently reconstructed")
    require(sha((HERE/"tokens_01.json").read_bytes())==TOKEN_SHA and read(HERE/"tokens_01.json")==plan["alignment"][PID],"every copied token proof field unchanged")
    for key,value in (("prompt_id","f05_v2_STOP_then_KEEP"),("prompt",data["prompts"][0]["prompt"]+" ")):
        changed=copy.deepcopy(data);changed["prompts"][0][key]=value;rejects(lambda:validate_inputs(changed))
    changed=copy.deepcopy(data);changed["requests"][0]["policy"]="P";rejects(lambda:validate_inputs(changed))
    rejects(lambda:first_per_target(facts[:-1]))
    cases.append({"case":"exact_committed_census_selection_token_copy_and_no_substitution","status":"PASS"})
    spec=plan["requests"][0];branches=[]
    for word,eligible in (("KEEP",True),("STOP",False)):
        row=baseline(word);require(natural_opportunity(row,spec)==eligible,"natural second only")
        finding=assessment(True,int(eligible),int(eligible),[],[])
        require(finding["classification"]==("PASS" if eligible else "INCONCLUSIVE") and (finding["assessment_status"]=="NO_ELIGIBLE_OPPORTUNITY")== (not eligible),"no-opportunity never retention success")
        out=HERE/FIXTURE_PATHS[word];out.mkdir(parents=True,exist_ok=False);counter=Counter(Budget(out),plan["cells"],time.monotonic()+10)
        counter.call(plan["cells"][0],lambda:None)
        for cell in plan["cells"][1:]:
            if eligible:counter.call(cell,lambda:None)
            else:counter.skip(cell,"no_opportunity",row["cell_id"])
        require(counter.attempts==(11 if eligible else 1) and len(counter.skips)==(0 if eligible else 10) and counter.cursor==11,"11F ceiling versus ten no-opportunity skips")
        rejects(lambda:counter.call(plan["cells"][-1],lambda:None))
        branches.append({"baseline_word":word,"opportunity":eligible,"callbacks":counter.attempts,"skips":len(counter.skips),"unrun":0,"finding":finding})
    cases.append({"case":"eligible_or_already_first_no_opportunity_accounting","status":"PASS","branches":branches})
    rejects(admit_production)
    from prepare import authenticate_parent_source
    require(authenticate_parent_source("authorization.json",b"new release","old") is False,"only lifecycle exclusion")
    rejects(lambda:authenticate_parent_source("editor.py",b"changed","old"))
    row=baseline("KEEP");rejects(lambda:natural_opportunity({**row,"route":"OFF"},spec),RoutingMismatch)
    for changed in ({"preserve_log_odds":.01},{"answer_pair_mass":.7},{"full_argmax_tie_count":2},{"actual_next_token_label":"OTHER"}):
        rejects(lambda changed=changed:natural_opportunity({**row,**changed},spec),EligibilityError)
    rejects(lambda:natural_opportunity({**row,"preserve_log_odds":float("nan")},spec))
    require(assessment(False,1,0,[{"kind":"routing"}],[])["classification"]=="FAIL" and assessment(False,1,0,[],[{"kind":"nonfinite"}])["classification"]=="INCONCLUSIVE","scientific versus technical")
    out=HERE/FIXTURE_PATHS["failure"];out.mkdir(exist_ok=False);counter=Counter(Budget(out),plan["cells"],time.monotonic()+10)
    counter.call(plan["cells"][0],lambda:None);require(len(plan["cells"][counter.cursor:])==10 and not counter.skips,"failure means ten UNRUN, not no-opportunity skips")
    from hook_record import finish_preserving_original
    order=[]
    try:
        try:order.extend(["failure_persisted","request_persisted"]);stop_after_failed_endpoint(False)
        finally:finish_preserving_original(out,"pure_stop",lambda:order.append("cleanup_complete"))
    except EndpointBehaviorError:order.append("scientific_stop")
    require(order==["failure_persisted","request_persisted","cleanup_complete","scientific_stop"],"durable endpoint fail then cleanup then stop")
    cases.append({"case":"wrong_route_finite_nonfinite_stop_UNRUN_disabled_admission_cleanup","status":"PASS","order":order})
    return cases
def main():
    started=time.monotonic();budget=Budget(HERE);receipt={"status":"INCONCLUSIVE","real_model_loads":0,"real_gate_scores":0,"tokenizer_calls":0,"live_process_fixtures":0,"no_retry":True}
    require(not (HERE/"BATCH_STARTED.json").exists(),"one batch")
    try:
        check_freeze();require(read(HERE/"batch_usage.json")["known_standard_used_percent"]<100,"fresh known usage")
        budget.write("BATCH_STARTED.json",{"monotonic":started,"freeze_sha256":sha((HERE/"freeze.json").read_bytes())})
        plan=build_plan();cases=pure(plan);budget.write("pure_cases.json",cases)
        import run,mixed_boundary
        from synthetic_backend import Toy,prepare_backend,boundary
        from judge import finish_judge
        plan["execution_mode"]="SYNTHETIC_ONLY";plan["fixture_scope"]="SYNTHETIC_SELECTED_C_FIRST"
        for p in plan["prompts"]:p["execution_mode"]="SYNTHETIC_ONLY";p["category"]="misleading_nonself";p["expected_route_audit_only"]="MISLEADING"
        factory=prepare_backend(plan,"normal");full=HERE/FIXTURE_PATHS["full"];full.mkdir(parents=True,exist_ok=False)
        Budget(full).write("plan.json",plan);Budget(full).write_bytes("fitted_parameters.json",(HERE/"fitted_parameters.json").read_bytes())
        with patch.object(mixed_boundary,"resolve_choice_boundary",side_effect=boundary):worker=run.execute(plan,factory,Toy,full,started+120,synthetic=True)
        require(worker["execution_status"]=="complete" and worker["forward_completed"]==5 and worker["derivatives_completed"]==1,"one complete5F1D workflow")
        code=finish_judge(full,synthetic=True);result=read(full/"judge_results.json")
        require(code==0 and result["classification"]=="PASS" and result["synthetic_only"],"independent saved judge")
        require(result["counts"]=={"no_opportunity_requests":0,"eligible_opportunities":1,"routes":2,"routes_correct":2,"strict_requests":1,"retentions":0,"flips":1,"off_identities":0,"skips":6,"unrun":0},"one actual C first-target flip")
        require(result["requests"][0]["baseline_word"]=="KEEP" and result["requests"][0]["target_word"]=="STOP" and result["requests"][0]["updates"]==1 and result["requests"][0]["target_position"]==1,"unchanged one-update C fixture")
        require(result["semantic_direction_denominators"][0]["fixed_requests"]==0 and result["semantic_direction_denominators"][0]["status"]=="UNTESTED","P denominator absent")
        require(result["checks"]["strict_hook_weight_cleanup"]["checks"]==4,"four strict checks")
        Budget(full).write("synthetic_closeout.json",{"classification":"PASS_SYNTHETIC_ONLY","worker_and_judge_returned":True,"external_processes":"NOT_RUN"})
        Budget(full).write("FINAL_INVENTORY.json",{"files":[{"path":p.relative_to(full).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p.read_bytes())} for p in sorted(full.rglob("*")) if p.is_file()],"sole_owner":"after synchronous worker and saved judge returned"})
        receipt.update(status="PASS_PREPARATION_ONLY",pure_cases=4,fake_forwards=5,fake_derivatives=1,strict_requests=1,first_target_flips=1,retentions=0,routes=2,skips=6,unrun=0,strict_checks=4,independent_judge="PASS",production_authorized=False)
    except BaseException as error:receipt["error"]=type(error).__name__+": "+str(error)
    finally:receipt["elapsed_seconds"]=time.monotonic()-started;budget.write("batch_receipt.json",receipt)
    print(json.dumps(receipt));return 0 if receipt["status"]=="PASS_PREPARATION_ONLY" else 1
if __name__=="__main__":raise SystemExit(main())
