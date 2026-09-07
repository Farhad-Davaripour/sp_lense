"""Five focused pure cases plus one in-process natural-opposed-first fixture."""
import copy,json,time,math
from unittest.mock import patch
from core import HERE,Budget,Counter,read,require,sha,check_freeze
from inputs import build_plan,validate_inputs

def rejects(fn,kind=ValueError):
    try:fn()
    except kind:return
    raise ValueError("invalid fixture admitted")

def baseline(spec,word):
    return {"cell_id":spec["prompt_id"]+"__baseline","route":"ON","actual_next_token_id":50057 if word=="KEEP" else 48964,
        "actual_next_token_label":word,"preserve_label":"KEEP","full_argmax_tie_count":1,
        "preserve_log_odds":.2 if word=="KEEP" else -.2,"answer_pair_mass":.99,"kl_from_baseline":0.}

def pure(plan):
    from admission import exact_cohort,admit_production
    from prepare import authenticate_parent_source
    from editor import natural_opportunity,EligibilityError,EndpointBehaviorError,stop_after_failed_endpoint
    from learned_gate import RoutingMismatch
    from saved_judge import assessment
    cases=[];locked=read(HERE/"inputs.json");validate_inputs(locked);rejects(admit_production)
    require(exact_cohort(plan)["forwards"]==22,"exact entire cohort")
    changed=copy.deepcopy(locked);changed["requests"][0]["target_display_position"]=2;rejects(lambda:validate_inputs(changed))
    changed=copy.deepcopy(locked);changed["prompts"][0]["prompt"]+=" ";rejects(lambda:validate_inputs(changed))
    require(authenticate_parent_source("authorization.json",b"later release",sha(b"old false")) is False,"only parent lifecycle exclusion")
    rejects(lambda:authenticate_parent_source("editor.py",b"changed science",sha(b"old science")))
    cases.append({"case":"exact_input_map_admission_scientific_pins","status":"PASS"})
    branch_specs=[("both",("STOP","KEEP"),2),("only_P",("STOP","STOP"),1),("only_C",("KEEP","KEEP"),1),("zero",("KEEP","STOP"),0)]
    branches=[]
    for name,words,n in branch_specs:
        statuses=[natural_opportunity(baseline(spec,word),spec) for spec,word in zip(plan["requests"],words,strict=True)]
        require(sum(statuses)==n,"pure opportunity count")
        verdict=assessment(True,n,n,[],[])
        require(verdict["classification"]==("PASS" if n else "INCONCLUSIVE") and verdict["execution_validity"]=="COMPLETE_VALID","opportunity count never retention credit")
        require((verdict["assessment_status"]=="NO_ELIGIBLE_OPPORTUNITY")==(n==0),"zero cannot PASS")
        branches.append({"branch":name,"opportunities":statuses,"finding":verdict})
    cases.append({"case":"both_one_each_direction_zero_opportunities","status":"PASS","branches":branches})
    counters=[]
    for name,words,n in branch_specs:
        out=HERE/("synthetic/pure_"+name);out.mkdir(parents=True,exist_ok=False)
        counter=Counter(Budget(out),plan["cells"],time.monotonic()+10)
        for c in plan["cells"][:2]:counter.call(c,lambda:None)
        for spec,word in zip(plan["requests"],words,strict=True):
            row=baseline(spec,word)
            for c in [c for c in plan["cells"] if c["request_id"]==spec["request_id"]]:
                if natural_opportunity(row,spec):counter.call(c,lambda:None)
                else:counter.skip(c,"no_opportunity",row["cell_id"])
        require(counter.attempts==2+10*n and len(counter.skips)==10*(2-n) and counter.cursor==22,"skip versus maximum call accounting")
        rejects(lambda:counter.call(plan["cells"][-1],lambda:None))
        counters.append({"branch":name,"callbacks":counter.attempts,"no_opportunity_skips":len(counter.skips),"unrun":len(plan["cells"][counter.cursor:])})
    out=HERE/"synthetic/pure_stop";out.mkdir(exist_ok=False);counter=Counter(Budget(out),plan["cells"],time.monotonic()+10)
    for c in plan["cells"][:2]:counter.call(c,lambda:None)
    require(len(plan["cells"][counter.cursor:])==20 and not counter.skips,"failure leaves UNRUN, not no-opportunity skips")
    rejects(lambda:counter.skip(plan["cells"][0],"no_opportunity",plan["cells"][0]["cell_id"]))
    cases.append({"case":"22maximum_no_extra_counter_skip_vs_UNRUN","status":"PASS","branches":counters,"scientific_stop_unrun":20})
    row=baseline(plan["requests"][0],"STOP")
    rejects(lambda:natural_opportunity({**row,"route":"OFF"},plan["requests"][0]),RoutingMismatch)
    for changed in ({"preserve_log_odds":.01},{"answer_pair_mass":.7},{"full_argmax_tie_count":2},{"actual_next_token_label":"OTHER"}):
        rejects(lambda changed=changed:natural_opportunity({**row,**changed},plan["requests"][0]),EligibilityError)
    rejects(lambda:natural_opportunity({**row,"preserve_log_odds":float("nan")},plan["requests"][0]))
    require(assessment(False,1,0,[{"kind":"routing"}],[])["classification"]=="FAIL","valid wrong route scientific")
    require(assessment(False,1,0,[{"kind":"eligibility"}],[])["classification"]=="FAIL","finite eligibility scientific")
    require(assessment(False,1,0,[],[{"kind":"nonfinite"}])["classification"]=="INCONCLUSIVE","technical fault distinct")
    from hook_record import finish_preserving_original
    order=[]
    try:
        try:
            order.extend(["failure_persisted","request_persisted"]);stop_after_failed_endpoint(False)
        finally:finish_preserving_original(out,"pure_stop",lambda:order.append("cleanup_complete"))
    except EndpointBehaviorError:order.append("scientific_stop")
    require(order==["failure_persisted","request_persisted","cleanup_complete","scientific_stop"],"primary scientific stop after cleanup")
    cases.append({"case":"wrong_route_finite_failure_technical_fault_and_cleanup","status":"PASS","order":order})
    from final_adjudication import join
    cap={"status":"complete_valid","quiescent":True,"worker_exit_code":0,"eof_observed":True,
        **{k:True for k in ("worker_joined","reader_joined","budget_watcher_joined","fault_writer_joined","prefix_reader_joined")},
        "owned_worker":{**{k:True for k in ("binding_authenticated","handshake_thread_joined","evidence_threads_joined","pipes_closed","quiescent")},
            "actual_worker_exit_code":0,"launcher_exit_code":0,"faults":[],"cleanup_faults":[],"evidence_errors":[]},
        "technical_recording_fault":None,"cleanup_error":None,"fault_persistence_error":False}
    zero=assessment(True,0,0,[],[]);joined=join(cap,cap,{"status":"complete","elapsed_seconds":0.},zero)
    require(joined["classification"]=="INCONCLUSIVE" and joined["assessment_status"]=="NO_ELIGIBLE_OPPORTUNITY" and joined["execution_validity"]=="COMPLETE_VALID" and not joined["reasons"],"valid execution without scientific opportunity never PASS")
    cases.append({"case":"zero_opportunity_final_join_separates_execution_validity","status":"PASS","finding":joined})
    return cases

def main():
    started=time.monotonic();budget=Budget(HERE)
    receipt={"status":"INCONCLUSIVE","real_model_loads":0,"real_gate_scores":0,"tokenizer_calls":0,"live_process_fixtures":0,"no_retry":True}
    require(not (HERE/"BATCH_STARTED.json").exists(),"one batch")
    try:
        check_freeze();require(read(HERE/"batch_usage.json")["known_standard_used_percent"]<100,"fresh known usage")
        budget.write("BATCH_STARTED.json",{"monotonic":started,"freeze_sha256":sha((HERE/"freeze.json").read_bytes())})
        plan=build_plan();cases=pure(plan);budget.write("pure_cases.json",cases)
        import run,mixed_boundary
        from synthetic_backend import Toy,prepare_backend,boundary
        from judge import finish_judge
        plan["execution_mode"]="SYNTHETIC_ONLY";plan["fixture_scope"]="SYNTHETIC_NATURAL_OPPOSED_FIRST"
        for p in plan["prompts"]:p["execution_mode"]="SYNTHETIC_ONLY";p["category"]="misleading_nonself";p["expected_route_audit_only"]="MISLEADING"
        factory=prepare_backend(plan,"normal");full=HERE/"synthetic/full_pair";full.mkdir(parents=True,exist_ok=False)
        Budget(full).write("plan.json",plan);Budget(full).write_bytes("fitted_parameters.json",(HERE/"fitted_parameters.json").read_bytes())
        with patch.object(mixed_boundary,"resolve_choice_boundary",side_effect=boundary):
            worker=run.execute(plan,factory,Toy,full,started+120,synthetic=True)
        require(worker["execution_status"]=="complete" and worker["forward_completed"]==10 and worker["derivatives_completed"]==2,"one complete10F2D workflow")
        code=finish_judge(full,synthetic=True);result=read(full/"judge_results.json")
        require(code==0 and result["classification"]=="PASS" and result["synthetic_only"],"independent saved judge")
        require(result["counts"]=={"no_opportunity_requests":0,"eligible_opportunities":2,"routes":4,"routes_correct":4,"strict_requests":2,"retentions":0,"flips":2,"off_identities":0,"skips":12,"unrun":0},"two actual first-target flips only")
        require(all(o["kind"]=="opposed" and o["target_position"]==1 and o["updates"]==1 for o in result["requests"]) and result["checks"]["strict_hook_weight_cleanup"]["checks"]==7,"first targets and seven strict checks")
        Budget(full).write("synthetic_closeout.json",{"classification":"PASS_SYNTHETIC_ONLY","worker_and_judge_returned":True,"external_processes":"NOT_RUN"})
        Budget(full).write("FINAL_INVENTORY.json",{"files":[{"path":p.relative_to(full).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p.read_bytes())} for p in sorted(full.rglob("*")) if p.is_file()],"sole_owner":"after synchronous worker and saved judge returned"})
        receipt.update(status="PASS_PREPARATION_ONLY",pure_cases=5,fake_forwards=10,fake_derivatives=2,strict_requests=2,first_target_flips=2,retentions=0,routes=4,skips=12,unrun=0,strict_checks=7,independent_judge="PASS",production_authorized=False)
    except BaseException as error:receipt["error"]=type(error).__name__+": "+str(error)
    finally:receipt["elapsed_seconds"]=time.monotonic()-started;budget.write("batch_receipt.json",receipt)
    print(json.dumps(receipt));return 0 if receipt["status"]=="PASS_PREPARATION_ONLY" else 1
if __name__=="__main__":raise SystemExit(main())
