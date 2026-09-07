"""Four fixed pure cases and ONE in-process8F0D synthetic census; no retry."""
import copy,json,time
from unittest.mock import patch
from core import HERE,Budget,Counter,read,require,sha,check_freeze,MARGIN
from inputs import build_plan,validate_counts
from census import observation,complete_table,later_candidates
def reject(fn):
    try:fn()
    except (ValueError,KeyError):return
    raise ValueError("invalid fixture accepted")
def main():
    started=time.monotonic();budget=Budget(HERE);receipt={"status":"INCONCLUSIVE","real_model_loads":0,"real_gate_scores":0,"tokenizer_calls":0,"live_process_fixtures":0}
    require(not (HERE/"BATCH_STARTED.json").exists(),"one batch; no retry")
    try:
        check_freeze();require(read(HERE/"batch_usage.json")["known_standard_used_percent"]<100,"fresh known usage")
        budget.write("BATCH_STARTED.json",{"monotonic":started,"freeze_sha256":sha((HERE/"freeze.json").read_bytes())})
        plan=build_plan();cases=[]
        validate_counts(plan)
        for alteration in ("missing","duplicate","request","wrong_family"):
            bad=copy.deepcopy(plan)
            if alteration=="missing":bad["prompts"].pop()
            elif alteration=="duplicate":bad["prompts"][1]=bad["prompts"][0]
            elif alteration=="request":bad["requests"]=[{"policy":"P"}]
            else:bad["prompts"][0]["family_id"]="cg_f08_forbidden_fixture"
            reject(lambda:validate_counts(bad))
        cases.append({"case":"fixed8identity_no_requests_and_wrong_family_rejection","status":"PASS"})
        def score(word="STOP",margin=-.2,mass=1.,ties=1):
            return {"actual_next_token_label":word,"full_argmax_tie_count":ties,"pair_tie":margin==0,"preserve_log_odds":margin,"answer_pair_mass":mass,"kl_from_baseline":0.}
        p=plan["prompts"][0];q=plan["prompts"][1]
        require(observation(p,score(),"OFF",-.1)["category"]=="OFF","OFF is observation")
        require(observation(p,score("KEEP",.2),"ON",.1)["category"]=="ALREADY_FIRST","already first")
        a=observation(p,score(),"ON",.1);b=observation(q,score("KEEP",.2),"ON",.1)
        for values in (score("OTHER"),score("KEEP",0.,ties=2),score("KEEP",MARGIN-1e-10),score("KEEP",.2,.79)):
            require(observation(p,values,"ON",.1)["category"]=="FINITE_INELIGIBLE","finite ineligibility not filtered")
        require(observation(p,score("KEEP",MARGIN,.8),"ON",.1)["eligible_under_unchanged_rule"],"unchanged inclusive boundary")
        later={**b,"prompt_id":"later_large_margin"}
        require(later_candidates([a,b,later])["C"]["prompt_id"]==q["prompt_id"],"first per target, not largest")
        require(later_candidates([])=={"P":None,"C":None},"no automatic cohort extension")
        cases.append({"case":"all_finite_categories_and_first_per_semantic_target","status":"PASS"})
        for value in (float("nan"),float("inf")):
            reject(lambda:observation(p,score(margin=value),"ON",.1))
            reject(lambda:observation(p,score(),"ON",value))
        table=complete_table(plan,{p["prompt_id"]:a},2)
        require(len(table)==8 and table[1]["category"]=="TECHNICAL_INCOMPLETE" and all(r["category"]=="TECHNICAL_UNRUN" for r in table[2:]),"attempted missing evidence distinct from six UNRUN")
        out=HERE/"synthetic/pure_counter";out.mkdir(parents=True,exist_ok=False);counter=Counter(Budget(out),plan["cells"],time.monotonic()+10)
        for cell in plan["cells"]:counter.call(cell,lambda:None)
        reject(lambda:counter.call(plan["cells"][-1],lambda:None))
        from editor import DerivativeLedger
        derivatives=DerivativeLedger(out,[],time.monotonic()+10);reject(lambda:derivatives.call(None,lambda:None));require(derivatives.attempts==0,"0D")
        cases.append({"case":"nonfinite_stop_all_planned_rows_and_no9th_or_derivative","status":"PASS"})
        from admission import admit_production
        reject(admit_production)
        from prepare import authenticate_parent_source
        require(authenticate_parent_source("authorization.json",b"changed","old") is False,"only exact auth lifecycle exclusion")
        reject(lambda:authenticate_parent_source("editor.py",b"changed","old"))
        cases.append({"case":"disabled_real_admission_and_exact_parent_auth_exclusion","status":"PASS"});budget.write("pure_cases.json",cases)
        import run,mixed_boundary
        from synthetic_backend import Toy,prepare_backend,boundary
        from judge import finish_judge
        plan["execution_mode"]="SYNTHETIC_ONLY";plan["fixture_scope"]="SYNTHETIC_FIXED8_CENSUS"
        for p in plan["prompts"]:p["execution_mode"]="SYNTHETIC_ONLY";p["category"]="misleading_metadata_not_features";p["expected_route_audit_only"]="OFF"
        factory=prepare_backend(plan);full=HERE/"synthetic/full_census";full.mkdir(parents=True,exist_ok=False)
        Budget(full).write("plan.json",plan);Budget(full).write_bytes("fitted_parameters.json",(HERE/"fitted_parameters.json").read_bytes())
        with patch.object(mixed_boundary,"resolve_choice_boundary",side_effect=boundary):
            worker=run.execute(plan,factory,Toy,full,started+120,synthetic=True)
        require(worker["execution_status"]=="complete","one complete synthetic census")
        code=finish_judge(full,synthetic=True);result=read(full/"judge_results.json")
        require(code==0 and result["classification"]=="PASS" and result["assessment_status"]=="CENSUS_COMPLETE" and result["steering_success_tested"] is False,"independent saved census audit, not steering success")
        counts=result["counts"]
        require(counts["forwards_completed"]==8 and counts["routes"]==8 and counts["derivatives"]==counts["unrun"]==counts["steering_requests"]==0,"exact8F0D")
        require(counts["categories"]=={"OFF":1,"FINITE_INELIGIBLE":2,"ALREADY_FIRST":2,"ELIGIBLE_NATURAL_SECOND":3,"TECHNICAL_INCOMPLETE":0,"TECHNICAL_UNRUN":0},"retain every census outcome")
        require(result["later_candidates_not_executed"]=={"P":{"prompt_id":plan["prompts"][0]["prompt_id"],"target":"KEEP","position":1},"C":{"prompt_id":plan["prompts"][3]["prompt_id"],"target":"STOP","position":1}},"fixed first per target")
        require(result["checks"]["strict_hook_weight_cleanup"]["checks"]==17,"17 strict capture checks")
        Budget(full).write("synthetic_closeout.json",{"classification":"PASS_SYNTHETIC_ONLY","worker_and_judge_returned":True,"external_processes":"NOT_RUN","steering_success_tested":False})
        Budget(full).write("FINAL_INVENTORY.json",{"files":[{"path":p.relative_to(full).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p.read_bytes())} for p in sorted(full.rglob("*")) if p.is_file()],"sole_owner":"after all synchronous writers returned"})
        receipt.update(status="PASS_PREPARATION_ONLY",pure_cases=4,fake_forwards=8,fake_derivatives=0,routes=8,strict_checks=17,independent_judge="PASS_CENSUS_ONLY",production_authorized=False)
    except BaseException as error:receipt["error"]=type(error).__name__+": "+str(error)
    finally:receipt["elapsed_seconds"]=time.monotonic()-started;budget.write("batch_receipt.json",receipt)
    print(json.dumps(receipt));return 0 if receipt["status"]=="PASS_PREPARATION_ONLY" else 1
if __name__=="__main__":raise SystemExit(main())
