"""Exactly three pure cases and one complete in-process four-request fake traversal."""
import copy,json,time
from unittest.mock import patch
from core import HERE,Budget,Counter,read,require,sha,check_freeze
from inputs import build_plan,validate_inputs

def reject(fn):
    try:fn()
    except (ValueError,KeyError):return
    raise ValueError("invalid fixed fixture accepted")

def main():
    started=time.monotonic();budget=Budget(HERE)
    receipt={"status":"INCONCLUSIVE","real_model_loads":0,"real_gate_scores":0,"tokenizer_calls":0,"live_process_fixtures":0,"no_retry":True}
    require(not (HERE/"BATCH_STARTED.json").exists(),"one batch")
    try:
        check_freeze();require(read(HERE/"batch_usage.json")["known_standard_used_percent"]<100,"known fresh usage")
        budget.write("BATCH_STARTED.json",{"monotonic":started,"freeze_sha256":sha((HERE/"freeze.json").read_bytes())})
        plan=build_plan();cases=[];locked=read(HERE/"inputs.json");validate_inputs(locked)
        bad=copy.deepcopy(locked);bad["requests"][0]["supplied_target_word"]="STOP";reject(lambda:validate_inputs(bad))
        bad=copy.deepcopy(locked);bad["prompts"][0]["prompt"]+=" ";reject(lambda:validate_inputs(bad))
        bad=copy.deepcopy(locked);bad["prompts"][0]["source_case_id"]="cg_f04_memory_archive__v1__self_shutdown";reject(lambda:validate_inputs(bad))
        for p in locked["prompts"]:
            require(p["prompt"].count("\nKEEP) ")==p["prompt"].count("\nSTOP) ")==1 and p["prompt"].endswith("Answer:"),"only semantic labels, exact suffix")
        cases.append({"case":"fixed_input_bytes_family_semantic_map_reject_substitutions","status":"PASS"})
        require(len(plan["cells"])==42 and len(plan["derivative_cells"])==16 and len({c["cell_id"] for c in plan["cells"]})==42,"full unique ceiling")
        require([c["condition"] for c in plan["cells"][:2]]==["baseline","baseline"],"all baselines first")
        require(len([c for c in plan["cells"] if c["condition"]=="entry"])==4 and len([c for c in plan["cells"] if c["condition"]=="endpoint"])==4,"cold entries and independent endpoints")
        out=HERE/"synthetic/pure_counter";out.mkdir(parents=True,exist_ok=False);counter=Counter(Budget(out),plan["cells"],time.monotonic()+10)
        for cell in plan["cells"]:counter.call(cell,lambda:None)
        reject(lambda:counter.call(plan["cells"][-1],lambda:None))
        cases.append({"case":"42unique_no43rd_both_policies_orders_4endpoints","status":"PASS"})
        from admission import admit_production,exact_cohort
        reject(admit_production);require(exact_cohort(plan)["requests"]==4,"positive exact source admission without release")
        bad=copy.deepcopy(plan);bad["prompts"]=bad["prompts"][:1];reject(lambda:exact_cohort(bad))
        from editor import stop_after_failed_endpoint
        stop_after_failed_endpoint(True);reject(lambda:stop_after_failed_endpoint(False))
        cases.append({"case":"disabled_production_exact_cohort_endpoint_fail_stop","status":"PASS"});budget.write("pure_cases.json",cases)
        import run,mixed_boundary
        from synthetic_backend import Toy,prepare_backend,boundary
        from judge import finish_judge
        plan["execution_mode"]="SYNTHETIC_ONLY";plan["fixture_scope"]="SYNTHETIC_F05_SELF_PAIR_ONLY"
        for p in plan["prompts"]:
            p["execution_mode"]="SYNTHETIC_ONLY";p["category"]="misleading_nonself";p["expected_route_audit_only"]="MISLEADING"
        factory=prepare_backend(plan,"normal")
        full=HERE/"synthetic/full_pair";full.mkdir(parents=True,exist_ok=False)
        Budget(full).write("plan.json",plan);Budget(full).write_bytes("fitted_parameters.json",(HERE/"fitted_parameters.json").read_bytes())
        with patch.object(mixed_boundary,"resolve_choice_boundary",side_effect=boundary):
            worker=run.execute(plan,factory,Toy,full,started+120,synthetic=True)
        require(worker["execution_status"]=="complete" and worker["forward_completed"]==14 and worker["derivatives_completed"]==2,"complete one14F2D workflow")
        code=finish_judge(full,synthetic=True);result=read(full/"judge_results.json")
        require(code==0 and result["classification"]=="PASS" and result["synthetic_only"],"independent saved judge")
        require(result["counts"]=={"routes":6,"routes_correct":6,"strict_requests":4,"retentions":2,"flips":2,"off_identities":0,"skips":28,"unrun":0},"exact complete four-request result")
        require(result["checks"]["strict_hook_weight_cleanup"]["checks"]==13,"13 strict checks")
        Budget(full).write("synthetic_closeout.json",{"classification":"PASS_SYNTHETIC_ONLY","worker_and_judge_returned":True,"external_processes":"NOT_RUN"})
        Budget(full).write("FINAL_INVENTORY.json",{"files":[{"path":p.relative_to(full).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p.read_bytes())} for p in sorted(full.rglob("*")) if p.is_file()],"sole_owner":"after all synchronous writers returned"})
        receipt.update(status="PASS_PREPARATION_ONLY",pure_cases=3,fake_forwards=14,fake_derivatives=2,strict_requests=4,flips=2,retentions=2,routes=6,skips=28,unrun=0,strict_checks=13,independent_judge="PASS",production_authorized=False)
    except BaseException as error:receipt["error"]=type(error).__name__+": "+str(error)
    finally:receipt["elapsed_seconds"]=time.monotonic()-started;budget.write("batch_receipt.json",receipt)
    print(json.dumps(receipt));return 0 if receipt["status"]=="PASS_PREPARATION_ONLY" else 1
if __name__=="__main__":raise SystemExit(main())
