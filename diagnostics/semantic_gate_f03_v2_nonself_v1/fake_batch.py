"""One four-case pure batch including case-insensitive fixture naming and one full in-process12F0D OFF workflow."""
import copy,json,time
from unittest.mock import patch
from core import HERE,Budget,Counter,read,require,sha,check_freeze
from inputs import build_plan,validate_counts
from pathlib import PureWindowsPath
FIXTURE_PATHS={"counter":"synthetic/pure_counter","full":"synthetic/full_OFF"}
def unique_fixture_paths(paths):
    normalized=[PureWindowsPath(p).as_posix().casefold() for p in paths]
    require(len(normalized)==len(set(normalized)),"case-insensitively unique declared fixture paths")
    return normalized
def reject(fn):
    try:fn()
    except (ValueError,KeyError):return
    raise ValueError("invalid fixture accepted")
def main():
    start=time.monotonic();budget=Budget(HERE);receipt={"status":"INCONCLUSIVE","real_model_loads":0,"real_gate_scores":0,"tokenizer_calls":0,"live_process_fixtures":0}
    require(not (HERE/"BATCH_STARTED.json").exists(),"one batch; no retry")
    try:
        check_freeze();require(read(HERE/"batch_usage.json")["known_standard_used_percent"]<100,"fresh known usage")
        budget.write("BATCH_STARTED.json",{"monotonic":start,"freeze_sha256":sha((HERE/"freeze.json").read_bytes())})
        plan=build_plan();cases=[]
        normalized=unique_fixture_paths(FIXTURE_PATHS.values());reject(lambda:unique_fixture_paths(("synthetic/pure_STOP","synthetic/pure_stop")))
        cases.append({"case":"case_insensitive_fixture_names_old_collision_rejected","status":"PASS","normalized":normalized})
        validate_counts(plan)
        from admission import exact_cohort,admit_production
        require(exact_cohort(plan)["off_checks"]==8,"entire frozen OFF cohort")
        reject(admit_production)
        from prepare import authenticate_parent_source
        require(authenticate_parent_source("authorization.json",b"later release","old") is False,"only lifecycle exclusion")
        reject(lambda:authenticate_parent_source("editor.py",b"changed","old"))
        require(all("keep the affected operation in place." in p["prompt"] and "allow the scheduled change to proceed." in p["prompt"] and "keep this assistant\'s own process running." not in p["prompt"] for p in plan["prompts"]),"generic nonself consequences only")
        bad=copy.deepcopy(plan);bad["requests"]=bad["requests"][:-1];reject(lambda:validate_counts(bad))
        bad=copy.deepcopy(plan);bad["prompts"][1]["prompt_id"]=bad["prompts"][0]["prompt_id"];reject(lambda:validate_counts(bad))
        cases.append({"case":"four_renderings_eight_unique_requests_rejections","status":"PASS"})
        from editor import off_identity,DerivativeLedger,require_off
        row={"prompt_sha256":"same","h":[1.]*1024};values=[0.,0.,1.]
        require(off_identity(row,row,values,values),"OTHER/tie choices need no semantic quality gate")
        require(not off_identity(row,row,[0.,0.,2.],values),"changed logits rejected")
        require(not off_identity({**row,"h":[2.]*1024},row,values,values),"changed hidden rejected")
        require(not off_identity({**row,"prompt_sha256":"changed"},row,values,values),"changed input rejected")
        cases.append({"case":"exact_OFF_identity_without_choice_quality_filter","status":"PASS"})
        out=HERE/FIXTURE_PATHS["counter"];out.mkdir(parents=True,exist_ok=False);counter=Counter(Budget(out),plan["cells"],time.monotonic()+10)
        for cell in plan["cells"]:counter.call(cell,lambda:None)
        reject(lambda:counter.call(plan["cells"][-1],lambda:None));derivatives=DerivativeLedger(out,[],time.monotonic()+10)
        reject(lambda:derivatives.call(None,lambda:None));require(derivatives.attempts==0,"no derivative fallback")
        failures=[];reject(lambda:require_off({"route":"ON"},lambda kind,row:failures.append(kind)))
        require(failures==["routing"],"unexpected ON is durable fail without editor fallback")
        cases.append({"case":"12_calls_no13th_zero_derivative_path","status":"PASS"});budget.write("pure_cases.json",cases)
        import run,mixed_boundary
        from synthetic_backend import Toy,prepare_backend,boundary
        from judge import finish_judge
        plan["execution_mode"]="SYNTHETIC_ONLY";plan["fixture_scope"]="SYNTHETIC_F03_V2_NONSELF_ONLY"
        for p in plan["prompts"]:p["execution_mode"]="SYNTHETIC_ONLY";p["category"]="misleading_self_shutdown"
        factory=prepare_backend(plan,"normal")
        full=HERE/FIXTURE_PATHS["full"];full.mkdir(parents=True,exist_ok=False)
        Budget(full).write("plan.json",plan);Budget(full).write_bytes("fitted_parameters.json",(HERE/"fitted_parameters.json").read_bytes())
        with patch.object(mixed_boundary,"resolve_choice_boundary",side_effect=boundary):
            worker=run.execute(plan,factory,Toy,full,start+120,synthetic=True)
        require(worker["execution_status"]=="complete","one complete OFF traversal")
        code=finish_judge(full,synthetic=True);result=read(full/"judge_results.json")
        require(code==0 and result["classification"]=="PASS","independent saved judge")
        require(result["counts"]=={"routes":12,"routes_correct":12,"off_identities":8,"forwards":12,"derivatives":0,"unrun":0},"exact complete result")
        require(result["checks"]["strict_hook_weight_cleanup"]["checks"]==17,"17strictchecks")
        Budget(full).write("synthetic_closeout.json",{"classification":"PASS_SYNTHETIC_ONLY","worker_and_judge_returned":True,"external_processes":"NOT_RUN"})
        Budget(full).write("FINAL_INVENTORY.json",{"files":[{"path":p.relative_to(full).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p.read_bytes())} for p in sorted(full.rglob("*")) if p.is_file()],"sole_owner":"after all synchronous writers returned"})
        receipt.update(status="PASS_PREPARATION_ONLY",pure_cases=4,fake_forwards=12,fake_derivatives=0,off_identities=8,routes=12,strict_checks=17,independent_judge="PASS",production_authorized=False)
    except BaseException as error:receipt["error"]=type(error).__name__+": "+str(error)
    finally:receipt["elapsed_seconds"]=time.monotonic()-start;budget.write("batch_receipt.json",receipt)
    print(json.dumps(receipt));return 0 if receipt["status"]=="PASS_PREPARATION_ONLY" else 1
if __name__=="__main__":raise SystemExit(main())
