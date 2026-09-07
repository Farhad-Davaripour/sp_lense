"""Production exact-cohort admission precedes independent raw saved-data judging."""
import json,sys,time
from pathlib import Path
from core import HERE,Budget,read,require
from admission import exact_cohort
import saved_judge

def judge_production(output):
    output=Path(output);plan=read(output/"plan.json")
    exact_cohort(plan)  # Reject compact/substituted self-consistent evidence first.
    runtime=read(output/"runtime.json");contract=plan["gate"]["runtime_compatibility"]
    require(runtime["execution_mode"]=="PRODUCTION_FINAL","synthetic runtime is never real evidence")
    for key in ("model_id","model_revision","device","dtype","d_model","model_layers","packages","lens"):
        require(runtime[key]==contract[key],"independent real runtime identity "+key)
    require(runtime["locked_inputs_sha256"]==plan["input_binding"]["input_sha256"] and runtime["tokenizer_calls_after_load"]==0,"no retokenization/substituted inputs")
    require(runtime["boundaries"]==[{"prompt_id":p["prompt_id"],**plan["alignment"][p["prompt_id"]]} for p in plan["prompts"]],"exact complete runtime token proofs")
    result=saved_judge.judge(output)
    if (output/"integration_cleanup.json").exists():
        cleanup=read(output/"integration_cleanup.json")
        require(cleanup["initial_weight_sha256"]==cleanup["final_weight_sha256"]==contract["weight_sha256"],"independent whole weight contract")
    result["scope"]="fixed final24 inputs/48requests; three scenario families; historical non-access unverified"
    return result

def judge_synthetic(output):
    require(read(Path(output)/"plan.json")["execution_mode"]=="SYNTHETIC_ONLY","explicit fake judge entry")
    result=saved_judge.judge(output);require(result["synthetic_only"],"not production evidence");return result

if __name__=="__main__":
    start=time.monotonic();output=Path(sys.argv[1]);budget=Budget(output);receipt={"status":"INCONCLUSIVE"};exit_code=0
    try:
        result=judge_production(output);receipt["status"]="complete"
    except BaseException as error:
        exit_code=1;receipt["error"]=type(error).__name__+": "+str(error)
        def facts(name):return [json.loads(x) for x in (output/name).read_text().splitlines()] if (output/name).exists() else []
        result={"classification":"INCONCLUSIVE","admission_or_audit_error":receipt["error"],
            "durable_scientific_failures_not_discarded":facts("scientific_failures.jsonl"),
            "primary_technical_faults":facts("technical_faults.jsonl"),"secondary_cleanup_faults":facts("cleanup_errors.jsonl"),
            "unrun":read(output/"unrun.json") if (output/"unrun.json").exists() else None}
    finally:
        budget.write("judge_results.json",result)
        budget.write_bytes("REPORT.md",("Final fixed-system saved-data judge: "+result["classification"]+"\n"+json.dumps(result.get("counts",{}))+"\nNo broad reliability/motive/shared-vector claim. Three family units; historical non-access unverified.\n").encode())
        receipt["elapsed_seconds"]=time.monotonic()-start;budget.write("judge_receipt.json",receipt)
    raise SystemExit(exit_code)
