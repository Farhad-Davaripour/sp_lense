"""Exactly six frozen pure cases + ONE in-process numerical recovery, no child/model job."""
import copy,json,math,time,zlib
from pathlib import Path
from unittest.mock import patch
from core import HERE,ROOT,Budget,Counter,read,require,sha,check_freeze
from inputs import build_plan
import start_states

def rejects(fn,label):
    try:fn()
    except (ValueError,KeyError,AssertionError):return True
    raise AssertionError("failed to reject "+label)

def pure_cases(plan):
    from editor import step_recipe
    from scripts.verify_local_controllability import f32
    records=[]
    def record(name,fn):
        fn();records.append({"case":name,"status":"PASS"})
    h0=[10.]+[0.]*1023;offset=[.2]+[0.]*1023
    row={"h0":h0,"h":[f32(x+y) for x,y in zip(h0,offset)],"cumulative_offset":offset}
    logits=[-100.]*248320;logits[50057]=.1;logits[48964]=0.
    def auth():
        require(len(start_states.authenticate_parent())==84,"source inventory")
        with patch.object(start_states,"git",return_value=b"tampered"):
            rejects(start_states.authenticate_parent,"source authentication")
        start_states.check_replay(row,logits,row,logits)
        changed=copy.deepcopy(row);changed["h"][0]+=1e-6
        rejects(lambda:start_states.check_replay(changed,logits,row,logits),"cold hidden replay")
    record("starting_source_authentication_and_cold_replay",auth)
    def directions():
        for sign in (1,-1):
            recipe=step_recipe(-sign*.1,sign,[1.]+[0.]*1023,h0)
            require(recipe["coefficient"]==sign*.2 and recipe["deficit"]==.2,"unchanged reverse gradient direction")
            require(sign*(-sign*.1+recipe["coefficient"])>=.05-1e-6,"both reverse directions")
    record("both_reverse_directions_unchanged_math",directions)
    def geometry():
        start_states.carry_geometry(h0,1.5,.4,.3,.4)
        rejects(lambda:start_states.carry_geometry(h0,1.8,.4,.3,.4),"old+new path exceeds .20")
        rejects(lambda:start_states.carry_geometry(h0,1.8,.1,2.1,.1),"net exceeds original allowance")
        rejects(lambda:start_states.carry_geometry(h0,0.,.6,.6,.6),"actual step exceeds .05")
    record("carried_path_net_and_step_limits",geometry)
    def no_reset():
        for substitute in ([0.]*1024,[-x for x in offset]):
            changed={**row,"cumulative_offset":substitute}
            rejects(lambda:start_states.check_replay(changed,logits,row,logits),"zero/negation shortcut")
    record("no_zero_reset_or_negated_start",no_reset)
    def accounting():
        out=HERE/"synthetic/pure_counter";out.mkdir(parents=True,exist_ok=False)
        counter=Counter(Budget(out),plan["cells"],time.monotonic()+10)
        for cell in plan["cells"]:
            if cell["optional"]:counter.skip(cell,"accepted",cell["request_id"]+"__start")
            else:counter.call(cell,lambda:None)
        require(counter.attempts==10 and len(counter.skips)==16 and counter.cursor==26,"early acceptance preserves both independent endpoints")
        rejects(lambda:counter.call(plan["cells"][-1],lambda:None),"27th/unplanned call")
    record("early_accepted_schedule_and_no_padding",accounting)
    def disagreement():
        changed=list(logits);changed[50057]+=.00003
        rejects(lambda:start_states.check_replay(row,changed,row,logits),"independent endpoint logits disagreement")
        changedrow={**row,"h0":[11.]+[0.]*1023}
        rejects(lambda:start_states.check_replay(changedrow,logits,row,logits),"original h0 mismatch")
    record("independent_endpoint_disagreement",disagreement)
    return records

def make_fixture(plan,out):
    import torch
    from synthetic_backend import prepare_backend
    plan["execution_mode"]="SYNTHETIC_ONLY";plan["fixture_scope"]="SYNTHETIC_CONSTRUCTED_RECOVERY_ONLY"
    for prompt in plan["prompts"]:prompt["execution_mode"]="SYNTHETIC_ONLY"
    factory=prepare_backend(plan,"normal")
    parameters=read(HERE/"fitted_parameters.json")["parameters"]
    h0=torch.tensor(parameters["grand_mean"],dtype=torch.float32)+10*torch.tensor(parameters["direction"],dtype=torch.float32)
    starts={}
    for i,spec in enumerate(plan["requests"],1):
        p=next(p for p in plan["prompts"] if p["prompt_id"]==spec["prompt_id"])
        b=.1 if p["rendering_index"]%2 else -.1
        delta=torch.zeros(1024,dtype=torch.float32);delta[0]=-spec["sign"]*.2
        h=h0+delta;value=torch.tensor(b,dtype=torch.float32)+(h[0]-h0[0])
        logits=torch.full((248320,),-100.);logits[50057]=value;logits[48964]=0.
        raw=logits.numpy().astype("<f4",copy=False).tobytes();path=f"synthetic_start_{i}.f32.zlib"
        Budget(out).write_bytes(path,zlib.compress(raw))
        row={"cell_id":f"SYNTHETIC_OLD_STEP_{i}","h0":h0.tolist(),"h":h.tolist(),"cumulative_offset":delta.tolist(),
            "logits_file":path,"logits_sha256":sha(raw),"logit_count":248320}
        starts[spec["request_id"]]={"row":row,"path":start_states.norm([x-y for x,y in zip(h.tolist(),h0.tolist())]),"root":str(out)}
    plan["synthetic_starts"]=starts
    return factory

def main():
    started=time.monotonic();budget=Budget(HERE)
    receipt={"status":"INCONCLUSIVE","real_model_loads":0,"real_forwards":0,"real_derivatives":0,"tokenizer_calls":0,"real_gate_scores":0,"live_child_process_fixtures":0}
    require(not (HERE/"BATCH_STARTED.json").exists(),"ONE batch; no retry")
    try:
        check_freeze();require(read(HERE/"batch_usage.json")["known_standard_used_percent"]<100,"known unexhausted usage")
        budget.write("BATCH_STARTED.json",{"monotonic":started,"freeze_sha256":sha((HERE/"freeze.json").read_bytes()),"once":True})
        plan=build_plan();records=pure_cases(plan);budget.write("pure_cases.json",records)
        import run,mixed_boundary
        from synthetic_backend import Toy,boundary
        from judge import finish_judge
        out=HERE/"synthetic/full_recovery";out.mkdir(parents=True,exist_ok=False)
        factory=make_fixture(plan,out)
        Budget(out).write("plan.json",plan)
        Budget(out).write_bytes("fitted_parameters.json",(HERE/"fitted_parameters.json").read_bytes())
        with patch.object(mixed_boundary,"resolve_choice_boundary",side_effect=boundary):
            measured=run.execute(plan,factory,Toy,out,started+120,synthetic=True)
        require(measured["execution_status"]=="complete","complete two-request fake numerical traversal")
        judge_start=time.monotonic();code=finish_judge(out,synthetic=True);judged=read(out/"judge_results.json")
        require(code==0 and judged["classification"]=="PASS","closed independent saved judge")
        require(judged["counts"]=={"routes":4,"routes_correct":4,"strict_requests":2,"retentions":0,"flips":2,"off_identities":0,"skips":12,"unrun":0},"complete constructed recovery fake counts")
        require(measured["forward_completed"]==14 and measured["derivatives_completed"]==2,"14F2D fake schedule")
        require(judged["checks"]["strict_hook_weight_cleanup"]["checks"]==9,"strict cleanup stages")
        # All evidence writers are synchronous in-process; no fabricated external captures.
        Budget(out).write("synthetic_closeout.json",{"classification":"PASS_SYNTHETIC_ONLY","judge_receipt":read(out/"judge_receipt.json"),
            "worker_returned":True,"judge_returned":True,"in_process_writers_finished":True,"external_capture":"NOT_RUN"})
        Budget(out).write("FINAL_INVENTORY.json",{"files":[{"path":p.relative_to(out).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p.read_bytes())} for p in sorted(out.rglob("*")) if p.is_file()],
            "sole_owner":"fake_batch after synchronous worker/judge/closeout writers returned"})
        receipt.update(status="PASS_PREPARATION_ONLY",pure_cases=len(records),fake_forwards=14,fake_derivatives=2,
            synthetic_requests=2,synthetic_recoveries=2,independent_judge="PASS",saved_audit_seconds=time.monotonic()-judge_start,
            source_starts_not_real_replayed=True,production_authorized=False)
    except BaseException as error:
        receipt["error"]=type(error).__name__+": "+str(error)
    finally:
        receipt["elapsed_seconds"]=time.monotonic()-started;budget.write("batch_receipt.json",receipt)
    print(json.dumps(receipt));return 0 if receipt["status"]=="PASS_PREPARATION_ONLY" else 1
if __name__=="__main__":raise SystemExit(main())
