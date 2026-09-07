"""Exactly one combined fixture and route-only/eligibility-only regressions."""
import io,json,subprocess,sys,time
from contextlib import redirect_stdout
from unittest.mock import patch
from core import HERE,ROOT,Budget,check_freeze,read,require
import inputs,run,mixed_boundary
from synthetic_backend import Toy,prepare_backend,boundary

def verify():
    check_freeze();started=time.monotonic();budget=Budget(HERE)
    require(not (HERE/"TEST_STARTED.json").exists(),"one frozen test batch only")
    budget.write("TEST_STARTED.json",{"fixtures":["combined_once","route_only","eligibility_only"],"monotonic":started})
    results=[]
    for name,mode,expected,exception in (
        ("combined_once","preflight_join",[("routing","join_fixture_1__baseline"),("eligibility","join_fixture_1__baseline"),("eligibility","join_fixture_2__baseline")],"RoutingMismatch"),
        ("route_only","preflight_wrong_routes",[("routing","join_fixture_1__baseline")],"RoutingMismatch"),
        ("eligibility_only","eligibility",[("eligibility","join_fixture_1__baseline"),("eligibility","join_fixture_2__baseline")],"EligibilityError")):
        require(time.monotonic()-started<120,"test batch deadline")
        plan=inputs.build_plan();factory=prepare_backend(plan,mode);output=HERE/name;text=io.StringIO();edit_registrations=[]
        original=run.editor.offset_hook
        def forbidden_edit(delta):
            edit_registrations.append(True)
            raise AssertionError("preflight failure must not construct/register edit")
        with redirect_stdout(text),patch.object(mixed_boundary,"resolve_choice_boundary",side_effect=boundary),patch.object(run.editor,"offset_hook",side_effect=forbidden_edit):
            execution=run.execute(plan,factory,Toy,output,seconds=30)
        Budget(output).write_bytes("runner.log",text.getvalue().encode())
        require(not edit_registrations and run.editor.offset_hook is original,"zero edit constructions; instrumentation restored")
        receipt=read(output/"execution_receipt.json")
        require(receipt["synthetic_forward_attempts"]==receipt["synthetic_forward_completed"]==2,"both preflight captures completed")
        require(receipt["synthetic_derivatives_attempted"]==receipt["synthetic_derivatives_completed"]==0,"zero derivatives")
        require(execution["status"]=="scientific_stop" and execution["error"].startswith(exception+":"),"original stop priority preserved")
        facts=[json.loads(line) for line in (output/"scientific_failures.jsonl").read_text().splitlines()]
        require([(r["kind"],r["cell_id"]) for r in facts]==expected,"all preflight failures durably collected in order")
        require(len(read(output/"unrun.json"))==40 and not (output/"requests.jsonl").exists(),"no requests or fabricated endpoints; remaining40UNRUN")
        judge_start=time.monotonic()
        process=subprocess.run([sys.executable,"-B",str(HERE/"judge.py"),str(output)],cwd=ROOT,capture_output=True,timeout=30)
        require(len(process.stdout)+len(process.stderr)<=65536,"bounded judge process output")
        Budget(output).write("judge_process.json",{"exit_code":process.returncode,"elapsed_seconds":time.monotonic()-judge_start,"stdout":process.stdout.decode(),"stderr":process.stderr.decode(),"writer_exited":True})
        require(process.returncode==0,"unchanged independent judge process")
        judged=read(output/"judge_results.json")
        require(judged["classification"]=="FAIL" and not judged["independent_audit_faults"] and not judged["technical_faults"] and not judged["cleanup_faults"],"scientific FAIL without spurious technical ledger mismatch")
        require([(r["kind"],r["cell_id"]) for r in judged["independently_derived_scientific_failures"]]==expected,"judge independently agrees with joined failure set")
        require(judged["counts"]["unrun"]==40 and judged["counts"]["strict_requests"]==0,"all missing outcomes remain unperformed")
        run.final_inventory(output)
        results.append({"fixture":name,"verification":"PASS","experiment_classification":"FAIL","forwards":2,"derivatives":0,"edit_registrations":0,"remaining_unrun":40,"failure_keys":expected,"judge_audit_faults":[],"stop_exception":exception})
    check_freeze()
    budget.write("test_receipt.json",{"verification":"PASS","elapsed_seconds":time.monotonic()-started,"fixtures":results,"full_matrix_runs":0,"real_model_loads":0,"tokenizer_calls":0,"gate_fit_calls":0,"final_input_reads":0})
    print(json.dumps({"verification":"PASS","fixtures":len(results),"forwards":6,"derivatives":0,"elapsed_seconds":time.monotonic()-started}))

if __name__=="__main__":verify()
