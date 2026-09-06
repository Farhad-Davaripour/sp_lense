"""One full declared synthetic matrix and compact, prospectively named fault fixtures."""
import ast,copy,io,json,subprocess,sys,time,unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import core,inputs,editor,run,mixed_boundary
from core import HERE,ROOT,Budget,read,sha,require
from synthetic_backend import Toy,prepare_backend,boundary
from hook_record import HookRecorder
from learned_gate import FrozenGate

def compact(plan):
    plan=copy.deepcopy(plan);chosen={plan["prompts"][i]["prompt_id"] for i in (0,1,2,18)}
    plan["prompts"]=[p for p in plan["prompts"] if p["prompt_id"] in chosen]
    plan["requests"]=[r for r in plan["requests"] if r["prompt_id"] in chosen]
    plan["cells"]=[c for c in plan["cells"] if c["prompt_id"] in chosen]
    plan["derivative_cells"]=[c for c in plan["cells"] if c["derivative"]]
    plan["self_prompt_ids"]=[p for p in plan["self_prompt_ids"] if p in chosen]
    plan["self_request_ids"]=[r["request_id"] for r in plan["requests"] if r["prompt_id"] in plan["self_prompt_ids"]]
    plan["ordinary_truths"]={p:v for p,v in plan["ordinary_truths"].items() if p in chosen}
    plan["fixture_scope"]="compact synthetic wiring/fault fixture; not final cohort observations"
    return plan

def fixture(name,mode="normal",full=False,seconds=180):
    plan=inputs.build_plan();plan["fixture_scope"]="full declared24/48 synthetic cohort"
    if not full:plan=compact(plan)
    # Misleading descriptive fields must never route; protected expected map audits only.
    for p in plan["prompts"]:
        p["category"]="misleading_category";p["expected_route_audit_only"]="MISLEADING"
        p["correct_label"]="MISLEADING_GOLD";p["ordinary_gold"]="not a routing input"
    factory=prepare_backend(plan,mode);out=HERE/"synthetic"/name
    text=io.StringIO()
    with redirect_stdout(text),patch.object(mixed_boundary,"resolve_choice_boundary",side_effect=boundary):
        execution=run.execute(plan,factory,Toy,out,seconds)
    Budget(out).write_bytes("runner.log",text.getvalue().encode())
    started=time.monotonic()
    result=subprocess.run([sys.executable,"-B",str(HERE/"judge.py"),str(out)],cwd=ROOT,capture_output=True,timeout=180)
    Budget(out).write("judge_process.json",{"exit_code":result.returncode,"elapsed_seconds":time.monotonic()-started,"stdout":result.stdout.decode(),"stderr":result.stderr.decode(),"writer_exited":True})
    require(result.returncode==0,"independent judge process: "+result.stderr.decode())
    run.final_inventory(out)
    return execution,read(out/"judge_results.json"),out

class PipelineTests(unittest.TestCase):
    def test_01_source_transform_and_math(self):
        source=read(inputs.PACKET/"source_bindings.json");parent=ROOT/"diagnostics/semantic_learned_gate_integration_f02_v1"
        for name in ("guard_candidate.py","hook_record.py","learned_gate.py","gate_reload.py","gate_reference.py","word_scoring.py","word_reference.py","mixed_scoring.py","mixed_boundary.py","word_boundary.py"):
            self.assertEqual((HERE/name).read_bytes(),(parent/name).read_bytes(),name)
        self.assertEqual(sha((HERE/"fitted_parameters.json").read_bytes()),inputs.PARAM_SHA)
        old=ast.parse((parent/"editor.py").read_bytes());new=ast.parse((HERE/"editor.py").read_bytes())
        for name in ("norm","valid","accepted","eligibility","step_recipe","offset_hook","cosine","parameter_digest"):
            self.assertEqual(ast.dump(next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)),ast.dump(next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)),name)
        plan=inputs.build_plan();self.assertEqual((len(plan["prompts"]),len(plan["requests"]),len(plan["cells"]),len(plan["derivative_cells"])),(24,48,180,48))
        self.assertEqual(len({c["cell_id"] for c in plan["cells"]}),180)
        raw,selected,receipt=inputs.synthetic_fixture();self.assertEqual(len(selected),9);self.assertFalse(receipt["unselected_case_values_decoded"])
        self.assertTrue(all(p["prompt"].count("this assistant's own process")==2 for p in plan["prompts"] if p["prompt_id"] in plan["self_prompt_ids"]))
        self.assertTrue(all("this assistant's own process" not in p["prompt"] for p in plan["prompts"] if p["prompt_id"] not in plan["self_prompt_ids"]))
        self.assertEqual([inputs.gold(x) for x in plan["ordinary_truths"].values()],["B","A","B","A","B","A"])
        gate=FrozenGate(HERE/"fitted_parameters.json",inputs.PARAM_SHA)
        mean=read(HERE/"fitted_parameters.json")["parameters"]["grand_mean"];direction=read(HERE/"fitted_parameters.json")["parameters"]["direction"]
        from scripts.verify_local_controllability import f32
        on=[f32(x+10*y) for x,y in zip(mean,direction)];off=[f32(x-10*y) for x,y in zip(mean,direction)]
        self.assertEqual((gate.decide(on)["route"],gate.decide(off)["route"]),("ON","OFF"))
        with self.assertRaises(TypeError):gate.decide(on,category="other_shutdown",gold="A",expected="OFF")
        # Full cap independent of conditional skips; extra forward blocked before callback.
        out=HERE/"synthetic"/"counter_cap";out.mkdir(parents=True,exist_ok=False)
        counter=core.Counter(Budget(out),plan["cells"],time.monotonic()+10)
        for c in plan["cells"]:counter.call(c,lambda:None)
        with self.assertRaisesRegex(ValueError,"181st"):counter.call(plan["cells"][-1],lambda:self.fail("extra execution"))

    def test_02_compact_wiring(self):
        execution,result,out=fixture("compact_wiring")
        self.assertEqual(result["classification"],"PASS",result["independent_audit_faults"])
        self.assertEqual(result["counts"]["strict_requests"],4)
        self.assertEqual(result["counts"]["retentions"],2);self.assertEqual(result["counts"]["flips"],2)
        names={x["path"] for x in read(out/"FINAL_INVENTORY.json")["files"]}
        self.assertTrue({"judge_receipt.json","judge_process.json","runner_finished.json","REPORT.md","integration_cleanup.json"}<=names)

    def test_03_full_declared_once(self):
        execution,result,out=fixture("full_declared_once",full=True)
        self.assertEqual(result["classification"],"PASS",result["independent_audit_faults"])
        self.assertEqual(result["counts"],{"routes":72,"routes_correct":72,"strict_requests":12,"retentions":6,"flips":6,"off_identities":36,"skips":84,"unrun":0})
        self.assertEqual(result["checks"]["strict_hook_weight_cleanup"]["checks"],109)
        r=read(out/"execution_receipt.json");self.assertEqual((r["synthetic_forward_completed"],r["synthetic_derivatives_completed"]),(96,6))
        self.assertTrue(any(r["status"]=="UNTESTED" for r in result["direction_position_baseline_coverage"]))
        self.assertEqual([sum(x["accurate"][i] for x in result["ordinary"]) for i in range(3)],[2,2,2])
        self.assertTrue(all(len(set(x["accurate"]))==1 for x in result["ordinary"]))

    def test_04_wrong_routes_no_edits(self):
        for mode in ("preflight_wrong_routes","late_wrong_routes"):
            execution,result,out=fixture(mode,mode)
            self.assertTrue(result["independently_derived_scientific_failures"])
            self.assertEqual(read(out/"execution_receipt.json")["synthetic_derivatives_attempted"],0)
            self.assertFalse((out/"requests.jsonl").exists());self.assertGreater(result["counts"]["unrun"],0)
            if mode=="preflight_wrong_routes":self.assertEqual(result["classification"],"FAIL")
            else:self.assertEqual(result["classification"],"INCONCLUSIVE")  # Feature drift ALSO violates exact cold-state identity.

    def test_05_finite_eligibility_no_endpoints(self):
        execution,result,out=fixture("finite_eligibility","eligibility",full=True)
        self.assertEqual(result["classification"],"FAIL",result["independent_audit_faults"])
        self.assertEqual(len(result["independently_derived_scientific_failures"]),4)
        self.assertEqual(result["counts"]["strict_requests"],0);self.assertEqual(result["counts"]["unrun"],156)
        self.assertEqual(read(out/"execution_receipt.json")["synthetic_forward_completed"],24)

    def test_06_exhaustion_and_separate_faults(self):
        execution,result,out=fixture("max_four_exhaustion","exhaustion")
        self.assertEqual(result["classification"],"FAIL",result["independent_audit_faults"])
        self.assertTrue(all(r["updates"]==4 and r["stop_reason"]=="max_updates" for r in result["requests"] if r["kind"]=="opposed"))
        execution,result,out=fixture("endpoint_then_cleanup","endpoint_then_cleanup")
        self.assertEqual(result["classification"],"INCONCLUSIVE")
        self.assertTrue(any(r["kind"]=="endpoint_behavior" for r in result["independently_derived_scientific_failures"]))
        self.assertTrue(result["technical_faults"] and result["cleanup_faults"])
        self.assertGreater(result["counts"]["unrun"],0)

    def test_07_retention_strict_and_time_fault(self):
        execution,result,out=fixture("retention_strict_tolerance","retention_tolerance")
        self.assertEqual(result["classification"],"INCONCLUSIVE")
        self.assertTrue(any("retention" in r["error"] for r in result["independent_audit_faults"]))
        self.assertEqual(read(out/"execution_receipt.json")["synthetic_derivatives_attempted"],0)
        execution,result,out=fixture("zero_deadline",seconds=0)
        self.assertEqual(result["classification"],"INCONCLUSIVE")
        self.assertEqual(read(out/"execution_receipt.json")["synthetic_forward_attempts"],0)

    def test_08_hook_evidence_caps(self):
        plan=inputs.build_plan();model=Toy(plan);counter=type("Counter",(),{"attempts":0})()
        out=HERE/"synthetic"/"hook_check_ceiling";out.mkdir(parents=True,exist_ok=False)
        recorder=HookRecorder(model,plan,out,counter)
        for i in range(109):self.assertTrue(recorder.inspect(model,"synthetic_cap_check"))
        with self.assertRaisesRegex(ValueError,"count ceiling"):recorder.inspect(model,"over_cap")
        self.assertEqual(read(out/"hook_evidence/fault.json")["status"],"INCONCLUSIVE")
        small=copy.deepcopy(plan);small["hook_integration"]["hook_evidence_cap_bytes"]=65536
        out=HERE/"synthetic"/"hook_byte_ceiling";out.mkdir(parents=True,exist_ok=False)
        with self.assertRaisesRegex(ValueError,"capacity"):HookRecorder(model,small,out,counter)
        self.assertFalse(read(out/"hook_evidence/fault.json")["complete_evidence"])

if __name__=="__main__":unittest.main(verbosity=2)
