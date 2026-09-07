"""One representative20F2D toy traversal plus compact admission/process faults."""
import copy,io,json,subprocess,sys,time,unittest
from pathlib import Path
from unittest.mock import patch
from core import HERE,Budget,Counter,read,require,sha,json_bytes
import admission,run,judge

def fixture_plan(indices):
    plan=copy.deepcopy(read(HERE/"production_plan.json"));keep={plan["prompts"][i]["prompt_id"] for i in indices}
    plan["execution_mode"]="SYNTHETIC_ONLY";plan["fixture_scope"]="compact_runtime_changed_path"
    plan["prompts"]=[p for p in plan["prompts"] if p["prompt_id"] in keep]
    for p in plan["prompts"]:p["execution_mode"]="SYNTHETIC_ONLY"
    for key in ("requests","cells","derivative_cells"):plan[key]=[x for x in plan[key] if x["prompt_id"] in keep]
    plan["self_prompt_ids"]=[i for i in plan["self_prompt_ids"] if i in keep]
    plan["self_request_ids"]=[r["request_id"] for r in plan["requests"] if r["prompt_id"] in plan["self_prompt_ids"]]
    for key in ("expected_routes","ordinary_truths","alignment"):plan[key]={k:v for k,v in plan[key].items() if k in keep}
    return plan

def initialize(output,plan):
    output.mkdir(exist_ok=False);b=Budget(output);b.write("plan.json",plan)
    b.write_bytes("fitted_parameters.json",(HERE/"fitted_parameters.json").read_bytes())
    b.write("release.json",{"freeze_sha256":sha((HERE/"freeze.json").read_bytes()),"scope":"SYNTHETIC_ONLY_NOT_REAL_EVIDENCE"})

def toy_fixture(name,indices,mode):
    import synthetic_backend as fake
    plan=fixture_plan(indices);original_factory=fake.prepare_backend(plan,mode)
    def factory():
        backend=original_factory()
        def forbidden_retokenization(text):raise AssertionError("underlying backend.encode/tokenizer must never be called")
        backend.encode=forbidden_retokenization
        return backend
    for i,p in enumerate(plan["prompts"],1):
        plan["alignment"][p["prompt_id"]]={"full_token_ids":[i,2,3],"attention_mask":[1,1,1],"final_input_mask":[0,0,1],
            "prompt_length":3,"final_input_index":2,"content_token_ids":p["token_map"],"synthetic_not_tokenized":True}
    output=HERE/name;initialize(output,plan)
    state=run.execute(plan,factory,fake.Toy,output,time.monotonic()+60,synthetic=True)
    # This independent judge consumes saved logits/states, not state['execution_status'].
    result=judge.judge_synthetic(output);Budget(output).write("independent_judge.json",result)
    return plan,state,result

class Focused(unittest.TestCase):
    def test_01_exact_admission_no_real_release_and_mismatches(self):
        plan=read(HERE/"production_plan.json");self.assertEqual(admission.exact_cohort(plan)["prompts"],24)
        with self.assertRaisesRegex(ValueError,"NO REAL MODEL RELEASE"):run.production_run()
        self.assertFalse((HERE/"real_attempt").exists())
        for variant in ("compact","prompt","map","mask","runtime"):
            changed=copy.deepcopy(plan)
            if variant=="compact":changed["prompts"]=changed["prompts"][:-1]
            elif variant=="prompt":changed["prompts"][0]["prompt"]+=" ";changed["prompts"][0]["prompt_sha256"]=sha(changed["prompts"][0]["prompt"].encode())
            elif variant=="map":changed["prompts"][0]["token_map"]={"KEEP":32,"STOP":33}
            elif variant=="mask":changed["alignment"][changed["prompts"][0]["prompt_id"]]["attention_mask"][0]=0
            else:changed["model"]["revision"]="not pinned"
            with self.assertRaises(ValueError):admission.exact_cohort(changed)
        with patch("admission.importlib.metadata.version",return_value="wrong"):
            with self.assertRaisesRegex(ValueError,"package"):admission.environment(plan)
        with self.assertRaises(ValueError):admission.validate_release({},b"{}",None,plan,time.time())

    def test_02_one_cold_successful_changed_path_traversal(self):
        plan,state,result=toy_fixture("representative_once",(0,1,2,18),"normal")
        self.assertEqual(state["forward_attempts"],20);self.assertEqual(state["derivatives_completed"],2)
        self.assertEqual(result["classification"],"PASS");self.assertEqual(result["counts"]["retentions"],2)
        self.assertEqual(result["counts"]["flips"],2);self.assertEqual(result["counts"]["off_identities"],4)
        self.assertEqual(result["counts"]["routes"],12)
        with self.assertRaises(ValueError):judge.judge_production(HERE/"representative_once")

    def test_03_repaired_simultaneous_preflight_failure(self):
        # Only two preflight toy forwards; no request traversal or endpoint invented.
        plan,state,result=toy_fixture("combined_fault",(0,1),"preflight_join")
        self.assertEqual(state["forward_attempts"],2);self.assertEqual(state["derivatives_completed"],0)
        self.assertEqual(result["classification"],"FAIL")
        kinds=[x["kind"] for x in result["independently_derived_scientific_failures"]]
        self.assertIn("routing",kinds);self.assertIn("eligibility",kinds)
        self.assertEqual(len(result["requests"]),0);self.assertEqual(len(read(HERE/"combined_fault/unrun.json")),40)

    def test_04_guard_before_loader_and_absolute_counter(self):
        import synthetic_backend as fake
        plan=fixture_plan((0,1));factory=fake.prepare_backend(plan,"normal");output=HERE/"preload_fault";initialize(output,plan)
        seen=[]
        def bad_loader():
            backend=factory();seen.append(backend);backend.model(backend.torch.tensor([[1,2,3]]));return backend
        state=run.execute(plan,bad_loader,fake.Toy,output,time.monotonic()+30,synthetic=True)
        self.assertEqual(state["forward_attempts"],0);self.assertEqual(seen[0].model.visits,{})
        class MemoryBudget:
            def event(self,*args):pass
        cells=[{"cell_id":str(i)} for i in range(180)];counter=Counter(MemoryBudget(),cells,time.monotonic()+10);executed=[]
        for cell in cells:counter.call(cell,lambda:executed.append(1))
        with self.assertRaises(ValueError):counter.call({"cell_id":"181"},lambda:executed.append(1))
        self.assertEqual(len(executed),180)

    def test_05_actual_child_deadline_logs_and_fault_preservation(self):
        for mode in ("complete","deadline","overflow","prior_science_then_fault"):
            output=HERE/("process_"+mode);output.mkdir();plan=fixture_plan((0,1));Budget(output).write("plan.json",plan)
            capture=run.supervise([sys.executable,"-B",str(HERE/"fake_child.py"),mode,str(output)],output,.3 if mode=="deadline" else 10)
            self.assertTrue(capture["quiescent"]);self.assertEqual(capture["process_attempts"],1)
            if mode=="complete":self.assertEqual(capture["status"],"complete_valid")
            else:
                self.assertEqual(capture["status"],"INCONCLUSIVE");run.recover_missing_closeout(output,plan,capture)
                self.assertEqual(len(read(output/"unrun.json")),len(plan["cells"]))
            if mode=="deadline":self.assertTrue(capture["termination_attempted"])
            if mode=="overflow":self.assertLessEqual((output/"worker.log").stat().st_size,4*1024**2)
            if mode=="prior_science_then_fault":
                self.assertIn("endpoint_behavior",(output/"scientific_failures.jsonl").read_text());self.assertIn("synthetic_cleanup_fault",(output/"cleanup_errors.jsonl").read_text())

    def test_06_kill_fallback_cap_and_inventory_quiescence(self):
        class FakeProcess:
            pid=987654321
            def __init__(self,*args,**kwargs):self.stdout=io.BytesIO(b"fake process\n");self.killed=False;self.terminated=False
            def poll(self):return -9 if self.killed else None
            def terminate(self):self.terminated=True
            def wait(self,timeout):
                if not self.killed:raise subprocess.TimeoutExpired("fake",timeout)
                return -9
            def kill(self):self.killed=True
        output=HERE/"fake_kill_fallback";output.mkdir()
        with patch("run.subprocess.Popen",FakeProcess):capture=run.supervise([sys.executable,"-B",str(HERE/"fake_child.py"),"deadline",str(output)],output,.04)
        self.assertTrue(capture["kill_attempted"] and capture["quiescent"])
        cap=HERE/"cap_fault";cap.mkdir()
        with self.assertRaisesRegex(ValueError,"per-file"):Budget(cap).write_bytes("too_big.bin",b"x"*(5*1024**2+1))
        with self.assertRaises((ValueError,FileNotFoundError)):run.final_inventory(output)
        from types import SimpleNamespace
        class FakeDiskEntry:
            def is_file(self):return True
            def stat(self):return SimpleNamespace(st_size=288*1024**2)
        with patch("core.Path.rglob",return_value=[FakeDiskEntry()]):
            with self.assertRaisesRegex(ValueError,"all-artifact"):Budget(cap).write_bytes("aggregate_fault.bin",b"x")
        self.assertFalse((cap/"aggregate_fault.bin").exists())
        complete=HERE/"inventory_lifecycle";complete.mkdir();(complete/"audit").mkdir()
        for path in (complete,complete/"audit"):
            Budget(path).write("capture.json",{"quiescent":True});Budget(path).write("supervisor_final.json",{"quiescent":True})
        Budget(complete).write("judge_process.json",{"synthetic_process_fixture":True})
        Budget(complete).write_bytes("REPORT.md",b"Synthetic inventory lifecycle only.\n")
        run.final_inventory(complete)
        names={r["path"] for r in read(complete/"FINAL_INVENTORY.json")["files"]}
        self.assertIn("audit/supervisor_final.json",names);self.assertIn("judge_process.json",names);self.assertIn("REPORT.md",names)

if __name__=="__main__":
    start=time.monotonic();Budget(HERE).write("TEST_STARTED.json",{"scope":"ONE declared focused fake batch","monotonic":start,"real_model_authorized":False})
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Focused))
    Budget(HERE).write("test_receipt.json",{"status":"PASS" if result.wasSuccessful() else "FAIL","tests":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
        "elapsed_seconds":time.monotonic()-start,"real_model_loads":0,"real_forwards":0,"tokenizer_calls":0,"real_state_gate_scores":0,"gate_fits":0,
        "one_representative_traversal":True,"synthetic_only":True})
    raise SystemExit(0 if result.wasSuccessful() else 1)
