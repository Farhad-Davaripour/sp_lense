"""One pure admission/closeout batch: no backend, model, tokenizer or process launch."""
import copy,io,json,math,sys,time,unittest
from pathlib import Path
from unittest.mock import patch
from core import HERE,ROOT,read,sha,json_bytes,check_freeze,require,Budget
import admission,usage_receipt,final_adjudication,judge

NOW=1900000000.0
SOURCE="a"*40
ROWS=[]

def payload(percent=89,secondary=None):
    return {"rateLimitsByLimitId":{"codex":{"limitId":"codex","primary":{"usedPercent":percent},"secondary":None if secondary is None else {"usedPercent":secondary}},
        "codex_bengalfox":{"limitId":"codex_bengalfox","primary":{"usedPercent":0},"spendControlReached":True}}}

def mcp(value):return {"content":[{"type":"text","text":json.dumps(value,allow_nan=True)}],"isError":False}

def release(receipt=None,declared=89):
    receipt=read(HERE/"observed_usage.json") if receipt is None else receipt
    return {"authorizer":"root","scope":"ONE_REAL_FINAL_ASSESSMENT","source_commit":SOURCE,
        "freeze_sha256":sha((HERE/"freeze.json").read_bytes()),"input_lock_sha256":admission.INPUT_LOCK,
        "limits":{"forwards":180,"derivatives":48,"loads":1,"worker_seconds":1500,"cleanup_seconds":15,"audit_seconds":180,"bytes":288*1024**2},
        "usage":{"source":"get_usage_limits","bucket":"standard_codex","captured_at_unix":NOW-1,"used_percent":declared,"tool_receipt":receipt},
        "usage_receipt_sha256":sha(json_bytes(receipt))}

def complete_capture():
    return {"status":"complete_valid","quiescent":True,"worker_exit_code":0,"eof_observed":True,
        **{k:True for k in ("worker_joined","reader_joined","budget_watcher_joined","fault_writer_joined","prefix_reader_joined")},
        "technical_recording_fault":None,"cleanup_error":None,"fault_persistence_error":False}

class Pure(unittest.TestCase):
    def setUp(self):self.plan=read(HERE/"production_plan.json")
    def note(self,name,**details):ROWS.append({"check":name,"status":"PASS",**details})
    def denied(self,fn):
        with self.assertRaises((ValueError,KeyError,TypeError)):fn()

    def validate(self,value,raw=None,pin=None,now=NOW):
        raw=json_bytes(value) if raw is None else raw
        path=SOURCE+":"+HERE.relative_to(ROOT).as_posix()+"/freeze.json"
        def git(*args):
            if args==("show",path):return (HERE/"freeze.json").read_bytes()
            raise ValueError("fixture rejects wrong source commit")
        with patch.object(admission,"git",git):return admission.validate_release(value,raw,sha(raw) if pin is None else pin,self.plan,now)

    def test_01_observed_shapes_and_allowed_percentages(self):
        self.assertEqual(usage_receipt.derive_standard_usage(read(HERE/"observed_usage.json"))["used_percent"],89)
        for percent in (0,89,90,95,99):
            direct=payload(percent);expected=usage_receipt.derive_standard_usage(direct)
            for shape in (direct,mcp(direct),{"structuredContent":direct},{**mcp(direct),"structuredContent":copy.deepcopy(direct)}):
                self.assertEqual(usage_receipt.derive_standard_usage(shape)["used_percent"],percent)
                self.assertTrue(self.validate(release(shape,percent)))
        legacy={"rateLimits":{"limitId":"codex","primary":{"usedPercent":94},"secondary":None},"rateLimitsByLimitId":{"codex_bengalfox":{"primary":{"usedPercent":0}}}}
        self.assertTrue(self.validate(release(mcp(legacy),94)))
        modern=payload(91,97);modern["rateLimits"]={"limitId":"codex","primary":{"usedPercent":4}}
        self.assertTrue(self.validate(release(mcp(modern),97)))
        secondary=payload(89,96);secondary["rateLimitsByLimitId"]["codex"]["primary"]=None
        self.assertTrue(self.validate(release(mcp(secondary),96)))
        self.note("observed_text_direct_structured_agreement_modern_preferred_legacy_fallback_0_89_90_95_99_Spark_ignored")

    def test_02_fail_closed_usage(self):
        invalid=[{}, {"isError":True,"structuredContent":payload()}, {"error":"unavailable","structuredContent":payload()},
            {"structuredContent":None}, {"content":[]}, {"content":[{"type":"text","text":"service unavailable"}]},
            {"content":[{"type":"image","data":"none"}]}, {"rateLimitsByLimitId":{"codex":None},"rateLimits":{"primary":{"usedPercent":2}}}]
        for percent in (None,-1,101,True,float("nan"),float("inf")):
            invalid.append({"structuredContent":payload(percent)})
        empty=payload();empty["rateLimitsByLimitId"]["codex"]["primary"]=None;invalid.append(empty)
        for key in usage_receipt.BOOLEAN_FLAGS:
            p=payload();p["rateLimitsByLimitId"]["codex"][key]=True;invalid.append(p)
        for where in ("top","bucket","legacy","window"):
            p=payload()
            target=p if where=="top" else p["rateLimitsByLimitId"]["codex"]
            if where=="legacy":p["rateLimits"]={"limitId":"codex"};target=p["rateLimits"]
            if where=="window":target=target["primary"]
            target["rateLimitReachedType"]="standard_exhausted";invalid.append(p)
        ambiguity={**mcp(payload(89,20)),"structuredContent":payload(89,21)};invalid.append(ambiguity)
        invalid.append({"content":[{"type":"text","text":'{"rateLimits":{"primary":{"usedPercent":89,"usedPercent":0}}}'}]})
        for value in invalid:self.denied(lambda value=value:usage_receipt.derive_standard_usage(value))
        for receipt,declared in ((mcp(payload(100)),89),(mcp(payload(89,100)),89),(mcp(payload(90)),89),(mcp(payload(100)),100)):
            self.denied(lambda: self.validate(release(receipt,declared)))
        self.note("errors_unavailable_nonfinite_flags_secondary100_ambiguous_views_duplicate_keys_percentage_mismatch",negative_cases=len(invalid)+4)

    def test_03_release_pin_freshness_and_positive_admission(self):
        baseline=release();self.assertTrue(self.validate(baseline))
        for change in ({"authorizer":"user"},{"scope":"another_run"},{"source_commit":"b"*40},{"freeze_sha256":"0"*64},{"input_lock_sha256":"0"*64},{"limits":{"forwards":12}},{"usage_receipt_sha256":"0"*64}):
            v=copy.deepcopy(baseline);v.update(change);self.denied(lambda:self.validate(v))
        for timestamp in (NOW-120.001,NOW+0.001,None,True):
            v=copy.deepcopy(baseline);v["usage"]["captured_at_unix"]=timestamp;self.denied(lambda:self.validate(v))
        for change in ({"source":"estimated"},{"bucket":"spark"},{"percentage":89}):
            v=copy.deepcopy(baseline);v["usage"].update(change);self.denied(lambda:self.validate(v))
        self.denied(lambda:self.validate(baseline,pin="0"*64))
        altered=copy.deepcopy(baseline);altered["scope"]="different";self.denied(lambda:self.validate(altered,raw=json_bytes(baseline)))
        duplicate=json_bytes(baseline).replace(b'"authorizer": "root",',b'"authorizer": "root", "authorizer": "root",')
        self.denied(lambda:self.validate(baseline,raw=duplicate))
        # Isolated in-memory committed document stubs; actual authorization bytes never change.
        authpath=HERE/"authorization.json";releasepath=HERE/"approved_root_release.json"
        oldauth=authpath.read_bytes();self.denied(admission.admit_production)
        raw=json_bytes(baseline);policy={"run_authorized":True,"root_release_sha256":sha(raw)};policyraw=json_bytes(policy)
        original_read=admission.read;original_bytes=Path.read_bytes;original_git=admission.git
        prefix=HERE.relative_to(ROOT).as_posix()
        def patched_read(path):return policy if Path(path)==authpath else original_read(path)
        def patched_bytes(path):
            if path==authpath:return policyraw
            if path==releasepath:return raw
            return original_bytes(path)
        def patched_git(*args):
            if args==("show","HEAD:"+prefix+"/authorization.json"):return policyraw
            if args==("show","HEAD:"+prefix+"/approved_root_release.json"):return raw
            if args==("show",SOURCE+":"+prefix+"/freeze.json"):return original_bytes(HERE/"freeze.json")
            return original_git(*args)
        with patch.object(admission,"read",patched_read),patch.object(Path,"read_bytes",patched_bytes),patch.object(admission,"git",patched_git),patch.object(admission.time,"time",lambda:NOW):
            admitted_plan,admitted_release=admission.admit_production()
            self.assertEqual(admitted_plan,self.plan);self.assertEqual(admitted_release,baseline)
        self.assertEqual(authpath.read_bytes(),oldauth);self.assertFalse(releasepath.exists());self.denied(admission.admit_production)
        self.note("positive_validate_and_admit_in_memory_actual_exact_cohort_environment_disabled_actual_authorization_no_launch")

    def test_04_exact_cohort_and_audit_external_source_order(self):
        counts=admission.exact_cohort(self.plan);self.assertEqual(counts["forwards"],180)
        compact=copy.deepcopy(self.plan);compact["prompts"]=compact["prompts"][:2];self.denied(lambda:admission.exact_cohort(compact))
        contract=self.plan["gate"]["runtime_compatibility"]
        runtime={**contract,"execution_mode":"PRODUCTION_FINAL","locked_inputs_sha256":self.plan["input_binding"]["input_sha256"],"tokenizer_calls_after_load":0,
            "boundaries":[{"prompt_id":p["prompt_id"],**self.plan["alignment"][p["prompt_id"]]} for p in self.plan["prompts"]]}
        output=HERE/"IN_MEMORY_AUDIT_NO_DIRECTORY";sequence=[]
        def reader(path):
            if path==output/"plan.json":return self.plan
            if path==output/"runtime.json":return runtime
            raise AssertionError("unexpected fixture read")
        def exact(plan):sequence.append("exact_cohort");return admission.exact_cohort(plan)
        def env(plan):sequence.append("environment");return admission.environment(plan)
        def saved(path):sequence.append("saved_judge_stub");return {"classification":"PASS","fixture_only":True}
        verifier=ROOT/"scripts/verify_local_controllability.py";original_bytes=Path.read_bytes
        external=read(HERE/"runtime_source_receipt.json")["external_sha256"]
        self.assertIn(str(verifier),external)
        def tampered(path):return original_bytes(path)+b"#changed verifier" if path==verifier else original_bytes(path)
        with patch.object(judge,"read",reader),patch.object(judge,"exact_cohort",exact),patch.object(judge,"environment",env),patch.object(judge.saved_judge,"judge",saved),patch.object(admission,"admit_production",side_effect=AssertionError("no final-audit usage/admission call")):
            with patch.object(Path,"read_bytes",tampered):self.denied(lambda:judge.judge_production(output))
            self.assertEqual(sequence,["exact_cohort","environment"])
            sequence.clear();self.assertEqual(judge.judge_production(output)["classification"],"PASS")
            self.assertEqual(sequence,["exact_cohort","environment","saved_judge_stub"])
        self.assertFalse(output.exists())
        self.note("exact_cohort_before_environment_before_saved_judge_external_verifier_tamper_blocks_no_usage_refresh")

    def test_05_authoritative_closeout(self):
        good=complete_capture();receipt={"status":"complete","elapsed_seconds":1.25}
        passed={"classification":"PASS","counts":{"fixture":1}}
        failed={"classification":"FAIL","independently_derived_scientific_failures":[{"kind":"finite_endpoint_failure","request_id":"fixture_P"}]}
        for verdict in (passed,failed):
            self.assertEqual(final_adjudication.join(good,good,receipt,verdict)["classification"],verdict["classification"])
            for field,value in (("status","technical_invalid"),("technical_recording_fault","worker_timeout"),("technical_recording_fault","log_limit"),("cleanup_error","join_failed"),("worker_exit_code",1),("worker_exit_code",False),("eof_observed",False)):
                bad={**good,field:value};result=final_adjudication.join(good,bad,receipt,verdict)
                self.assertEqual(result["classification"],"INCONCLUSIVE");self.assertEqual(result["raw_judge_classification"],verdict["classification"])
                self.assertEqual(result["scientific_findings"],verdict.get("independently_derived_scientific_failures",[]))
        for malformed in (None,{},[],{"status":"complete"},{"status":"complete","elapsed_seconds":True},{"status":"complete","elapsed_seconds":float("nan")},{"status":"complete","elapsed_seconds":-1},{"status":"complete","elapsed_seconds":1,"error":"latefault"}):
            self.assertEqual(final_adjudication.join(good,good,malformed,passed)["classification"],"INCONCLUSIVE")
        self.assertEqual(final_adjudication.join(None,good,receipt,passed)["classification"],"INCONCLUSIVE")
        self.assertEqual(final_adjudication.join(good,good,receipt,{})["classification"],"INCONCLUSIVE")
        output=HERE/"pure_closeout_fixture";output.mkdir(exist_ok=False);budget=Budget(output)
        budget.write("judge_results.json",failed);budget.write("judge_receipt.json",receipt)
        budget.write_bytes("judge_report.md",b"Intermediate synthetic judge FAIL, preserved.\n")
        budget.event("scientific_failures.jsonl",failed["independently_derived_scientific_failures"][0])
        budget.event("technical_faults.jsonl",{"kind":"fixture_audit_timeout"})
        budget.event("cleanup_errors.jsonl",{"kind":"fixture_cleanup_fault"});budget.write("unrun.json",[{"request_id":"later_UNRUN"}])
        preserved={p.name:sha(p.read_bytes()) for p in output.iterdir()}
        self.denied(lambda:final_adjudication.finalize(output,good,{**good,"quiescent":False}))
        self.assertFalse((output/"final_closeout.json").exists())
        final=final_adjudication.finalize(output,good,{**good,"status":"technical_invalid","technical_recording_fault":"worker_timeout"})
        self.assertEqual(final["classification"],"INCONCLUSIVE");self.assertEqual(final["raw_judge_classification"],"FAIL")
        for name,digest in preserved.items():self.assertEqual(sha((output/name).read_bytes()),digest)
        self.assertEqual(set(final["preserved_artifacts"]),set(preserved))
        self.assertIn("Final assessment: INCONCLUSIVE",(output/"REPORT.md").read_text())
        self.note("capture_fault_overrides_PASS_retains_FAIL_malformed_receipt_failclosed_quiescent_writer_raw_artifact_identity")

if __name__=="__main__":
    check_freeze();started=time.monotonic();exit_code=1
    result=None;stream=io.StringIO()
    try:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Pure))
        require(not any(name in sys.modules for name in ("torch","transformers","transformer_lens")),"no model/runtime imports in pure batch")
        require(not (HERE/"real_attempt").exists() and not (HERE/"approved_root_release.json").exists(),"no real attempt or authorization written")
        check_freeze();exit_code=0 if result.wasSuccessful() else 1
    finally:
        Budget(HERE).write_bytes("test_output.log",stream.getvalue().encode());print(stream.getvalue(),end="")
        Budget(HERE).write("test_receipt.json",{"status":"PASS" if exit_code==0 else "FAIL","tests_run":0 if result is None else result.testsRun,
            "failures":[] if result is None else [{"test":str(t),"trace":s} for t,s in result.failures+result.errors],"checks":ROWS,
            "elapsed_seconds":time.monotonic()-started,"synthetic_in_memory_admission_only":True,"model_loads":0,"forwards":0,"derivatives":0,
            "tokenizer_calls":0,"gate_scores":0,"gate_fits":0,"child_process_launches":0,"production_authorized":False})
    raise SystemExit(exit_code)
