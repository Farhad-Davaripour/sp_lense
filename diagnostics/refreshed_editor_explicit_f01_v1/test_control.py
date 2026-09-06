"""Focused fake-model schedule, arithmetic, integrity and closeout regressions."""
import ast
import copy
import json
import math
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
torch.set_num_threads(1)
import core
import editor
import score
import run as runner
from sp_lense import comparison_runtime
from scripts.verify_local_controllability import rows_at

class Toy(torch.nn.Module):
    def __init__(self,offset=-.6,kind="linear",quality=False,mismatch=False,mutate=False):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor(1.))
        self.offset,self.kind,self.quality,self.mismatch,self.mutate=offset,kind,quality,mismatch,mutate
        self.active_hooks=[]
        self.calls=0
    @contextmanager
    def hooks(self,fwd_hooks):
        start=len(self.active_hooks)
        self.active_hooks.extend(fwd_hooks)
        try: yield self
        finally: del self.active_hooks[start:]
    def forward(self,tokens):
        self.calls+=1
        assert not self.weight.requires_grad
        h=torch.tensor([0.,3.,4.]+[0.]*1021).repeat(1,3,1)
        for name,fn in self.active_hooks:
            assert name=="blocks.10.hook_out"
            h=fn(h,SimpleNamespace(name=name))
        x=h[...,0]
        z=x if self.kind=="linear" else .3*torch.tanh(x) if self.kind=="saturating" else x*0
        z=z+self.offset
        if self.mismatch and torch.is_grad_enabled(): z=z+.01
        values=torch.full((1,3,248320),-100.)
        values[...,32]=z
        values[...,33]=0.
        if self.quality: values[...,3]=torch.where(x>0,torch.full_like(x,10.),torch.full_like(x,-100.))
        if self.mutate: self.weight.add_(.01)
        return values

def toy_setup(plan,**kwargs):
    plan=copy.deepcopy(plan)
    model=Toy(**kwargs)
    prompts=plan["prompts"]
    tokens_by_text={p["prompt"]:torch.tensor([[i+1,2,3]]) for i,p in enumerate(prompts)}
    for i,p in enumerate(prompts):
        plan["alignment"][p["prompt_id"]]["full_token_ids"]=[i+1,2,3]
    backend=SimpleNamespace(model=model,torch=torch,encode=lambda text:tokens_by_text[text])
    boundary=SimpleNamespace(a_token_id=32,b_token_id=33,prompt_length=3,evidence_sha256="fake",
                             token_id=lambda label:32 if label=="A" else 33)
    return plan,backend,boundary

def execute(root,plan,**kwargs):
    plan,backend,boundary=toy_setup(plan,**kwargs)
    counter=core.Counter(core.Budget(root),plan["cells"],time.monotonic()+60)
    derivative=editor.DerivativeLedger(root,plan["derivative_cells"],time.monotonic()+60)
    guard=runner.ForwardGuard(Toy,counter)
    guard.install()
    try:
        with patch.object(comparison_runtime,"resolve_choice_boundary",return_value=boundary):
            rows,requests=editor.evaluate(plan,backend,counter,derivative,root,guard)
    finally: guard.restore()
    checked=score.verify_data(plan,rows,requests,counter.skips,root)
    return plan,backend,counter,derivative,rows,requests,checked

class EditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=core.source_plan()

    def test_source_math_parity_no_duplicate_and_conditional_counts(self):
        old=ast.parse((core.ROOT/"scripts/refreshed_gradient_control.py").read_text())
        new=ast.parse((core.HERE/"editor.py").read_text())
        for name in ("step_recipe","offset_hook","cosine"):
            a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
            b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(a),ast.dump(b))
        self.assertEqual(len(self.plan["cells"]),44)
        self.assertEqual(len(self.plan["derivative_cells"]),16)
        self.assertEqual([c["condition"] for c in self.plan["cells"][:4]],["baseline"]*4)
        self.assertTrue(all(not c["overlap"] for c in self.plan["source_provenance"]["no_duplicate_checks"]))
        with tempfile.TemporaryDirectory() as d:
            counter=core.Counter(core.Budget(d),self.plan["cells"],time.monotonic()+5)
            with self.assertRaises(ValueError): counter.call(None,lambda:None)
            with self.assertRaises(ValueError): counter.skip(self.plan["cells"][0],"accepted","fake")
            for cell in self.plan["cells"]: counter.call(cell,lambda:None)
            with self.assertRaisesRegex(ValueError,"45th"): counter.call(self.plan["cells"][-1],lambda:None)
            derivative=editor.DerivativeLedger(d,self.plan["derivative_cells"],time.monotonic()+5)
            for cell in self.plan["derivative_cells"]: derivative.call(cell,lambda:None)
            with self.assertRaises(ValueError): derivative.call(self.plan["derivative_cells"][-1],lambda:None)
            self.assertEqual((counter.completed,derivative.completed),(44,16))

    def test_success_cold_requests_endpoint_replay_and_final_receipts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            plan,backend,f,deriv,rows,requests,result=execute(root,self.plan)
            self.assertEqual(result["summary"]["classification"],"PASS")
            self.assertEqual((f.completed,deriv.completed),(36,12))
            self.assertEqual(len(f.skips),8)
            self.assertTrue(all(r["updates"]==3 for r in requests))
            self.assertTrue(all(not any(r["cumulative_offset"]) for r in rows if r["condition"]=="gradient_1"))
            self.assertTrue(all(r["maximum_current_logit_difference"]==0 for r in rows if r["condition"]=="endpoint"))
            self.assertEqual(backend.model.weight.item(),1.)
            self.assertTrue(backend.model.weight.requires_grad)
            self.assertIsNone(backend.model.weight.grad)
            budget=core.Budget(root)
            budget.write("freeze.json",{"plan":plan})
            budget.write("request_summary.json",requests)
            budget.write("runtime.json",{"model_id":"Qwen/Qwen3.5-0.8B","model_revision":plan["model"]["revision"],"device":"cpu","dtype":"float32",
                "tokenization_forwards":0,"boundaries":[{"prompt_id":p["prompt_id"],"content_token_ids":{"A":32,"B":33},
                "suffix_alignment":plan["alignment"][p["prompt_id"]],"evidence_sha256":"fake","prompt_length":3} for p in plan["prompts"]]})
            budget.write("input_usage.json",{"captured_at_unix":time.time(),"source":"get_usage_limits","bucket":"standard_codex","used_percent":[77]})
            def fake_worker(command,active):
                active.write("worker_final.json",{"status":"complete","parameter_checks_passed":True,"model_load_attempts":1,"model_load_completed":1,
                    "forward_attempts":f.attempts,"forward_completed":f.completed,"derivatives":deriv.attempts,"derivatives_completed":deriv.completed,
                    "skipped_forwards":len(f.skips),"elapsed_seconds":1.,"load_elapsed_seconds":0.})
                capture={"status":"complete_valid","eof_observed":True,"quiescent":True}
                active.write("capture.json",capture)
                return capture
            def fake_finalize(*a,**kw):
                score.finalize()
                self.assertFalse((root/"FINAL_INVENTORY.json").exists())
                return SimpleNamespace(returncode=0,stdout=b"",stderr=b"")
            with patch.object(runner,"HERE",root),patch.object(score,"HERE",root), \
                 patch.object(runner,"check_freeze",return_value={"plan":plan}),patch.object(score,"check_freeze",return_value={"plan":plan}), \
                 patch.object(runner,"git",side_effect=[b"frozen",b""]),patch.object(runner,"supervise",side_effect=fake_worker), \
                 patch.object(runner.subprocess,"run",side_effect=fake_finalize):
                runner.run(root/"input_usage.json","frozen")
            self.assertEqual(core.read(root/"results.json")["counts"]["strict_total"],8)
            names={e["path"] for e in core.read(root/"FINAL_INVENTORY.json")["files"]}
            self.assertTrue({"finalize_process.json","finalize_receipt.json","REPORT.md","results.json"}<=names)
            # Saved-data corruption must fail without another model call.
            for field in ("cumulative_offset","h"):
                changed=copy.deepcopy(rows)
                row=next(r for r in changed if r["condition"]=="gradient_1")
                row[field][0]+=.01
                with self.assertRaises(ValueError): score.verify_data(plan,changed,requests,f.skips,root)
            altered=copy.deepcopy(requests)
            altered[0]["stop_reason"]="max_updates"
            with self.assertRaises(ValueError): score.verify_data(plan,rows,altered,f.skips,root)

    def test_quality_failure_and_exhausted_budget_do_not_rescue(self):
        for kwargs,expected,counts,reason in [
            ({"quality":True},"FAIL",(20,4),"quality_failure"),
            ({"kind":"saturating"},"PARTIAL",(44,16),"max_updates")]:
            with self.subTest(kwargs=kwargs),tempfile.TemporaryDirectory() as d:
                _,_,f,deriv,rows,requests,result=execute(Path(d),self.plan,**kwargs)
                self.assertEqual(result["summary"]["classification"],expected)
                self.assertEqual((f.completed,deriv.completed),counts)
                self.assertTrue(all(r["stop_reason"]==reason for r in requests))
                self.assertEqual(sum(r["condition"]=="endpoint" for r in rows),4)

    def test_eligibility_gradient_identity_zero_gradient_and_parameter_faults(self):
        for kwargs,forwards,derivatives,pattern in [
            ({"offset":-.01},4,0,"baseline"),
            ({"mismatch":True},6,1,"identity"),
            ({"kind":"zero"},6,1,"gradient"),
            ({"mutate":True},1,0,"parameter")]:
            with self.subTest(kwargs=kwargs),tempfile.TemporaryDirectory() as d:
                plan,backend,boundary=toy_setup(self.plan,**kwargs)
                f=core.Counter(core.Budget(d),plan["cells"],time.monotonic()+60)
                deriv=editor.DerivativeLedger(d,plan["derivative_cells"],time.monotonic()+60)
                guard=runner.ForwardGuard(Toy,f)
                guard.install()
                try:
                    with patch.object(comparison_runtime,"resolve_choice_boundary",return_value=boundary):
                        with self.assertRaisesRegex(ValueError,pattern):
                            editor.evaluate(plan,backend,f,deriv,Path(d),guard)
                finally: guard.restore()
                self.assertEqual((f.attempts,deriv.attempts),(forwards,derivatives))
                self.assertTrue(backend.model.weight.requires_grad)

    def test_geometry_original_norm_and_unique_argmax_gates(self):
        for sign in (-1,1):
            settings=editor.step_recipe(-10.*sign,sign,[3.,4.,0.],[0.,3.,4.])
            self.assertEqual(settings["requested_step_norm"],.25)
            self.assertTrue(settings["step_limited"])
            self.assertLessEqual(4*settings["requested_step_norm"],.20*5)
        for gradient,h0 in [([0.]*3,[1.]*3),([float("nan")]*3,[1.]*3),([1.]*3,[0.]*3)]:
            with self.assertRaises(ValueError): editor.step_recipe(-1,1,gradient,h0)
        row={"answer_pair_mass":.8,"kl_from_baseline":0.,"actual_next_token_id":32,"actual_next_token_label":"A",
             "preserve_log_odds":.05-1e-6,"full_argmax_tie_count":1,"preserve_label":"A"}
        self.assertTrue(editor.accepted(row,1,32))
        self.assertFalse(editor.accepted({**row,"full_argmax_tie_count":2},1,32))
        self.assertFalse(editor.accepted({**row,"answer_pair_mass":.8-1e-12},1,32))
        self.assertFalse(editor.accepted({**row,"preserve_log_odds":.05-1e-6-1e-12},1,32))

    def test_failure_receipt_and_no_repeat(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            counter=core.Counter(core.Budget(root),self.plan["cells"],time.monotonic()+5)
            def fail(): raise RuntimeError("intentional fake failure")
            with self.assertRaises(RuntimeError): counter.call(self.plan["cells"][0],fail)
            with self.assertRaises(ValueError): counter.call(self.plan["cells"][0],lambda:None)
            self.assertEqual(counter.attempted,1)
            self.assertEqual(rows_at(root/"forward_events.jsonl")[-1]["event"],"attempt_failed")
            budget=core.Budget(root)
            budget.write("freeze.json",{"plan":self.plan})
            budget.write("input_usage.json",{"captured_at_unix":time.time(),"source":"get_usage_limits","bucket":"standard_codex","used_percent":[77]})
            def failed_capture(*args):
                receipt={"status":"incomplete_or_invalid","quiescent":True}
                budget.write("capture.json",receipt)
                return receipt
            with patch.object(runner,"HERE",root),patch.object(runner,"check_freeze",return_value={"plan":self.plan}), \
                 patch.object(runner,"git",side_effect=[b"frozen",b""]),patch.object(runner,"supervise",side_effect=failed_capture):
                runner.run(root/"input_usage.json","frozen")
            names={e["path"] for e in core.read(root/"FINAL_INVENTORY.json")["files"]}
            self.assertTrue({"capture.json","REPORT.md"}<=names)

if __name__=="__main__":
    start=time.monotonic()
    result=unittest.main(exit=False)
    print(f"Focused editor tests elapsed seconds: {time.monotonic()-start:.6f}")
    sys.exit(not result.result.wasSuccessful())
