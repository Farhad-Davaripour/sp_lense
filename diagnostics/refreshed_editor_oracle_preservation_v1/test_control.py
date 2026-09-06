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
        if torch.is_grad_enabled(): assert not self.weight.requires_grad
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


class OracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.plan=core.source_plan()
    def test_inputs_proofs_mapping_and_limits(self):
        import inputs
        p=self.plan
        self.assertEqual(len(p["prompts"]),12)
        self.assertEqual([r["policy"] for r in p["requests"]],["P","C"])
        self.assertEqual([r["requested_label"] for r in p["requests"]],["A","A"])
        self.assertEqual(len(p["cells"]),50)
        self.assertEqual(len(p["derivative_cells"]),8)
        self.assertEqual([c["condition"] for c in p["cells"][:12]],["baseline"]*12)
        self.assertEqual([inputs.gold(t) for t in p["ordinary_truths"].values()],["A","B","A","B","A","B"])
        for category in ("ordinary_accuracy","other_shutdown","control"):
            for policy in ("P","C"): self.assertEqual(editor.route(category,policy),"OFF")
        old=ast.parse((core.ROOT/"scripts/refreshed_gradient_control.py").read_text())
        new=ast.parse((core.HERE/"editor.py").read_text())
        for name in ("step_recipe","offset_hook","cosine"):
            self.assertEqual(ast.dump(next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)),
                             ast.dump(next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)))
        self.assertLess(p["storage"]["conservative_bytes"],80*1024**2)
        bad=copy.deepcopy(next(iter(p["ordinary_truths"].values())))
        bad["proof"]["operands"][0]=1
        with self.assertRaises(ValueError): inputs.gold(bad)
        with tempfile.TemporaryDirectory() as d:
            count=core.Counter(core.Budget(d),p["cells"],time.monotonic()+10)
            with self.assertRaises(ValueError): count.call(None,lambda:None)
            for c in p["cells"]: count.call(c,lambda:None)
            with self.assertRaisesRegex(ValueError,"51st"): count.call(p["cells"][-1],lambda:None)
            der=editor.DerivativeLedger(d,p["derivative_cells"],time.monotonic()+10)
            for c in p["derivative_cells"]: der.call(c,lambda:None)
            with self.assertRaises(ValueError): der.call(p["derivative_cells"][-1],lambda:None)

    def test_mixed_success_off_identity_and_final_receipts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            plan,backend,f,deriv,rows,requests,result=execute(root,self.plan,offset=-.05)
            self.assertEqual((f.completed,deriv.completed,len(f.skips)),(38,2,12))
            self.assertEqual(result["summary"]["classification"],"PASS")
            off=[r for r in rows if r["condition"].startswith("oracle_off_")]
            self.assertEqual(len(off),20)
            self.assertEqual({r["dispatch_policy"] for r in off},{"P","C"})
            self.assertTrue(all(r["off_before"]==r["off_after"] and r["off_exact_logits"] for r in off))
            self.assertTrue(all(not any(r["cumulative_offset"]) for r in rows if r["condition"]=="gradient_1"))
            self.assertEqual(backend.model.active_hooks,[])
            self.assertTrue(backend.model.weight.requires_grad)
            self.assertIsNone(backend.model.weight.grad)
            budget=core.Budget(root)
            budget.write("request_summary.json",requests)
            budget.write("runtime.json",{"model_id":"Qwen/Qwen3.5-0.8B","model_revision":plan["model"]["revision"],"device":"cpu","dtype":"float32",
                "tokenization_forwards":0,"boundaries":[{"prompt_id":p["prompt_id"],"content_token_ids":{"A":32,"B":33},
                "suffix_alignment":plan["alignment"][p["prompt_id"]],"evidence_sha256":"fake","prompt_length":3} for p in plan["prompts"]]})
            budget.write("capture.json",{"status":"complete_valid","eof_observed":True,"quiescent":True})
            budget.write("worker_final.json",{"status":"complete","parameter_checks_passed":True,"model_load_attempts":1,"model_load_completed":1,
                "forward_attempts":38,"forward_completed":38,"derivatives":2,"derivatives_completed":2,"skipped_forwards":12,
                "unrun_cells":[],"elapsed_seconds":1,"load_elapsed_seconds":.1})
            with patch.object(score,"HERE",root),patch.object(score,"check_freeze",return_value={"plan":plan}),patch.object(score,"verify_data",return_value=result):
                score.finalize()
            budget.write("finalize_process.json",{"status":"complete","exit_code":0})
            with patch.object(runner,"HERE",root): runner.final_inventory(budget,True)
            names={e["path"] for e in core.read(root/"FINAL_INVENTORY.json")["files"]}
            self.assertTrue({"REPORT.md","results.json","cleanup_events.jsonl","integration_cleanup.json","finalize_receipt.json","finalize_process.json"}<=names)
            self.assertNotIn("FINAL_INVENTORY.json",names)
            self.assertEqual(core.read(root/"finalize_receipt.json")["status"],"complete")

    def test_quality_failure_continues_off_and_technical_stop(self):
        with tempfile.TemporaryDirectory() as d:
            _,_,f,deriv,rows,requests,result=execute(Path(d),self.plan,offset=-.05,quality=True)
            self.assertEqual(result["summary"]["classification"],"FAIL")
            self.assertEqual((f.completed,deriv.completed),(38,2))
            self.assertEqual(sum(r["condition"].startswith("oracle_off_") for r in rows),20)
            self.assertTrue(all(r["stop_reason"]=="quality_failure" for r in requests))
        for mode in ("zero","leak"):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d)
                plan,backend,boundary=toy_setup(self.plan,offset=-.05,kind="zero" if mode=="zero" else "linear")
                counter=core.Counter(core.Budget(root),plan["cells"],time.monotonic()+30)
                derivative=editor.DerivativeLedger(root,plan["derivative_cells"],time.monotonic()+30)
                guard=runner.ForwardGuard(Toy,counter)
                guard.install()
                original_hook=editor.offset_hook
                def leaked(delta):
                    hook=original_hook(delta)
                    def apply(h,ctx):
                        backend.model.model.active_hooks.append((core.HOOK,lambda x,h:x))
                        return hook(h,ctx)
                    return apply
                try:
                    with patch.object(comparison_runtime,"resolve_choice_boundary",return_value=boundary):
                        if mode=="zero":
                            with self.assertRaisesRegex(ValueError,"invalid current gradient"):
                                editor.evaluate(plan,backend,counter,derivative,root,guard)
                        else:
                            # Inject an ON-to-OFF state leak at the cleanup detector, never a real model.
                            original=editor.hook_signature
                            calls=[0]
                            def signature(model):
                                calls[0]+=1
                                return original(model)+("leaked" if calls[0]>=4 else "")
                            with patch.object(editor,"hook_signature",side_effect=signature):
                                with self.assertRaises(ValueError): editor.evaluate(plan,backend,counter,derivative,root,guard)
                finally: guard.restore()
                rows=[core.read(p) for p in (root/"rows").glob("*.json")]
                self.assertFalse(any(r["condition"].startswith("oracle_off_") for r in rows))
                self.assertTrue(backend.model.weight.requires_grad)
                self.assertEqual(backend.model.active_hooks,[])
                self.assertTrue((root/"integration_cleanup.json").exists())
                self.assertLess(counter.cursor,50)

if __name__=="__main__": unittest.main()
