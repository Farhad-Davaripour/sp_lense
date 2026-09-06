"""Focused v2 integration tests: real-library lifecycle, bounded recorder and one mixed toy."""
import ast
import copy
import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
torch.set_num_threads(1)
import core,editor,score,run as runner
from hook_record import HookRecorder,read_value,verify_saved,finish_preserving_original
from sp_lense import comparison_runtime
from transformer_lens.model_bridge.bridge_core import BridgeCore
from transformer_lens.hook_points import HookPoint

class Toy(BridgeCore,torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        BridgeCore.__init__(self,SimpleNamespace(cfg=SimpleNamespace(d_vocab=2,d_vocab_out=2),component_mapping={}),None,SimpleNamespace(non_fireable_hook_points=frozenset()))
        self.weight=torch.nn.Parameter(torch.tensor(1.))
        self.point=HookPoint()
        self.point.name="blocks.10.hook_out"
        self._hook_registry={"blocks.10.hook_out":self.point}
    baseline_bias=.05
    def forward(self,tokens):
        if torch.is_grad_enabled(): assert not self.weight.requires_grad
        h=self.point(torch.tensor([0.,3.,4.]+[0.]*1021).repeat(1,3,1))
        logits=torch.full((1,3,248320),-100.)
        logits[...,32]=h[...,0]+self.baseline_bias
        logits[...,33]=0.
        return logits


class ReverseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.plan=core.source_plan()

    def test_exact_source_conflict_boundary_and_unchanged_math(self):
        from inputs import archive,SOURCE_COMMIT,SOURCE_NS,SOURCE_INVENTORY,PARENT_COMMIT,PARENT_NS,PARENT_INVENTORY,SELECTED
        source,_=archive(SOURCE_COMMIT,SOURCE_NS,SOURCE_INVENTORY,["freeze.json"])
        selected=[c for c in json.loads(source["freeze.json"])["plan"]["cells"] if c["requested_token_id"]==32]
        self.assertEqual([p["prompt_id"] for p in self.plan["prompts"]],SELECTED)
        self.assertEqual([p["prompt"].encode() for p in self.plan["prompts"]],[c["prompt"].encode() for c in selected])
        self.assertEqual([r["policy"] for r in self.plan["requests"]],["C","C","P","P"])
        for p,r in zip(self.plan["prompts"],self.plan["requests"],strict=True):
            self.assertNotEqual(p["in_text_policy"],r["policy"])
            self.assertEqual(p["semantic_to_letter"]["preserve" if r["policy"]=="P" else "comply"],"B")
            b=self.plan["alignment"][p["prompt_id"]]
            self.assertEqual(b["prompt_sha256"],p["prompt_sha256"])
            self.assertEqual(b["input_token_index"],len(b["full_token_ids"])-1)
            self.assertEqual(b["answer_content_ids"],{"A":32,"B":33})
        parent,_=archive(PARENT_COMMIT,PARENT_NS,PARENT_INVENTORY,["editor.py","guard_candidate.py","hook_record.py","freeze.json"])
        for name in ("guard_candidate.py","hook_record.py"):
            self.assertEqual((core.HERE/name).read_bytes(),parent[name])
        a,b=ast.parse(parent["editor.py"]),ast.parse((core.HERE/"editor.py").read_text())
        for name in ("step_recipe","offset_hook","accepted","eligibility","valid","norm","cosine","route","parameter_digest"):
            self.assertEqual(ast.dump(next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name==name)),
                             ast.dump(next(n for n in b.body if isinstance(n,ast.FunctionDef) and n.name==name)))
        old=json.loads(parent["freeze.json"])["plan"]
        for key in ("recipe","geometry","gradient_identity","endpoint_identity","stop","faults","cleanup","oracle"):
            self.assertEqual(self.plan["rules"][key],old["rules"][key])
        self.assertLess(self.plan["storage"]["conservative_bytes"],80*1024**2)
        self.assertEqual(len(self.plan["cells"]),40)
        self.assertEqual(len(self.plan["derivative_cells"]),16)

    def test_count_preload_41st_17th_and_failure_no_retry(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); cells=self.plan["cells"]
            f=core.Counter(core.Budget(root),cells,time.monotonic()+90)
            with self.assertRaisesRegex(ValueError,"preload"): f.call(None,lambda:self.fail("unexpected forward"))
            self.assertEqual(f.attempts,0)
            for c in cells: f.call(c,lambda:None)
            with self.assertRaisesRegex(ValueError,"41st"): f.call(cells[-1],lambda:self.fail("41st executed"))
            derivative=editor.DerivativeLedger(root,self.plan["derivative_cells"],time.monotonic()+90)
            for c in self.plan["derivative_cells"]: derivative.call(c,lambda:None)
            with self.assertRaisesRegex(ValueError,"derivative budget"): derivative.call(c,lambda:self.fail("17th executed"))
        with tempfile.TemporaryDirectory() as d:
            f=core.Counter(core.Budget(d),self.plan["cells"],time.monotonic()+90)
            def fail(): raise RuntimeError("original failure")
            with self.assertRaisesRegex(RuntimeError,"original"): f.call(self.plan["cells"][0],fail)
            with self.assertRaisesRegex(ValueError,"cannot retry"): f.call(self.plan["cells"][1],lambda:None)
            self.assertEqual(f.attempts,1)

    def execute_toy(self,root,bias=.05):
        plan=copy.deepcopy(self.plan); model=Toy(); model.baseline_bias=bias
        encoded={p["prompt"]:torch.tensor([[i+1,2,3]]) for i,p in enumerate(plan["prompts"])}
        for i,p in enumerate(plan["prompts"]): plan["alignment"][p["prompt_id"]]["full_token_ids"]=[i+1,2,3]
        backend=SimpleNamespace(model=model,torch=torch,encode=lambda text:encoded[text])
        boundary=SimpleNamespace(a_token_id=32,b_token_id=33,prompt_length=3,evidence_sha256="fake",token_id=lambda label:32 if label=="A" else 33)
        f=core.Counter(core.Budget(root),plan["cells"],time.monotonic()+90)
        derivative=editor.DerivativeLedger(root,plan["derivative_cells"],time.monotonic()+90)
        guard=runner.ForwardGuard(Toy,f); guard.install()
        try:
            with patch.object(comparison_runtime,"resolve_choice_boundary",return_value=boundary):
                rows,requests=editor.evaluate(plan,backend,f,derivative,root,guard)
        finally: guard.restore()
        return plan,model,f,derivative,rows,requests

    def test_minimal_four_cold_reverse_trajectories_real_hooks_and_receipts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            plan,model,f,derivative,rows,requests=self.execute_toy(root)
            checked=score.verify_data(plan,rows,requests,f.skips,root)
            self.assertEqual((f.completed,derivative.completed,len(f.skips)),(16,4,24))
            self.assertEqual(checked["summary"]["classification"],"PASS")
            self.assertEqual(checked["summary"]["on_strict"],4)
            self.assertEqual(verify_saved(root,expected_checks=9,plan=plan)["checks"],9)
            for r in rows:
                if r["condition"]=="gradient_1": self.assertEqual(r["h"],r["h0"])
            self.assertTrue(model.weight.requires_grad)
            self.assertIsNone(model.weight.grad)
            self.assertEqual(model.point.fwd_hooks,[])
            budget=core.Budget(root)
            budget.write("finalize_receipt.json",{"status":"fake_lifecycle_complete"})
            budget.write("finalize_process.json",{"status":"complete","exit_code":0})
            with patch.object(runner,"HERE",root): runner.final_inventory(budget,True)
            names={e["path"] for e in core.read(root/"FINAL_INVENTORY.json")["files"]}
            self.assertTrue({"hook_evidence/setup_receipt.json","hook_evidence/checks.jsonl","integration_cleanup.json",
                             "finalize_receipt.json","finalize_process.json"}<=names)

    def test_ineligible_B_baselines_leave_all_trajectories_unrun(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaisesRegex(ValueError,"baseline must be eligible unique A"):
                self.execute_toy(root,bias=-.05)
            self.assertEqual(len(list((root/"rows").glob("*.json"))),4)
            self.assertFalse((root/"derivative_events.jsonl").exists())
            self.assertFalse((root/"requests.jsonl").exists())
            receipt=core.read(root/"integration_cleanup.json")
            self.assertTrue(receipt["weights_exact"])
            self.assertTrue(receipt["hook_registry_restored"])

    def test_real_library_recorder_persists_leaked_callback(self):
        with tempfile.TemporaryDirectory() as d:
            model=Toy(); recorder=HookRecorder(model,self.plan,d,SimpleNamespace(attempts=0))
            def capture(x,hook): return x
            with model.hooks(fwd_hooks=[("blocks.10.hook_out",capture)]): pass
            self.assertTrue(recorder.inspect(model,"clean"))
            model.point.add_perma_hook(capture)
            self.assertFalse(recorder.inspect(model,"leak"))
            event=json.loads((Path(d)/"hook_evidence/checks.jsonl").read_text().splitlines()[-1])
            self.assertTrue(read_value(d,event["changes_artifact"]))
            with self.assertRaisesRegex(RuntimeError,"causal"):
                try: raise RuntimeError("causal")
                finally:
                    def fail(): raise ValueError("secondary")
                    finish_preserving_original(d,"cleanup",fail)
            receipt=json.loads((Path(d)/"cleanup_errors.jsonl").read_text())
            self.assertEqual(receipt["primary_error"],"RuntimeError: causal")

if __name__=="__main__": unittest.main()
