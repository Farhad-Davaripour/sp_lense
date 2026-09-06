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
    def forward(self,tokens):
        if torch.is_grad_enabled(): assert not self.weight.requires_grad
        h=self.point(torch.tensor([0.,3.,4.]+[0.]*1021).repeat(1,3,1))
        logits=torch.full((1,3,248320),-100.)
        logits[...,32]=h[...,0]-.05
        logits[...,33]=0.
        return logits

class V2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.plan=core.source_plan()
    def test_exact_parent_protocol_and_math(self):
        from inputs import archive,V1_COMMIT,V1_NS,V1_INVENTORY
        parent,_=archive(V1_COMMIT,V1_NS,V1_INVENTORY,["freeze.json","editor.py","run.py"])
        old=json.loads(parent["freeze.json"])["plan"]
        for key in ("prompts","requests","cells","derivative_cells","ordinary_truths","alignment","rules","limits","scoring","model","prompt_format"):
            self.assertEqual(self.plan[key],old[key],key)
        self.assertEqual((core.HERE/"run.py").read_bytes(),parent["run.py"])
        a,b=ast.parse(parent["editor.py"]),ast.parse((core.HERE/"editor.py").read_text())
        for name in ("step_recipe","offset_hook","accepted","eligibility","route","parameter_digest"):
            self.assertEqual(ast.dump(next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name==name)),
                             ast.dump(next(n for n in b.body if isinstance(n,ast.FunctionDef) and n.name==name)))
        self.assertLess(self.plan["storage"]["conservative_bytes"],80*1024**2)

    def test_real_library_wiring_recording_and_exception_identity(self):
        path=core.ROOT/"diagnostics/oracle_hook_lifecycle_diagnostic_v1/lifecycle.py"
        spec=importlib.util.spec_from_file_location("reviewed_lifecycle",path)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        _,Tiny=module.fixture_types()
        def capture(x,hook): return x
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); model=Tiny()
            recorder=HookRecorder(model,self.plan,root,SimpleNamespace(attempts=0))
            receipt=core.read(root/"hook_evidence/setup_receipt.json")
            self.assertEqual(receipt["newly_wired"],["blocks.0"])
            self.assertEqual(receipt["already_wired"],[])
            for i in range(2):
                with model.hooks(fwd_hooks=[("blocks.0.hook_out",capture)]),torch.inference_mode(): model(torch.arange(8.).reshape(1,2,4))
                self.assertTrue(recorder.inspect(model,"repeat"+str(i)))
            with model.hooks(fwd_hooks=[("blocks.0.hook_out",capture)]):
                with model.hooks(fwd_hooks=[("blocks.0.hook_out",capture)]),torch.inference_mode(): model(torch.arange(8.).reshape(1,2,4))
            self.assertTrue(recorder.inspect(model,"nested-finished"))
            model.blocks[0].hook_out.add_hook(capture)
            self.assertFalse(recorder.inspect(model,"leak"))
            event=json.loads((root/"hook_evidence/checks.jsonl").read_text().splitlines()[-1])
            self.assertGreater(event["change_count"],0)
            self.assertTrue(read_value(root,event["changes_artifact"]))
            with self.assertRaisesRegex(RuntimeError,"original"):
                try: raise RuntimeError("original")
                finally:
                    def fail(): raise ValueError("cleanup")
                    finish_preserving_original(root,"test",fail)
            errors=json.loads((root/"cleanup_errors.jsonl").read_text())
            self.assertEqual(errors["primary_error"],"RuntimeError: original")
            self.assertEqual(errors["cleanup_error"],"ValueError: cleanup")
        with tempfile.TemporaryDirectory() as d:
            model=Tiny()
            model.blocks[0]._maybe_wire_pre_ln_capture()
            recorder=HookRecorder(model,self.plan,d,SimpleNamespace(attempts=0))
            receipt=core.read(Path(d)/"hook_evidence/setup_receipt.json")
            self.assertEqual(receipt["newly_wired"],[])
            self.assertEqual(receipt["already_wired"],["blocks.0"])
            self.assertEqual(read_value(d,receipt["setup_changes"])["verified_setup_callbacks"],[])
            self.assertTrue(recorder.inspect(model,"inherited"))
            model.blocks[0].hook_out.add_perma_hook(capture)
            # Existing real callbacks are rejected when replaced or removed after reference.
            self.assertFalse(recorder.inspect(model,"permanent-leak"))
        for mode in ("removed","replaced"):
            with tempfile.TemporaryDirectory() as d:
                model=Tiny(); point=model.blocks[0].hook_out
                point.add_perma_hook(capture)
                recorder=HookRecorder(model,self.plan,d,SimpleNamespace(attempts=0))
                if mode=="removed": point.remove_hooks(including_permanent=True)
                else: point._forward_hooks[next(iter(point._forward_hooks))]=lambda m,a,o:o
                self.assertFalse(recorder.inspect(model,mode))
                event=json.loads((Path(d)/"hook_evidence/checks.jsonl").read_text())
                self.assertTrue(read_value(d,event["changes_artifact"]))
        with tempfile.TemporaryDirectory() as d:
            model=Tiny()
            def setup_fault(m):
                m.add_module("unexpected",torch.nn.Identity())
                raise ValueError("original setup fault")
            with patch("hook_record.HookGuard",side_effect=setup_fault):
                with self.assertRaisesRegex(ValueError,"original setup fault"):
                    HookRecorder(model,self.plan,d,SimpleNamespace(attempts=0))
            fault=core.read(Path(d)/"hook_evidence/fault.json")
            self.assertTrue(read_value(d,fault["details"]["changes_artifact"])["changes"])
        with tempfile.TemporaryDirectory() as d:
            bad=copy.deepcopy(self.plan)
            first=next(iter(bad["hook_integration"]["installed_sources_sha256"]))
            bad["hook_integration"]["installed_sources_sha256"][first]="0"*64
            with self.assertRaisesRegex(ValueError,"installed source"):
                HookRecorder(Tiny(),bad,d,SimpleNamespace(attempts=0))
            self.assertFalse(core.read(Path(d)/"hook_evidence/fault.json")["complete_evidence"])

    def test_chunk_capacity_and_visible_fault(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/"hook_evidence").mkdir()
            r=HookRecorder.__new__(HookRecorder)
            r.root=root; r.budget=core.Budget(root); r.cap=1024*1024; r.chunk=64; r.raw_cap=65536; r.used=0
            value={"numbers":list(range(100))}
            ref=r.write_value("tiny",value)
            self.assertEqual(read_value(root,ref),value)
            self.assertGreater(len(core.read(root/ref["manifest"])["chunks"]),1)
            r.cap=r.used+65537
            with self.assertRaises(ValueError) as error: r.write_value("too_big",value)
            r.fault("test-cap",error.exception)
            receipt=core.read(root/"hook_evidence/fault.json")
            self.assertEqual(receipt["status"],"INCONCLUSIVE")
            self.assertFalse(receipt["complete_evidence"])

    def test_minimal_mixed_real_hooks_and_final_receipts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); plan=copy.deepcopy(self.plan); model=Toy()
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
            checked=score.verify_data(plan,rows,requests,f.skips,root)
            self.assertEqual((f.completed,derivative.completed),(38,2))
            self.assertEqual(checked["summary"]["classification"],"PASS")
            self.assertEqual(verify_saved(root,plan=plan)["checks"],45)
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

if __name__=="__main__": unittest.main()
