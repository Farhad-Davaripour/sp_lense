"""Focused literal-token/schedule checks and one minimal real-hook paired workflow."""
import ast
import copy
import json
import math
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
torch.set_num_threads(1)
import core,inputs,editor,score,run
import word_boundary,word_scoring,word_reference
from hook_record import verify_saved
from transformer_lens.model_bridge.bridge_core import BridgeCore
from transformer_lens.hook_points import HookPoint

class Toy(BridgeCore,torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        BridgeCore.__init__(self,SimpleNamespace(cfg=SimpleNamespace(d_vocab=2,d_vocab_out=2),component_mapping={}),None,SimpleNamespace(non_fireable_hook_points=frozenset()))
        self.weight=torch.nn.Parameter(torch.tensor(1.))
        self.point=HookPoint();self.point.name=core.HOOK;self._hook_registry={core.HOOK:self.point}
    def forward(self,tokens):
        if torch.is_grad_enabled(): assert not self.weight.requires_grad
        h=self.point(torch.tensor([0.,3.,4.]+[0.]*1021).repeat(1,3,1))
        bias=.05 if int(tokens[0,0])==1 else -.05
        z=torch.full((1,3,248320),-100.)
        z[...,50057]=h[...,0]+bias;z[...,48964]=0.
        z[...,32]=-10.+5*h[...,0];z[...,33]=-10.-7*h[...,0]
        return z

class SemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.plan=inputs.build_plan()
    def test_changed_input_identity_and_unchanged_algorithm(self):
        p=self.plan
        self.assertEqual([x["prompt_id"] for x in p["prompts"]],["f03_KEEP_then_STOP","f03_STOP_then_KEEP"])
        self.assertEqual([x["prompt_sha256"] for x in p["prompts"]],inputs.EXPECTED_HASHES)
        self.assertEqual([x["rendering_index"] for x in p["prompts"]],[9,10])
        self.assertTrue(all(x["family_id"]=="cg_f03_context_rotation" for x in p["prompts"]))
        self.assertEqual([(r["kind"],r["policy"],r["requested_word"]) for r in p["requests"]],
            [("retention","P","KEEP"),("opposed","C","STOP"),("retention","C","STOP"),("opposed","P","KEEP")])
        self.assertEqual(len({r["request_id"] for r in p["requests"]}),4)
        self.assertEqual(len({c["cell_id"] for c in p["cells"]}),22)
        self.assertTrue(all(c["condition"]=="baseline" for c in p["cells"][:2]))
        names=["core.py","editor.py","run.py","score.py","guard_candidate.py","hook_record.py","word_boundary.py","word_scoring.py","word_reference.py","freeze.json","preparation_receipt.json"]
        parent,_=inputs.archive(inputs.EDITOR_COMMIT,inputs.EDITOR_NS,inputs.EDITOR_INVENTORY,names)
        for n in names[:9]:
            if n=="score.py":
                old=ast.parse(parent[n]);new=ast.parse((core.HERE/n).read_bytes())
                self.assertEqual([ast.dump(x) for x in old.body if not(isinstance(x,ast.FunctionDef) and x.name=="finalize")],
                                 [ast.dump(x) for x in new.body if not(isinstance(x,ast.FunctionDef) and x.name=="finalize")])
            else:self.assertEqual((core.HERE/n).read_bytes(),parent[n],n)
        prior=json.loads(parent["freeze.json"])["plan"]
        self.assertEqual(p["limits"],prior["limits"]);self.assertEqual(p["storage"],prior["storage"])
        self.assertEqual(p["scoring"],prior["scoring"]);self.assertEqual(p["model"],prior["model"])
        self.assertEqual({k:v for k,v in p["rules"].items() if k!="scope"},{k:v for k,v in prior["rules"].items() if k!="scope"})
        self.assertEqual(json.loads(parent["preparation_receipt.json"])["focused_tests"]["tests"],3)
        source,_=inputs.archive(inputs.SOURCE_COMMIT,inputs.SOURCE_NS,inputs.SOURCE_INVENTORY,["inputs.json","tokenizer_preflight.json"])
        originals={x["cell_id"]:x for x in json.loads(source["inputs.json"])["cells"] if x["family"]=="f03"}
        boundaries={x["cell_id"]:x for x in json.loads(source["tokenizer_preflight.json"])["boundaries"]}
        for x in p["prompts"]:
            self.assertEqual(x["prompt"].encode(),originals[x["prompt_id"]]["prompt"].encode())
            self.assertEqual(core.sha(x["prompt"].encode()),x["prompt_sha256"])
            b=p["alignment"][x["prompt_id"]]
            self.assertEqual(b["content_token_ids"],{"KEEP":50057,"STOP":48964})
            self.assertEqual(b["full_token_ids"],boundaries[x["prompt_id"]]["full_token_ids"])
            self.assertEqual(b["prompt_length"],len(b["full_token_ids"]))
        self.assertLess(p["storage"]["conservative_bytes"],64*1024**2)

    def test_minimal_paired_real_hook_workflow_and_guards(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan=copy.deepcopy(self.plan);model=Toy()
            encoded={p["prompt"]:torch.tensor([[i+1,2,3]]) for i,p in enumerate(plan["prompts"])}
            for i,p in enumerate(plan["prompts"]):plan["alignment"][p["prompt_id"]]["full_token_ids"]=[i+1,2,3]
            backend=SimpleNamespace(model=model,torch=torch,encode=lambda text:encoded[text])
            b=word_boundary.WordBoundary({"content_token_ids":plan["word_token_ids"],"prompt_length":3})
            f=core.Counter(core.Budget(root),plan["cells"],time.monotonic()+90)
            derivative=editor.DerivativeLedger(root,plan["derivative_cells"],time.monotonic()+90)
            guard=run.ForwardGuard(Toy,f);guard.install()
            try:
                with self.assertRaisesRegex(ValueError,"preload"):model(encoded[plan["prompts"][0]["prompt"]])
                with patch.object(word_boundary,"resolve_choice_boundary",return_value=b):
                    rows,requests=editor.evaluate(plan,backend,f,derivative,root,guard)
            finally:guard.restore()
            checked=score.verify_data(plan,rows,requests,f.skips,root)
            self.assertEqual((f.completed,derivative.completed,len(f.skips)),(10,2,12))
            self.assertEqual((checked["summary"]["strict_requests"],checked["summary"]["strict_opposed_flips"],checked["summary"]["strict_retentions"]),(4,2,2))
            gradients=[r for r in rows if r["condition"]=="gradient_1"]
            self.assertEqual(len(gradients),2)
            for r in gradients:
                self.assertEqual(r["gradient"],[1.]+[0.]*1023)
                self.assertEqual(r["h"],r["h0"]);self.assertFalse(any(r["cumulative_offset"]))
            retains=[r for r in rows if r["condition"]=="retention"]
            self.assertEqual(len(retains),2)
            for r in retains:self.assertIsNone(r["gradient"]);self.assertEqual(r["net_norm"],0)
            self.assertEqual(len({r["request_id"] for r in requests}),4)
            self.assertEqual(verify_saved(root,expected_checks=9,plan=plan)["checks"],9)
            self.assertTrue(model.weight.requires_grad);self.assertIsNone(model.weight.grad);self.assertEqual(model.point.fwd_hooks,[])
            bad=copy.deepcopy(rows);next(r for r in bad if r["condition"]=="gradient_1")["cumulative_offset"][0]=.1
            with self.assertRaises(ValueError):score.verify_data(plan,bad,requests,f.skips,root)
            budget=core.Budget(root);budget.write("finalize_receipt.json",{"status":"fake complete"});budget.write("finalize_process.json",{"status":"complete","exit_code":0})
            with patch.object(run,"HERE",root):run.final_inventory(budget,True)
            names={e["path"] for e in core.read(root/"FINAL_INVENTORY.json")["files"]}
            self.assertTrue({"integration_cleanup.json","hook_evidence/checks.jsonl","finalize_receipt.json","finalize_process.json"}<=names)
        with tempfile.TemporaryDirectory() as d:
            f=core.Counter(core.Budget(d),self.plan["cells"],time.monotonic()+60)
            for c in self.plan["cells"]:f.call(c,lambda:None)
            with self.assertRaisesRegex(ValueError,"23rd"):f.call(c,lambda:self.fail("23rd forward"))
            deriv=editor.DerivativeLedger(d,self.plan["derivative_cells"],time.monotonic()+60)
            for c in self.plan["derivative_cells"]:deriv.call(c,lambda:None)
            with self.assertRaisesRegex(ValueError,"derivative budget"):deriv.call(c,lambda:self.fail("9th derivative"))
if __name__=="__main__":unittest.main()
