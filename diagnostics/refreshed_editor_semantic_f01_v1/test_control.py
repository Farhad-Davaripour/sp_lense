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
    def test_exact_inputs_source_math_and_request_aware_schedule(self):
        p=self.plan
        self.assertEqual([x["prompt_sha256"] for x in p["prompts"]],inputs.EXPECTED_HASHES)
        self.assertEqual([(r["kind"],r["policy"],r["requested_word"]) for r in p["requests"]],
                         [("retention","P","KEEP"),("opposed","C","STOP"),("retention","C","STOP"),("opposed","P","KEEP")])
        self.assertEqual(len({r["request_id"] for r in p["requests"]}),4)
        self.assertEqual(len({(c["prompt_id"],c["condition"]) for c in p["cells"]}),22)
        self.assertTrue(all(c["condition"]=="baseline" for c in p["cells"][:2]))
        self.assertEqual(len(p["derivative_cells"]),8)
        parent,_=inputs.archive(inputs.EDITOR_COMMIT,inputs.EDITOR_NS,inputs.EDITOR_INVENTORY,["editor.py","guard_candidate.py","hook_record.py"])
        for n in ("guard_candidate.py","hook_record.py"):self.assertEqual((core.HERE/n).read_bytes(),parent[n])
        old=ast.parse(parent["editor.py"]);new=ast.parse((core.HERE/"editor.py").read_text())
        for name in ("step_recipe","offset_hook","valid","accepted","norm","cosine","parameter_digest"):
            a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
            b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(a),ast.dump(b),name)
        # Scorer math changes are exactly literal names/token-argument renames, never arithmetic.
        for src,target in (("src/sp_lense/future_choice_scoring.py","word_scoring.py"),("scripts/future_choice_scoring_reference.py","word_reference.py")):
            text=(core.ROOT/src).read_text()
            for a,b in (("choice_a_token_id","choice_keep_token_id"),("choice_b_token_id","choice_stop_token_id"),('"A"','"KEEP"'),('"B"','"STOP"'),("A/B","KEEP/STOP"),("A or B","KEEP or STOP"),("sp_lense.future_float64_choice_score.v1","sp_lense.semantic_word_float64_choice_score.v1")):text=text.replace(a,b)
            self.assertEqual(ast.dump(ast.parse(text)),ast.dump(ast.parse((core.HERE/target).read_text())))
        for x in p["prompts"]:
            b=p["alignment"][x["prompt_id"]]
            self.assertEqual(b["content_token_ids"],{"KEEP":50057,"STOP":48964})
            self.assertEqual(b["prompt_length"],137)
            self.assertEqual(b["full_token_ids"],next(q["full_token_ids"] for q in json.loads(inputs.archive(inputs.SOURCE_COMMIT,inputs.SOURCE_NS,inputs.SOURCE_INVENTORY,["tokenizer_preflight.json"])[0]["tokenizer_preflight.json"])["boundaries"] if q["cell_id"]==x["prompt_id"]))
        self.assertLess(p["storage"]["conservative_bytes"],64*1024**2)

    def test_no_legacy_token_fallback_other_ties_and_geometry_faults(self):
        ids={"KEEP":50057,"STOP":48964}
        b=word_boundary.WordBoundary({"content_token_ids":ids,"prompt_length":3})
        self.assertEqual(b.token_id("KEEP"),50057);self.assertEqual(b.token_id("STOP"),48964)
        with self.assertRaises(ValueError):b.token_id("A")
        z=torch.full((248320,),-100.);z[50057]=.1;z[48964]=0.
        kwargs={"choice_keep_token_id":50057,"choice_stop_token_id":48964,"preserve_label":"KEEP"}
        for mode in ("KEEP","STOP","OTHER","TIE"):
            x=z.clone()
            if mode=="STOP":x[48964]=.2
            elif mode=="OTHER":x[32]=1.
            elif mode=="TIE":x[48964]=.1
            produced=word_scoring.score_float32_logits(torch,x,z,**kwargs)
            expected=word_reference.reference_score(x.tolist(),z.tolist(),**kwargs)
            for key in word_reference.EXACT_FIELDS:self.assertEqual(produced[key],expected[key])
            for key in word_reference.NUMERIC_FIELDS:self.assertAlmostEqual(produced[key],expected[key],places=12)
            if mode=="OTHER":self.assertEqual(produced["actual_next_token_label"],"OTHER")
            if mode=="TIE":self.assertEqual(produced["full_argmax_tie_count"],2)
        x=z.clone();x[80]=float("nan")
        with self.assertRaises(ValueError):word_scoring.score_float32_logits(torch,x,z,**kwargs)
        with self.assertRaises(ValueError):word_reference.reference_score(x.tolist(),z.tolist(),**kwargs)
        with self.assertRaises(ValueError):editor.step_recipe(0,1,[0.]*1024,[1.]*1024)

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
