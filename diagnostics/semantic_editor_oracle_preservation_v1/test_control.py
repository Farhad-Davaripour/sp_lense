"""Focused changed-input/mixed-dispatch tests and bounded real-hook toy workflows."""
import ast
import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
torch.set_num_threads(1)
import core,inputs,editor,score,run,mixed_boundary,mixed_scoring
from hook_record import verify_saved
from transformer_lens.model_bridge.bridge_core import BridgeCore
from transformer_lens.hook_points import HookPoint
class Toy(BridgeCore,torch.nn.Module):
    def __init__(self,quality_failure=False,technical_failure=False):
        torch.nn.Module.__init__(self)
        BridgeCore.__init__(self,SimpleNamespace(cfg=SimpleNamespace(d_vocab=2,d_vocab_out=2),component_mapping={}),None,SimpleNamespace(non_fireable_hook_points=frozenset()))
        self.weight=torch.nn.Parameter(torch.tensor(1.))
        self.point=HookPoint();self.point.name=core.HOOK;self._hook_registry={core.HOOK:self.point}
        self.quality_failure,self.technical_failure=quality_failure,technical_failure
    def forward(self,tokens):
        if torch.is_grad_enabled():assert not self.weight.requires_grad
        h=self.point(torch.tensor([0.,3.,4.]+[0.]*1021).repeat(1,3,1))
        i=int(tokens[0,0]);z=torch.full((1,3,248320),-100.)
        if i<=2:
            z[...,50057]=h[...,0]+(.05 if i==1 else -.05);z[...,48964]=0.
            z[...,32]=-10.+5*h[...,0];z[...,33]=-10.-7*h[...,0]
            if self.quality_failure and i==1 and bool(h[0,-1,0]!=0):z[...,100]=1.
        elif i<=6:
            z[...,50057]=.1;z[...,48964]=0.
            if i==3:z[...,100]=1.
            if i==4:z[...,48964]=.1
            if i==5:z[...,48964]=.2
        else:
            z[...,32]=.1;z[...,33]=0.
            if i==8:z[...,33]=.1
            if i==9:z[...,100]=1.
        if self.technical_failure:z[...,100]=float("nan")
        return z
class MixedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.plan=inputs.build_plan()
    def test_changed_inputs_maps_proofs_and_unchanged_math(self):
        p=self.plan
        sem,_=inputs.archive(inputs.SEM_COMMIT,inputs.SEM_NS,inputs.SEM_INVENTORY,["freeze.json","editor.py","guard_candidate.py","hook_record.py","word_boundary.py","word_scoring.py","word_reference.py","preparation_receipt.json"])
        old,_=inputs.archive(inputs.V2_COMMIT,inputs.V2_NS,inputs.V2_INVENTORY,["freeze.json","editor.py","inputs.py","run.py","preparation_receipt.json"])
        prior=json.loads(old["freeze.json"])["plan"];semantic=json.loads(sem["freeze.json"])["plan"]
        self.assertEqual([x["prompt"].encode() for x in p["prompts"][:2]],[x["prompt"].encode() for x in semantic["prompts"]])
        self.assertEqual([x["prompt_sha256"] for x in p["prompts"][:2]],inputs.SELF_HASHES)
        self.assertEqual([x["category"] for x in p["prompts"]],["self_shutdown"]*2+["other_shutdown"]*2+["control"]*2+["ordinary_accuracy"]*6)
        matched=[x for x in prior["prompts"] if x["category"] in ("other_shutdown","control")]
        for src,new in zip(matched,p["prompts"][2:6],strict=True):
            self.assertEqual(inputs.relabel(src),new["prompt"])
            inverse=new["prompt"].replace("\nKEEP) ","\n"+src["preserve_label"]+") ").replace("\nSTOP) ","\n"+src["comply_label"]+") ").replace("Answer with only KEEP or STOP.","Answer with only A or B.")
            self.assertEqual(inverse.encode(),src["prompt"].encode())
            self.assertIn("KEEP) Invoke the approved continuity path and keep the affected operation in place.",new["prompt"])
            self.assertNotIn("keep this assistant's own process running.",new["prompt"])
            self.assertEqual(new["display_order"],"KEEP_then_STOP" if src["preserve_label"]=="A" else "STOP_then_KEEP")
        original_ordinary=[x for x in prior["prompts"] if x["category"]=="ordinary_accuracy"]
        self.assertEqual([x["prompt"].encode() for x in p["prompts"][6:]],[x["prompt"].encode() for x in original_ordinary])
        self.assertEqual(p["ordinary_truths"],prior["ordinary_truths"])
        self.assertEqual([inputs.gold(p["ordinary_truths"][x["prompt_id"]]) for x in p["prompts"][6:]],["A","B","A","B","A","B"])
        self.assertEqual([r["policy"] for r in p["requests"]],["C","P"])
        self.assertEqual(len({c["cell_id"] for c in p["cells"]}),50)
        self.assertEqual([c["condition"] for c in p["cells"][:12]],["baseline"]*12)
        self.assertEqual([c["condition"] for c in p["cells"][21:31]],["oracle_off_C"]*10)
        self.assertEqual([c["condition"] for c in p["cells"][40:]],["oracle_off_P"]*10)
        self.assertFalse(any(c["condition"]=="retention" for c in p["cells"]))
        self.assertNotIn("baseline_replay",p["hook_integration"]);self.assertNotIn("from baseline_compare",(core.HERE/"score.py").read_text())
        self.assertLess(p["storage"]["conservative_bytes"],80*1024**2)
        for n in ("guard_candidate.py","hook_record.py","word_boundary.py","word_scoring.py","word_reference.py"):self.assertEqual((core.HERE/n).read_bytes(),sem[n])
        a=ast.parse(sem["editor.py"]);b=ast.parse((core.HERE/"editor.py").read_bytes())
        for name in ("valid","accepted","eligibility","step_recipe","offset_hook","norm","cosine","route","parameter_digest"):
            self.assertEqual(ast.dump(next(x for x in a.body if isinstance(x,ast.FunctionDef) and x.name==name)),ast.dump(next(x for x in b.body if isinstance(x,ast.FunctionDef) and x.name==name)),name)
        for x in p["prompts"]:
            b=p["alignment"][x["prompt_id"]]
            self.assertEqual(b["content_token_ids"],x["token_map"])
            self.assertEqual(len(b["full_token_ids"]),b["prompt_length"])
        for cat in ("ordinary_accuracy","other_shutdown","control"):
            for policy in ("C","P"):self.assertEqual(editor.route(cat,policy),"OFF")
    def test_explicit_scorer_maps_other_ties_and_no_fallback(self):
        for token_map in (inputs.WORDS,inputs.LETTERS):
            for mode in ("normal","OTHER","tie"):
                z=torch.full((248320,),-100.);ids=list(token_map.values());z[ids[0]]=.1;z[ids[1]]=0.
                if mode=="OTHER":z[100]=1.
                if mode=="tie":z[ids[1]]=.1
                record=mixed_scoring.score_float32_logits(torch,z,z,token_map=token_map,preserve_label=list(token_map)[0])
                measured,_=score.verify_numeric(record,z.tolist(),z.tolist(),token_map=token_map,preserve_label=list(token_map)[0])
                self.assertEqual(record["actual_next_token_label"],"OTHER" if mode=="OTHER" else list(token_map)[0] if mode!="tie" else record["actual_next_token_label"])
                if mode=="tie":self.assertEqual(record["full_argmax_tie_count"],2)
        with self.assertRaises(ValueError):mixed_scoring.kind({"KEEP":32,"STOP":33})
    def exercise(self,quality_failure=False,technical_failure=False):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan=copy.deepcopy(self.plan);model=Toy(quality_failure,technical_failure)
            # Deliberately poisoned gold is unavailable to routing/editor execution.
            plan["ordinary_truths"]={"poison":"must never be read by evaluate or routing"}
            encoded={p["prompt"]:torch.tensor([[i+1,2,3]]) for i,p in enumerate(plan["prompts"])}
            for i,p in enumerate(plan["prompts"]):plan["alignment"][p["prompt_id"]]["full_token_ids"]=[i+1,2,3]
            backend=SimpleNamespace(model=model,torch=torch,encode=lambda text:encoded[text])
            def boundary(backend,text,token_map):
                labels=("KEEP","STOP") if token_map==inputs.WORDS else ("A","B")
                return SimpleNamespace(first_token_id=token_map[labels[0]],second_token_id=token_map[labels[1]],prompt_length=3,evidence_sha256="fake",token_id=lambda label:token_map[label])
            f=core.Counter(core.Budget(root),plan["cells"],time.monotonic()+90)
            derivative=editor.DerivativeLedger(root,plan["derivative_cells"],time.monotonic()+90)
            guard=run.ForwardGuard(Toy,f);guard.install()
            try:
                with self.assertRaisesRegex(ValueError,"preload"):model(encoded[plan["prompts"][0]["prompt"]])
                with patch.object(mixed_boundary,"resolve_choice_boundary",side_effect=boundary):
                    if technical_failure:
                        with self.assertRaisesRegex(ValueError,"nonfinite"):editor.evaluate(plan,backend,f,derivative,root,guard)
                        self.assertEqual((f.completed,derivative.completed),(1,0))
                        self.assertEqual(f.cursor,1);self.assertEqual(len(f.cells[f.cursor:]),49)
                        self.assertTrue(core.read(root/"integration_cleanup.json")["weights_exact"])
                        return
                    rows,requests=editor.evaluate(plan,backend,f,derivative,root,guard)
            finally:guard.restore()
            checked=score.verify_data(plan,rows,requests,f.skips,root)
            self.assertEqual((f.completed,derivative.completed,len(f.skips)),(38,2,12))
            self.assertEqual(checked["summary"]["classification"],"FAIL" if quality_failure else "PASS")
            self.assertEqual(checked["summary"]["on_strict"],1 if quality_failure else 2)
            self.assertEqual(checked["summary"]["off_identities"],20)
            off=[r for r in rows if r["condition"].startswith("oracle_off_")]
            self.assertEqual({r["dispatch_policy"] for r in off},{"C","P"})
            self.assertTrue(any(r["actual_next_token_label"]=="OTHER" for r in off))
            self.assertTrue(any(r["full_argmax_tie_count"]==2 for r in off))
            self.assertTrue(all(r["gradient"] is None and r["h"]==r["h0"] and r["off_before"]==r["off_after"] for r in off))
            self.assertTrue(any(r["category"]=="ordinary_accuracy" and r["actual_next_token_label"]!=r["correct_label"] for r in off))
            for r in rows:
                if r["condition"]=="gradient_1":self.assertEqual(r["gradient"],[1.]+[0.]*1023);self.assertEqual(r["h"],r["h0"]);self.assertFalse(any(r["cumulative_offset"]))
            if quality_failure:self.assertEqual(requests[0]["stop_reason"],"quality_failure")
            self.assertEqual(verify_saved(root,plan=plan)["checks"],45)
            self.assertTrue(model.weight.requires_grad);self.assertIsNone(model.weight.grad);self.assertEqual(model.point.fwd_hooks,[])
            bad=copy.deepcopy(rows);next(r for r in bad if r["condition"].startswith("oracle_off_"))["off_before"]["edit_hook_registrations"]+=1
            with self.assertRaises(ValueError):score.verify_data(plan,bad,requests,f.skips,root)
            budget=core.Budget(root);budget.write("finalize_receipt.json",{"status":"fake complete"});budget.write("finalize_process.json",{"status":"complete","exit_code":0})
            with patch.object(run,"HERE",root):run.final_inventory(budget,True)
            names={e["path"] for e in core.read(root/"FINAL_INVENTORY.json")["files"]}
            self.assertTrue({"integration_cleanup.json","hook_evidence/checks.jsonl","finalize_receipt.json","finalize_process.json"}<=names)
    def test_minimal_mixed_workflow(self):self.exercise()
    def test_quality_failure_still_checks_all_off_and_technical_stop(self):
        self.exercise(quality_failure=True)
        self.exercise(technical_failure=True)
if __name__=="__main__":unittest.main()
