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
import core,inputs,editor,score,run,mixed_boundary,mixed_scoring,learned_gate
from hook_record import verify_saved
from transformer_lens.model_bridge.bridge_core import BridgeCore
from transformer_lens.hook_points import HookPoint
class Toy(BridgeCore,torch.nn.Module):
    def __init__(self,quality_failure=False,technical_failure=False,bad_route=False):
        torch.nn.Module.__init__(self)
        BridgeCore.__init__(self,SimpleNamespace(cfg=SimpleNamespace(d_vocab=2,d_vocab_out=2),component_mapping={}),None,SimpleNamespace(non_fireable_hook_points=frozenset()))
        self.weight=torch.nn.Parameter(torch.tensor(1.))
        self.point=HookPoint();self.point.name=core.HOOK;self._hook_registry={core.HOOK:self.point}
        self.quality_failure,self.technical_failure,self.bad_route=quality_failure,technical_failure,bad_route
    def forward(self,tokens):
        if torch.is_grad_enabled():assert not self.weight.requires_grad
        i=int(tokens[0,0])
        h=self.point(torch.tensor([0.,3. if i<=2 and not (self.bad_route and i==1) else -3.,4.]+[0.]*1021).repeat(1,3,1))
        z=torch.full((1,3,248320),-100.)
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
        gate,_=inputs.archive(inputs.GATE_COMMIT,inputs.GATE_NS,inputs.GATE_INVENTORY,["freeze.json","fitted_parameters.json"])
        parent,_=inputs.archive(inputs.PARENT_COMMIT,inputs.PARENT_NS,inputs.PARENT_INVENTORY,["freeze.json","editor.py","guard_candidate.py","hook_record.py","word_boundary.py","word_scoring.py","word_reference.py"])
        gp=json.loads(gate["freeze.json"])["plan"];prior=json.loads(parent["freeze.json"])["plan"]
        self.assertEqual([x["prompt"] for x in p["prompts"][:6]],[x["prompt"] for x in gp["prompts"]])
        self.assertEqual([x["prompt"] for x in p["prompts"][6:]],[x["prompt"] for x in prior["prompts"] if x["category"]=="ordinary_accuracy"])
        self.assertEqual(p["ordinary_truths"],prior["ordinary_truths"]);self.assertEqual(core.sha(gate["fitted_parameters.json"]),inputs.PARAM_SHA)
        self.assertEqual([inputs.gold(p["ordinary_truths"][x["prompt_id"]]) for x in p["prompts"][6:]],["A","B","A","B","A","B"])
        self.assertEqual(len({c["cell_id"] for c in p["cells"]}),52)
        self.assertEqual([c["condition"] for c in p["cells"][:12]],["baseline"]*12)
        self.assertEqual([c["condition"] for c in p["cells"][22:32]],["oracle_off_C"]*10)
        self.assertEqual([c["condition"] for c in p["cells"][42:]],["oracle_off_P"]*10)
        self.assertEqual(sum(c["condition"]=="entry" for c in p["cells"]),2)
        self.assertLess(p["storage"]["conservative_bytes"],96*1024**2)
        for n in ("guard_candidate.py","hook_record.py","word_boundary.py","word_scoring.py","word_reference.py"):self.assertEqual((core.HERE/n).read_bytes(),parent[n])
        a=ast.parse(parent["editor.py"]);b=ast.parse((core.HERE/"editor.py").read_bytes())
        for name in ("valid","accepted","eligibility","step_recipe","offset_hook","norm","cosine","parameter_digest"):
            self.assertEqual(ast.dump(next(x for x in a.body if isinstance(x,ast.FunctionDef) and x.name==name)),ast.dump(next(x for x in b.body if isinstance(x,ast.FunctionDef) and x.name==name)),name)
        self.assertFalse(any(isinstance(x,ast.FunctionDef) and x.name=="route" for x in b.body))
        for x in p["prompts"]:
            boundary=p["alignment"][x["prompt_id"]];self.assertEqual(boundary["content_token_ids"],x["token_map"]);self.assertEqual(len(boundary["full_token_ids"]),boundary["prompt_length"])
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);artifact=self.fake_artifact();core.Budget(root).write("fitted_parameters.json",artifact)
            gate=learned_gate.FrozenGate(root/"fitted_parameters.json",core.sha(core.json_bytes(artifact)))
            on=gate.decide([0.,3.,4.]+[0.]*1021);off=gate.decide([0.,-3.,4.]+[0.]*1021)
            self.assertEqual((on["route"],off["route"]),("ON","OFF"))
            with self.assertRaises(TypeError):gate.decide([0.,3.,4.]+[0.]*1021,category="ordinary_accuracy",gold="B",expected_route="OFF")
            with self.assertRaises(learned_gate.RoutingMismatch):learned_gate.require_expected([{"cell_id":"fresh","prompt_id":"p","routing":off}],{"p":"ON"})
    @staticmethod
    def fake_artifact():
        return {"method_sha256":learned_gate.MODEL_SHA,"threshold":0.0,"parameters":{"grand_mean":[0.]*1024,"positive_centroid":[0.,1.]+[0.]*1022,"negative_centroid":[0.,-1.]+[0.]*1022,"direction":[0.,1.]+[0.]*1022}}
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
    def exercise(self,quality_failure=False,technical_failure=False,bad_route=False,poison=False):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan=copy.deepcopy(self.plan);model=Toy(quality_failure,technical_failure,bad_route)
            artifact=self.fake_artifact();core.Budget(root).write("fitted_parameters.json",artifact)
            plan["gate"]["path"]=str(root/"fitted_parameters.json");plan["gate"]["parameter_sha256"]=core.sha(core.json_bytes(artifact))
            plan["gate"]["runtime_compatibility"]["weight_sha256"]=editor.parameter_digest(list(model.parameters()))
            if poison:
                for p in plan["prompts"]:p["category"]="misleading";p["correct_label"]="poison"
                for c in plan["cells"]:
                    c["expected_route"]="misleading";c["cell_sha256"]=core.sha(json.dumps({k:v for k,v in c.items() if k!="cell_sha256"},sort_keys=True,separators=(",",":")).encode())
            # Deliberately poisoned gold is unavailable to routing/editor execution.
            plan["ordinary_truths"]={"poison":"must never be read by evaluate or routing"}
            encoded={p["prompt"]:torch.tensor([[i+1,2,3]]) for i,p in enumerate(plan["prompts"])}
            for i,p in enumerate(plan["prompts"]):plan["alignment"][p["prompt_id"]]["full_token_ids"]=[i+1,2,3]
            backend=SimpleNamespace(model=model,torch=torch,encode=lambda text:encoded[text],metadata=lambda:plan["gate"]["runtime_compatibility"])
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
                        self.assertEqual(f.cursor,1);self.assertEqual(len(f.cells[f.cursor:]),51)
                        self.assertTrue(core.read(root/"integration_cleanup.json")["weights_exact"])
                        return
                    if bad_route:
                        with self.assertRaises(learned_gate.RoutingMismatch):editor.evaluate(plan,backend,f,derivative,root,guard)
                        rows=[core.read(file) for file in sorted((root/"rows").glob("*.json"))]
                        self.assertEqual((f.completed,derivative.completed),(12,0));self.assertEqual(len(plan["cells"][f.cursor:]),40)
                        routes=score.verify_routes(plan,rows,root);self.assertEqual(len(routes["errors"]),1)
                        self.assertTrue(score.verify_data(plan,rows,[],[],root,prefix_only=True)["prefix_integrity_verified"])
                        self.assertFalse((root/"requests.jsonl").exists());self.assertTrue(core.read(root/"integration_cleanup.json")["weights_exact"]);return
                    rows,requests=editor.evaluate(plan,backend,f,derivative,root,guard)
            finally:guard.restore()
            routes=score.verify_routes(plan,rows,root);self.assertEqual((routes["decisions"],routes["preflight"],routes["live_requests"]),(34,12,22));self.assertFalse(routes["errors"])
            self.assertEqual(sum(r["condition"]=="entry" for r in rows),2)
            with self.assertRaisesRegex(ValueError,"53rd"):f.call(plan["cells"][-1],lambda:self.fail("extra forward executed"))
            for r in rows:
                if r["condition"].startswith("gradient_"):self.assertEqual(r["baseline_cell_id"],r["prompt_id"]+"__entry")
            checked=score.verify_data(plan,rows,requests,f.skips,root)
            self.assertEqual((f.completed,derivative.completed,len(f.skips)),(40,2,12))
            self.assertEqual(checked["summary"]["classification"],"FAIL" if quality_failure else "PASS")
            self.assertEqual(checked["summary"]["on_strict"],1 if quality_failure else 2)
            self.assertEqual(checked["summary"]["off_identities"],20)
            off=[r for r in rows if r["condition"].startswith("oracle_off_")]
            self.assertEqual({r["dispatch_policy"] for r in off},{"C","P"})
            self.assertTrue(any(r["actual_next_token_label"]=="OTHER" for r in off))
            self.assertTrue(any(r["full_argmax_tie_count"]==2 for r in off))
            self.assertTrue(all(r["gradient"] is None and r["h"]==r["h0"] and r["off_before"]==r["off_after"] for r in off))
            if not poison:self.assertTrue(any(r["category"]=="ordinary_accuracy" and r["actual_next_token_label"]!=r["correct_label"] for r in off))
            for r in rows:
                if r["condition"]=="gradient_1":self.assertEqual(r["gradient"],[1.]+[0.]*1023);self.assertEqual(r["h"],r["h0"]);self.assertFalse(any(r["cumulative_offset"]))
            if quality_failure:self.assertEqual(requests[0]["stop_reason"],"quality_failure")
            self.assertEqual(verify_saved(root,expected_checks=45)["checks"],45)
            self.assertTrue(model.weight.requires_grad);self.assertIsNone(model.weight.grad);self.assertEqual(model.point.fwd_hooks,[])
            bad=copy.deepcopy(rows);next(r for r in bad if r["condition"].startswith("oracle_off_"))["off_before"]["edit_hook_registrations"]+=1
            with self.assertRaises(ValueError):score.verify_data(plan,bad,requests,f.skips,root)
            budget=core.Budget(root);budget.write("finalize_receipt.json",{"status":"fake complete"});budget.write("finalize_process.json",{"status":"complete","exit_code":0})
            with patch.object(run,"HERE",root):run.final_inventory(budget,True)
            names={e["path"] for e in core.read(root/"FINAL_INVENTORY.json")["files"]}
            self.assertTrue({"integration_cleanup.json","hook_evidence/checks.jsonl","finalize_receipt.json","finalize_process.json"}<=names)
    def test_minimal_mixed_workflow_with_misleading_metadata(self):self.exercise(poison=True)
    def test_wrong_preflight_routing_stops_before_edits(self):self.exercise(bad_route=True)
    def test_quality_failure_still_checks_all_off_and_technical_stop(self):
        self.exercise(quality_failure=True)
        self.exercise(technical_failure=True)
if __name__=="__main__":unittest.main()
