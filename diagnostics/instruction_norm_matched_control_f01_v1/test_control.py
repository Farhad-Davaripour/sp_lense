"""Focused changed-input, guard, mask, arithmetic, scoring and lifecycle regressions."""
import array
import math
import sys
import tempfile
import time
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run as runner
import score as scorer
from core import HERE, Budget, Counter, read, sha, source_plan
from construct import SOURCES, authenticate, load_matrix, scaled_offset
from transfer import f32, norm, apply_values, verify_fixed, verify_pair, baseline_match, patch_fixed, write_capture, load_capture

class Vector(list):
    def __add__(self, other):
        return Vector(f32(a+b) for a, b in zip(self, other, strict=True))


class Window(list):
    def __add__(self, other):
        return Window(Vector(a)+Vector(b) for a,b in zip(self,other,strict=True))


class Matrix:
    device = "cpu"
    def __init__(self, rows):
        self.rows = [Vector(r) for r in rows]
    def clone(self):
        return Matrix(self.rows)
    def __getitem__(self, key):
        return Window(self.rows[key[1]]) if isinstance(key[1],slice) else self.rows[key[1]]
    def __setitem__(self, key, value):
        self.rows[key[1]] = [Vector(r) for r in value] if isinstance(key[1],slice) else Vector(value)



class ControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=source_plan()

    def test_source_authentication_deterministic_scaling_and_pairing(self):
        with self.assertRaisesRegex(ValueError,"authentication"):
            authenticate(SOURCES["last"][0],SOURCES["last"][1],"0"*64)
        plan=self.plan
        self.assertEqual([c["kind"] for c in plan["cells"]],["neutral"]*4+["control"]*8+["suffix"]*8)
        self.assertEqual(len(plan["prompts"]),4)
        self.assertEqual([c["requested_label"] for c in plan["cells"][4:12]],list("ABABBABA"))
        for c,s in zip(plan["cells"][4:12],plan["cells"][12:],strict=True):
            self.assertEqual((c["pair_id"],c["prompt_id"],c["requested_label"]),(s["pair_id"],s["prompt_id"],s["requested_label"]))
            self.assertEqual(plan["candidates"][c["cell_id"]]["mask"],[136])
            self.assertEqual(plan["candidates"][s["cell_id"]]["mask"],list(range(72,137)))
        pre=[[.5]*1024,[.5]*1024]
        post=[[f32(.55)]*1024,[f32(.6)]*1024]
        first,scaling=scaled_offset(pre,post,.1)
        self.assertEqual((first,scaling),scaled_offset(pre,post,.1))
        self.assertEqual(scaling["scale"],.1/scaling["archived_suffix_actual_frobenius"])
        for p,q,L in [(pre,pre,.1),(pre,post,float("nan")),(pre,[[float("nan")]*1024]*2,.1)]:
            with self.assertRaises(ValueError):
                scaled_offset(p,q,L)

    def test_twenty_same_process_guard_no_preload_or_21st(self):
        with tempfile.TemporaryDirectory() as d:
            counter=Counter(Budget(d),self.plan["cells"],time.monotonic()+5)
            calls=[]
            class Fake:
                def forward(self): calls.append(1)
            guard=runner.ForwardGuard(Fake,counter)
            guard.install()
            try:
                model=Fake()
                with self.assertRaises(ValueError): model.forward()
                for cell in self.plan["cells"]:
                    guard.cell=cell
                    model.forward()
                with self.assertRaisesRegex(ValueError,"21st"): model.forward()
            finally: guard.restore()
            self.assertEqual((counter.attempted,counter.completed,len(calls)),(20,20,20))

    def test_masks_float32_norm_matching_and_faults(self):
        original=Matrix([[1.]*1024,[.5]*1024,[.5]*1024])
        delta=[[f32(.001)]*1024]
        torch=SimpleNamespace(float32="float32",tensor=lambda rows,**kw:Window(Vector(row) for row in rows))
        changed=patch_fixed(original,2,delta,torch)
        self.assertEqual(changed.rows[:2],original.rows[:2])
        self.assertNotEqual(changed.rows[2],original.rows[2])
        self.assertEqual(original.rows[2],[.5]*1024)
        allchanged=patch_fixed(original,1,delta*2,torch)
        self.assertEqual(allchanged.rows[0],original.rows[0])
        self.assertNotEqual(allchanged.rows[1],original.rows[1])
        for cell in self.plan["cells"][4:]:
            item=self.plan["candidates"][cell["cell_id"]]
            h0=load_matrix(HERE,self.plan["archived_baselines"][cell["baseline_cell_id"]])
            offset=load_matrix(HERE,item)
            metrics=verify_fixed(h0,h0,apply_values(h0,offset,item["window_offset"]),offset,item["window_offset"])
            verify_pair(item["archived_last_actual_norm"],metrics["actual_total_norm"])
        self.assertEqual(verify_pair(1.,1.+.5e-6),abs(1.-(1.+.5e-6)))
        with self.assertRaisesRegex(ValueError,"paired"): verify_pair(1.,1.+2e-6)
        with self.assertRaises(ValueError): verify_pair(float("nan"),1.)
        base=[[.5]*1024]
        zero=[[0.]*1024]
        self.assertEqual(verify_fixed(base,base,base,zero,0)["actual_total_norm"],0)
        with self.assertRaisesRegex(ValueError,"baseline"): baseline_match(base,[[.6]*1024])
        with self.assertRaisesRegex(ValueError,"nonfinite"): baseline_match(base,[[float("nan")]*1024])
        with self.assertRaisesRegex(ValueError,"arithmetic"): verify_fixed(base,base,base,delta,0)
        large=[[f32(.2)]*1024]
        with self.assertRaisesRegex(ValueError,"cap"): verify_fixed(base,base,apply_values(base,large,0),large,0)
        with self.assertRaisesRegex(ValueError,"zero"): verify_fixed([[0.]*1024],[[0.]*1024],[[0.]*1024],zero,0)

    def test_strict_flip_retention_other_tie_and_margin_only_rule(self):
        def score(label,target=32,margin=5.):
            values=[-100.]*40
            values[32]=0.
            values[33]=0.
            values[{"A":32,"B":33,"OTHER":3}[label]]=margin
            return scorer.score(values,target)
        base=score("B")
        rawonly=score("A",margin=.04)
        self.assertTrue(rawonly["raw_requested_choice"])
        self.assertFalse(rawonly["strict_pass"])
        self.assertTrue(scorer.classify(base,score("A"),"A")["strict_flip"])
        self.assertTrue(scorer.classify(score("A"),score("A"),"A")["strict_retention"])
        self.assertFalse(scorer.classify(score("OTHER"),score("A"),"A")["eligible_flip"])
        self.assertEqual(scorer.score([0.]*40,32)["choice"],"TIE")
        self.assertEqual(scorer.score([float("nan")]*40,32)["choice"],"NONFINITE")
        self.assertEqual(scorer.opportunity_summary([],"flip")["status"],"UNTESTED")
        self.assertFalse(scorer.gates(True,.05-1e-6-1e-12,.8)["strict_pass"])
        self.assertTrue(scorer.gates(True,.05-1e-6,.8)["strict_pass"])
        self.assertFalse(scorer.gates(True,1.,.8-1e-12)["strict_pass"])
        c={"pair_id":"x","requested_label":"A","policy":"P","choice":"B","strict_pass":False,
           "requested_margin":-.2,"strict_flip":False,"strict_retention":False,"actual_total_norm":.1}
        s={**c,"choice":"A","requested_margin":.04}
        self.assertFalse(scorer.compare_pairs([c],[s])["suffix_advantage"])
        s.update(strict_pass=True,strict_flip=True,requested_margin=.2)
        self.assertTrue(scorer.compare_pairs([c],[s])["suffix_advantage"])
        lost={**c,"pair_id":"y","strict_retention":True,"strict_pass":True}
        failed={**lost,"strict_retention":False,"strict_pass":False}
        self.assertFalse(scorer.compare_pairs([c,lost],[s,failed])["suffix_advantage"])

    def test_failed_call_and_timeout_saved_no_retry(self):
        with tempfile.TemporaryDirectory() as d:
            counter=Counter(Budget(d),self.plan["cells"],time.monotonic()+5)
            def fail(): raise RuntimeError("deliberate")
            with self.assertRaises(RuntimeError): counter.call(self.plan["cells"][0],fail)
            with self.assertRaisesRegex(ValueError,"retry"): counter.call(self.plan["cells"][0],lambda:None)
            self.assertEqual(counter.attempted,1)
            self.assertIn('"event": "failed"',(Path(d)/"forward_events.jsonl").read_text())
        with tempfile.TemporaryDirectory() as d:
            capture=runner.supervise([sys.executable,"-c","import time; time.sleep(10)"],Budget(d),timeout=.1)
            self.assertNotEqual(capture["status"],"complete_valid")
            self.assertTrue((Path(d)/"supervisor_final.json").is_file())

    def test_fake_success_lifecycle_receipt_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            budget=Budget(root)
            (root/"candidates").mkdir()
            for item in list(self.plan["candidates"].values())+list(self.plan["archived_baselines"].values()):
                budget.write_bytes(item["file"],(HERE/item["file"]).read_bytes())
            budget.write("freeze.json",{"plan":self.plan})
            budget.write("input_usage.json",{"captured_at_unix":time.time(),"source":"get_usage_limits","bucket":"standard_codex","used_percent":[76]})
            replay={}
            def fake_worker(command,active):
                capture={"status":"complete_valid","eof_observed":True}
                active.write("capture.json",capture)
                active.write("worker_final.json",{"status":"complete","model_load_attempts":1,"model_load_completed":1,"forward_attempts":20,"forward_completed":20,"derivatives":0})
                active.write("runtime.json",{"boundaries":[{"prompt_id":p["prompt_id"],"prompt_length":137,"input_token_index":136,
                       "content_token_ids":{"A":32,"B":33},"suffix_alignment":self.plan["alignment"][p["prompt_id"]]} for p in self.plan["prompts"]]})
                (root/"logits").mkdir()
                (root/"states").mkdir()
                states={}
                for i,cell in enumerate(self.plan["cells"],1):
                    active.event("forward_events.jsonl",{"event":"started","attempt":i,"cell_id":cell["cell_id"],"monotonic":i*2})
                    active.event("forward_events.jsonl",{"event":"completed","attempt":i,"monotonic":i*2+1})
                    h0=load_matrix(root,self.plan["archived_baselines"][cell["prompt_id"]])
                    alignment=self.plan["alignment"][cell["prompt_id"]]
                    state={"pre":h0,"post":h0,"hook_calls":1,"integrity_passed":True,"outside_sha_before":"same","outside_sha_after":"same",
                           "outside_max_abs_difference":0,"sequence_length":137,"input_token_index":136,"suffix_length":65,
                           "selected_positions":alignment["selected_positions"],"edited_positions":[]}
                    if cell["kind"]!="neutral":
                        item=self.plan["candidates"][cell["cell_id"]]
                        delta=load_matrix(root,item)
                        post=apply_values(h0,delta,item["window_offset"])
                        state.update(post=post,edited_positions=item["mask"])
                        state.update(verify_fixed(h0,h0,post,delta,item["window_offset"]))
                        expected=item["archived_last_actual_norm"] if cell["kind"]=="control" else states[cell["paired_control_cell_id"]]["actual_total_norm"]
                        state["paired_or_archived_norm_error"]=verify_pair(expected,state["actual_total_norm"])
                    states[cell["cell_id"]]=state
                    name=write_capture(active,i,state,cell["kind"]!="neutral")
                    values=array.array("f",[-100.])*248320
                    values[33 if cell["kind"] in ("neutral","control") else cell["requested_token_id"]]=5.
                    replay[cell["cell_id"]]=values
                    packed=array.array("f",values)
                    if sys.byteorder!="little": packed.byteswap()
                    raw=packed.tobytes()
                    compressed=zlib.compress(raw)
                    file=f"logits/{i:02d}.f32.zlib"
                    active.write_bytes(file,compressed)
                    active.event("raw_rows.jsonl",{**cell,"logits_file":file,"raw_sha256":sha(raw),"compressed_sha256":sha(compressed),"vocabulary":248320,
                                 "capture_file":name,"capture_sha256":sha((root/name).read_bytes()),"prompt_length":137,"input_token_index":136,"input_token_ids_sha256":cell["prompt_id"]})
                return capture
            def fake_finalize(*args,**kwargs):
                scorer.finalize()
                self.assertFalse((root/"FINAL_INVENTORY.json").exists())
                return SimpleNamespace(returncode=0,stdout=b"",stderr=b"")
            with patch.object(runner,"HERE",root),patch.object(scorer,"HERE",root), \
                 patch.object(runner,"check_freeze",return_value={"plan":self.plan}),patch.object(scorer,"check_freeze",return_value={"plan":self.plan}), \
                 patch.object(runner,"git",side_effect=[b"frozen",b""]),patch.object(runner,"supervise",side_effect=fake_worker), \
                 patch.object(runner.subprocess,"run",side_effect=fake_finalize), \
                 patch.object(scorer,"archived_logits",side_effect=lambda item,plan:replay[next(k for k,v in self.plan["candidates"].items() if v is item)]):
                runner.run(root/"input_usage.json","frozen")
            result=read(root/"results.json")
            self.assertEqual((result["control"]["strict_passes"],result["suffix"]["strict_passes"]),(4,8))
            self.assertTrue(result["comparison"]["suffix_advantage"])
            names={f["path"] for f in read(root/"FINAL_INVENTORY.json")["files"]}
            self.assertTrue({"finalize_receipt.json","finalize_process.json","results.json","REPORT.md"}<=names)


if __name__=="__main__":
    start=time.monotonic()
    result=unittest.main(exit=False)
    print(f"Focused tests elapsed seconds: {time.monotonic()-start:.6f}")
    sys.exit(not result.result.wasSuccessful())
