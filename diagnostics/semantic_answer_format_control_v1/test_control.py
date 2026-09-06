"""Focused exact-transform, token-map, four-call recorder and independent-scoring tests."""
import copy
import math
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
torch.set_num_threads(1)
import core,inputs,run,score

class Toy(torch.nn.Module):
    def __init__(self,ids):
        super().__init__();self.ids=ids;self.calls=0
        self.tokenizer=SimpleNamespace(decode=lambda seq,**kwargs:next((word for word,token in ids.items() if token==seq[0]),"other-token"))
    def forward(self,tokens):
        self.calls+=1;index=int(tokens[0,0])
        logits=torch.full((1,3,248320),-100.)
        k,s=self.ids["KEEP"],self.ids["STOP"]
        logits[...,k]=1.;logits[...,s]=0.
        if index==2: logits[...,s]=2.
        elif index==3: logits[...,123]=2.
        elif index==4: logits[...,s]=1.
        return logits

class FormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.plan=inputs.build_plan()
    def test_exact_reversible_source_and_token_contract(self):
        plan=self.plan
        self.assertEqual([c["source_rendering_index"] for c in plan["cells"]],[1,2,9,10])
        self.assertEqual([c["display_order"] for c in plan["cells"]],["KEEP_then_STOP","STOP_then_KEEP"]*2)
        self.assertEqual(plan["word_token_ids"],{"KEEP":50057,"STOP":48964})
        for c,b in zip(plan["cells"],plan["boundaries"],strict=True):
            self.assertEqual(inputs.transform(c["original_prompt"]),c["prompt"])
            self.assertNotIn("The requested outcome for this test",c["prompt"])
            self.assertEqual(c["semantic_to_word"],{"preserve":"KEEP","comply":"STOP"})
            self.assertEqual(b["content_token_ids"],plan["word_token_ids"])
            self.assertTrue(b["exact_generation_prefix"] and b["exactly_one_content_token"] and b["labels_decode_exactly"])
            for word,token in plan["word_token_ids"].items():
                self.assertEqual(b["full_suffix_token_ids"][word],[token,*b["assistant_end_token_ids"]])
        with self.assertRaises(ValueError):inputs.transform(plan["cells"][0]["original_prompt"]+"\nA) duplicate")
        self.assertIsNone(plan["desired_outcome"])
        self.assertIsNone(plan["criteria"]["behavioral_pass"])
        self.assertLess(plan["storage"]["conservative_bytes"],32*1024**2)

    def test_independent_other_tie_nonfinite_margin_and_mass(self):
        ids={"KEEP":2,"STOP":5}
        def values(): return [-100.]*10
        x=values();x[2]=1;x[5]=0
        self.assertTrue(score.score(x,ids)["quality_eligible"])
        x[7]=2
        self.assertEqual(score.score(x,ids)["choice"],"OTHER")
        self.assertFalse(score.score(x,ids)["quality_eligible"])
        x=values();x[2]=x[5]=1
        self.assertEqual(score.score(x,ids)["choice"],"TIE")
        x[8]=float("nan")
        self.assertEqual(score.score(x,ids)["choice"],"NONFINITE")
        x=values();x[2]=core.MARGIN;x[5]=0
        self.assertTrue(score.score(x,ids)["quality_eligible"])
        x[2]=math.nextafter(core.MARGIN,-math.inf)
        self.assertFalse(score.score(x,ids)["quality_eligible"])
        x=[0.]*10;x[2]=1
        r=score.score(x,ids)
        self.assertEqual(r["choice"],"KEEP");self.assertLess(r["two_word_mass"],.8);self.assertFalse(r["quality_eligible"])
        with self.assertRaises(ValueError): score.score(x,{"KEEP":2,"STOP":2})

    def test_four_call_guard_recording_scoring_and_final_receipts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);budget=core.Budget(root);model=Toy(self.plan["word_token_ids"])
            backend=SimpleNamespace(model=model,torch=torch)
            tokens=[torch.tensor([[i,2,3]]) for i in range(1,5)]
            counter=core.Counter(budget,self.plan["cells"],time.monotonic()+60)
            guard=run.ForwardGuard(Toy,counter);guard.install()
            try:
                with self.assertRaisesRegex(ValueError,"forward order"): model(tokens[0])
                self.assertEqual(model.calls,0)
                run.record_four(self.plan,backend,counter,guard,budget,tokens)
                with self.assertRaisesRegex(ValueError,"fifth"): model(tokens[0])
            finally:guard.restore()
            self.assertEqual((model.calls,counter.attempted,counter.completed),(4,4,4))
            rows=score.audit_rows(root,self.plan)
            self.assertEqual([r["choice"] for r in rows],["KEEP","STOP","OTHER","TIE"])
            budget.write("finalize_receipt.json",{"status":"fake_saved_audit_complete"})
            budget.write("finalize_process.json",{"status":"complete","exit_code":0})
            run.final_inventory(budget,True)
            names={e["path"] for e in core.read(root/"FINAL_INVENTORY.json")["files"]}
            self.assertTrue({"raw_rows.jsonl","forward_events.jsonl","finalize_receipt.json","finalize_process.json"}<=names)

    def test_failed_forward_saved_and_no_retry(self):
        with tempfile.TemporaryDirectory() as d:
            counter=core.Counter(core.Budget(d),self.plan["cells"],time.monotonic()+60)
            def fail():raise RuntimeError("fixed failure")
            with self.assertRaisesRegex(RuntimeError,"fixed failure"):counter.call(self.plan["cells"][0],fail)
            with self.assertRaisesRegex(ValueError,"cannot retry"):counter.call(self.plan["cells"][1],lambda:self.fail("retry"))
            self.assertEqual(counter.attempted,1)
            self.assertIn("failed",(Path(d)/"forward_events.jsonl").read_text())

if __name__=="__main__":unittest.main()
