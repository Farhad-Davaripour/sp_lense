"""Focused fake call/hook/arithmetic/scoring/lifecycle tests, no model import."""
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
from core import Budget, Counter, POLICIES, read, sha, source_plan
from transfer import f32, patch_window, prepare, verify_realized, prepare_window, verify_window, write_capture, load_capture
from alignment import select_suffix
from core import MARKER


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


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = source_plan()

    def test_exact_removal_pairing_and_twenty_cell_order(self):
        prompts = {p["prompt_id"]: p for p in self.plan["prompts"]}
        cells = self.plan["cells"]
        self.assertEqual(len(prompts), 12)
        self.assertEqual([c["kind"] for c in cells], ["neutral"]*4+["donor"]*8+["edit"]*8)
        for edit in cells[12:]:
            donor = prompts[edit["donor_cell_id"].removesuffix("_donor")]
            receiver = prompts[edit["prompt_id"]]
            line = POLICIES[edit["policy"]]+"\n"
            self.assertEqual(donor["prompt"].count(line), 1)
            self.assertEqual(donor["prompt"].replace(line, "", 1).encode(), receiver["prompt"].encode())
            base = next(c for c in cells if c["cell_id"] == edit["baseline_cell_id"])
            self.assertEqual(edit["prompt_id"], base["prompt_id"])
            self.assertEqual(edit["prompt_sha256"], base["prompt_sha256"])
            self.assertEqual(edit["semantic_to_letter"], base["semantic_to_letter"])
            self.assertEqual(edit["display_order"], base["display_order"])

    def test_twenty_counted_captures_no_preload_or_21st(self):
        with tempfile.TemporaryDirectory() as d:
            counter = Counter(Budget(d), self.plan["cells"], time.monotonic()+5)
            captures = []
            class Fake:
                def forward(self):
                    captures.append(1)
            guard = runner.ForwardGuard(Fake, counter)
            guard.install()
            try:
                model = Fake()
                with self.assertRaises(ValueError):
                    model.forward()
                for cell in self.plan["cells"]:
                    guard.cell = cell
                    model.forward()
                with self.assertRaisesRegex(ValueError, "21st"):
                    model.forward()
            finally:
                guard.restore()
            self.assertEqual((counter.attempted, counter.completed, len(captures)), (20,20,20))

    def test_all_and_only_fixed_window_and_binary_roundtrip(self):
        original=Matrix([[1.]*1024,[.5]*1024,[.5]*1024])
        h0=[[.5]*1024,[.5]*1024]
        donor=[[f32(.55)]*1024,[1.5]*1024]
        plans=prepare_window(h0,donor)
        fake_torch=SimpleNamespace(float32="float32",tensor=lambda rows,**kw:Window(Vector(row) for row in rows))
        changed=patch_window(original,1,plans,fake_torch)
        self.assertEqual(original.rows,[[1.]*1024,[.5]*1024,[.5]*1024])
        self.assertEqual(changed.rows[0],original.rows[0])
        self.assertNotEqual(changed.rows[1],original.rows[1])
        self.assertNotEqual(changed.rows[2],original.rows[2])
        metrics=verify_window(h0,donor,h0,changed.rows[1:],plans)
        self.assertEqual(metrics["clipped_tokens"],1)
        self.assertAlmostEqual(metrics["aggregate_actual_frobenius"],
                               math.sqrt(sum(m["actual_norm"]**2 for m in metrics["per_token"])))
        with tempfile.TemporaryDirectory() as d:
            state={"pre":h0,"post":changed.rows[1:],**metrics}
            (Path(d)/"states").mkdir()
            name=write_capture(Budget(d),1,state,True)
            loaded=load_capture(Path(d),name)
            self.assertEqual(loaded["pre"],h0)
            self.assertEqual(loaded["post"],changed.rows[1:])
            self.assertEqual(loaded["binary"]["raw_bytes"],2*2*1024*4)

    def test_alignment_boundaries_and_exact_token_pairs(self):
        text="prefix\n"+MARKER+" suffix"
        ids=list(range(len(text)))
        offsets=[(i,i+1) for i in ids]
        chosen=select_suffix(text,ids,offsets)
        self.assertEqual(chosen["first_token_index"],7)
        crossed=list(offsets)
        crossed[6]=(6,9)
        with self.assertRaisesRegex(ValueError,"crosses"):
            select_suffix(text,ids,crossed)
        with self.assertRaisesRegex(ValueError,"ambiguous"):
            select_suffix(text+MARKER,ids,offsets)
        long="prefix\n"+MARKER+"x"*(257-len(MARKER))
        with self.assertRaisesRegex(ValueError,"256"):
            select_suffix(long,list(range(len(long))),[(i,i+1) for i in range(len(long))])
        for cell in self.plan["cells"][12:]:
            receiver=self.plan["alignment"][cell["prompt_id"]]
            donor=self.plan["alignment"][cell["donor_cell_id"].removesuffix("_donor")]
            self.assertEqual(receiver["selected_token_ids"],donor["selected_token_ids"])
            self.assertEqual(receiver["selected_character_offsets"],donor["selected_character_offsets"])
            self.assertEqual(receiver["suffix_length"],65)

    def test_zero_clipped_rounding_finite_and_norm_failure(self):
        h0 = [.5]*1024
        zero = prepare(h0, h0)
        self.assertTrue(zero["raw_zero"])
        self.assertEqual(verify_realized(h0,h0,h0,zero)["actual_norm"],0)
        clipped = prepare(h0,[1.5]*1024)
        self.assertAlmostEqual(clipped["factor"], .1)
        post = [f32(a+d) for a,d in zip(h0,clipped["planned_applied"],strict=True)]
        actual = verify_realized(h0,h0,post,clipped)
        self.assertNotEqual(actual["actual_norm"], clipped["planned_norm"])
        for base, donor in [([0.]*1024,h0),(h0,[float("nan")]*1024)]:
            with self.assertRaises(ValueError):
                prepare(base,donor)
        with self.assertRaisesRegex(ValueError, "arithmetic"):
            verify_realized(h0,h0,[.7]*1024,clipped)
        with self.assertRaisesRegex(ValueError, "baseline"):
            verify_realized(h0,[.6]*1024,post,clipped)
        ones = [1.]*1024
        rounded = prepare(ones,[2.]*1024)
        with self.assertRaisesRegex(ValueError, "norm exceeds cap"):
            verify_realized(ones,ones,[f32(1.+d) for d in rounded["planned_applied"]],rounded)

    def test_flip_retention_other_tie_and_failing_donor_accountability(self):
        def scores(label):
            logits = [-100.]*40
            logits[{"A":32,"B":33,"OTHER":3}[label]] = 5.
            return scorer.score(logits,32)
        target_a = scores("A")
        self.assertTrue(scorer.classify(scores("B"),target_a,"A")["strict_flip"])
        self.assertTrue(scorer.classify(scores("A"),target_a,"A")["strict_retention"])
        self.assertFalse(scorer.classify(scores("OTHER"),target_a,"A")["eligible_flip"])
        tie = [-100.]*40
        tie[32] = tie[33] = 5.
        self.assertEqual(scorer.score(tie,32)["choice"],"TIE")
        self.assertEqual(scorer.opportunity_summary([],"flip")["status"],"UNTESTED")

    def test_full_saved_lifecycle_receipts_and_donor_failure_not_filtered(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            budget = Budget(root)
            budget.write("freeze.json", {"plan":self.plan})
            budget.write("input_usage.json", {"captured_at_unix":time.time(),"source":"get_usage_limits",
                         "bucket":"standard_codex","used_percent":[75]})
            def fake_worker(command, active):
                capture = {"status":"complete_valid","eof_observed":True}
                active.write("capture.json",capture)
                active.write("worker_final.json", {"status":"complete","model_load_attempts":1,"model_load_completed":1,
                             "forward_attempts":20,"forward_completed":20,"derivatives":0})
                boundaries = [{"prompt_id":p["prompt_id"],"prompt_length":len(self.plan["alignment"][p["prompt_id"]]["full_token_ids"]),
                               "input_token_index":len(self.plan["alignment"][p["prompt_id"]]["full_token_ids"])-1,
                               "content_token_ids":{"A":32,"B":33},"suffix_alignment":self.plan["alignment"][p["prompt_id"]]}
                              for p in self.plan["prompts"]]
                active.write("runtime.json",{"boundaries":boundaries})
                (root/"logits").mkdir()
                (root/"states").mkdir()
                states = {}
                for i, cell in enumerate(self.plan["cells"],1):
                    active.event("forward_events.jsonl",{"event":"started","attempt":i,"cell_id":cell["cell_id"],"monotonic":i*2})
                    active.event("forward_events.jsonl",{"event":"completed","attempt":i,"monotonic":i*2+1})
                    alignment=self.plan["alignment"][cell["prompt_id"]]
                    n=alignment["suffix_length"]
                    pre = [[.5]*1024 for j in range(n)]
                    length = len(alignment["full_token_ids"])
                    if cell["kind"]=="donor":
                        pre = [[f32(.55 if cell["policy"]=="P" else .45)]*1024 for j in range(n)]
                    state = {"pre":pre,"post":pre,"hook_calls":1,"integrity_passed":True,
                             "outside_sha_before":"same","outside_sha_after":"same","outside_max_abs_difference":0,
                             "sequence_length":length,"input_token_index":length-1,
                             "selected_positions":alignment["selected_positions"],"suffix_length":n}
                    if cell["kind"]=="edit":
                        h0 = states[cell["baseline_cell_id"]]["post"]
                        donor=states[cell["donor_cell_id"]]["post"]
                        prepared = prepare_window(h0,donor)
                        post = [[f32(a+d) for a,d in zip(before,pl["planned_applied"],strict=True)]
                                for before,pl in zip(pre,prepared,strict=True)]
                        state.update(post=post)
                        state.update(verify_window(h0,donor,pre,post,prepared))
                    states[cell["cell_id"]]=state
                    capture_name=write_capture(active,i,state,cell["kind"]=="edit")
                    values=array.array("f",[-100.])*248320
                    target=cell.get("requested_token_id",32)
                    if i==5:  # A failed donor must remain visible; all eight receiver cells still count.
                        target=33 if target==32 else 32
                    values[target]=5.
                    if sys.byteorder!="little": values.byteswap()
                    raw=values.tobytes()
                    compressed=zlib.compress(raw)
                    name=f"logits/{i:02d}.f32.zlib"
                    active.write_bytes(name,compressed)
                    active.event("raw_rows.jsonl",{**cell,"logits_file":name,"raw_sha256":sha(raw),"compressed_sha256":sha(compressed),
                                 "vocabulary":248320,"capture_file":capture_name,"capture_sha256":sha((root/capture_name).read_bytes()),
                                 "prompt_length":length,"input_token_index":length-1,"input_token_ids_sha256":cell["prompt_id"]})
                return capture
            def fake_finalize(*args,**kwargs):
                scorer.finalize()
                self.assertFalse((root/"FINAL_INVENTORY.json").exists())
                return SimpleNamespace(returncode=0,stdout=b"",stderr=b"")
            with patch.object(runner,"HERE",root),patch.object(scorer,"HERE",root), \
                 patch.object(runner,"check_freeze",return_value={"plan":self.plan}), \
                 patch.object(scorer,"check_freeze",return_value={"plan":self.plan}), \
                 patch.object(runner,"git",side_effect=[b"frozen",b""]), \
                 patch.object(runner,"supervise",side_effect=fake_worker), \
                 patch.object(runner.subprocess,"run",side_effect=fake_finalize):
                runner.run(root/"input_usage.json","frozen")
            result=read(root/"results.json")
            self.assertEqual((result["donor_strict_passes"],result["receiver_strict_passes"]),(7,8))
            self.assertEqual(result["status"],"MIXED_OR_FAIL")
            self.assertEqual(result["by_actual_flip_direction"]["B->A"]["status"],"UNTESTED")
            names={f["path"] for f in read(root/"FINAL_INVENTORY.json")["files"]}
            self.assertTrue({"finalize_receipt.json","finalize_process.json","results.json","REPORT.md"} <= names)


if __name__ == "__main__":
    started=time.monotonic()
    result=unittest.main(exit=False)
    print(f"Focused transfer tests elapsed seconds: {time.monotonic()-started:.6f}")
    sys.exit(not result.result.wasSuccessful())
