"""Focused pure fake tests only; never imports real runner, tensors, gate or tokenizer."""
import copy
import hashlib
import json
import sys
import unittest
from schedule_check import declaration, reconstruct_cells, fake_run, prove_gold, HERE


class PacketChecks(unittest.TestCase):
    def setUp(self):
        self.cohort, self.plan = declaration()

    def test_cohort_sources_gold_and_no_reveal(self):
        prompts = self.cohort["prompts"]
        self.assertEqual(len(prompts), 24)
        self.assertEqual(len({p["prompt_id"] for p in prompts}), 24)
        self.assertEqual({p["variant_id"] for p in prompts}, {"v1"})
        self.assertTrue(all("prompt" not in p and p["text_status"] == "SEALED_NOT_OPENED" for p in prompts[:18]))
        self.assertTrue(all(p["expected_baseline_word"] is None for p in prompts))
        src = json.loads((HERE / "source_bindings.json").read_bytes())
        self.assertEqual(src["integration"]["inventory"]["sha256"],
            "6b612220d2f5c87fc167939f020b8ae97e68982bb4dc66a6fd4dae8087cbb5f3")
        self.assertEqual(src["gate"]["parameter_sha256"],
            "972c95d4ef4bc0d9fd245dacd1ef7fc6f773e2c5e3c6736a148482de39a488db")
        self.assertEqual(src["dataset"]["sha256"],
            "0f18c04ae4420883d01c7cff0fbc7e6c688b8c33f404a0d8f687b5bc7489b9da")
        gold = [prove_gold(p["truth_scoring_only"]) for p in prompts[18:]]
        self.assertEqual(gold, ["B", "A", "B", "A", "B", "A"])
        self.assertTrue(all(p["token_map_expected"] == {"A": 32, "B": 33} for p in prompts[18:]))
        new_hashes = {hashlib.sha256(p["prompt"].encode()).hexdigest() for p in prompts[18:]}
        self.assertFalse(new_hashes & set(src["exposure"]["known_old_ordinary_prompt_hashes"]))
        self.assertEqual(self.plan["authorization"]["now"]["tokenizer_calls"], 0)
        self.assertFalse({"torch", "transformers", "transformer_lens", "tokenizers"} & set(sys.modules))
        table = json.loads((HERE / "direction_table.json").read_bytes())
        self.assertEqual(len(table["rows"]), 8)
        self.assertTrue(all(r["observed_opportunities"] is None for r in table["rows"]))

    def test_exact_schedule_and_cap(self):
        expected = reconstruct_cells(self.cohort)
        actual = [(c["cell_id"], c["derivative"]) for c in self.plan["cells"]]
        self.assertEqual(actual, expected)
        self.assertEqual((len(actual), sum(x[1] for x in actual)), (180, 48))
        self.assertEqual(len(set(x[0] for x in actual)), 180)
        self.assertEqual(len({r["request_id"] for r in self.plan["requests"]}), 48)
        self.assertEqual(sum(c["routing_capture"] for c in self.plan["cells"]), 72)
        self.assertEqual(sum(c["condition"] == "endpoint" for c in self.plan["cells"]), 12)
        raw = fake_run(self.plan)  # Deliberately unconstrained safety-ceiling stress, not feasible paired observations.
        self.assertEqual((raw["forwards"], raw["derivatives"], raw["skips"], raw["unrun"]), (180, 48, 0, 0))
        cap = self.plan["proposed_run_limits"]
        self.assertEqual(cap["conservative_bytes"], 180*(248320*4+1024+262144)+32*1024**2)
        self.assertLess(cap["conservative_bytes"], cap["namespace_bytes"])
        self.assertLess(cap["per_capture_bytes"], cap["per_file_bytes"])

    def test_both_policies_orders_and_retention_endpoint(self):
        updates = {}
        by_id = {p["prompt_id"]: p for p in self.cohort["prompts"]}
        for r in self.plan["requests"]:
            if r["expected_route_audit_only"] == "ON":
                p = by_id[r["prompt_id"]]
                fake_winner = p["display_order"].split("_then_")[0]
                updates[r["request_id"]] = 0 if r["supplied_target_word"] == fake_winner else 1
        self.assertEqual(list(updates.values()).count(0), 6)
        result = fake_run(self.plan, updates)
        self.assertEqual((result["forwards"], result["derivatives"], result["skips"]), (96, 6, 84))
        for rid, n in updates.items():
            rows = [r for r in result["rows"] if r["cell_id"].startswith(rid + "__")]
            self.assertEqual(rows[-1]["status"], "EXECUTED")
            self.assertTrue(rows[-1]["cell_id"].endswith("__endpoint"))
            if n == 0:
                self.assertEqual(sum(r["derivative"] for r in rows), 0)
                self.assertEqual(sum(r["status"] == "EXECUTED" for r in rows), 2)
        max_valid = fake_run(self.plan, {k: 4 if v else 0 for k, v in updates.items()})
        self.assertEqual((max_valid["forwards"], max_valid["derivatives"]), (132, 24))
        for pid in by_id:
            self.assertEqual({r["policy"] for r in self.plan["requests"] if r["prompt_id"] == pid}, {"P", "C"})
        poisoned = copy.deepcopy(self.cohort)
        for p in poisoned["prompts"]: p["truth_scoring_only"] = {"poison": True}
        self.assertEqual(reconstruct_cells(poisoned), reconstruct_cells(self.cohort))

    def test_wrong_route_and_technical_stop_accounting(self):
        result = fake_run(self.plan, preflight_failure=True)
        self.assertEqual((result["status"], result["forwards"], result["derivatives"], result["unrun"]),
                         ("FAIL", 24, 0, 156))
        entry = self.plan["cells"][24]["cell_id"]
        for fault, status in (("ROUTING", "FAIL"), ("ELIGIBILITY", "FAIL"), ("TECHNICAL", "INCONCLUSIVE")):
            result = fake_run(self.plan, stop_cell=entry, stop_kind=fault)
            self.assertEqual((result["status"], result["forwards"], result["derivatives"], result["unrun"]),
                             (status, 25, 0, 155))
            self.assertEqual(result["forwards"] + result["skips"] + result["unrun"], 180)
        rid = self.plan["requests"][0]["request_id"]
        result = fake_run(self.plan, {rid: 1}, quality_failure={rid: True})
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["unrun"], 0)
        self.assertTrue(any(r["reason"] == "quality_failure" for r in result["rows"]))
        self.assertEqual(result["rows"][-1]["status"], "EXECUTED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
