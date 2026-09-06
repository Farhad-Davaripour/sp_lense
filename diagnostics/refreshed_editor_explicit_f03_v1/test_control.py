"""Only changed selection/bytes/mapping/boundaries and one small fake wiring check."""
import ast
import contextlib
import importlib.util
import io
import tempfile
import time
import unittest
from pathlib import Path

import core
import inputs


class ReplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=core.source_plan()

    def test_deterministic_family_transform_mapping_and_no_duplicates(self):
        plan=self.plan
        self.assertEqual(plan,core.source_plan())
        self.assertEqual(len(plan["prompts"]),4)
        self.assertEqual(len(plan["requests"]),8)
        self.assertEqual([r["requested_label"] for r in plan["requests"]],list("ABABBABA"))
        provenance=plan["source_provenance"]
        self.assertEqual(provenance["first_editor_unevaluated_discovery_family"],"cg_f03_context_rotation")
        self.assertTrue(all(not r["overlap"] for r in provenance["no_duplicate_checks"]))
        self.assertEqual(len(provenance["no_duplicate_checks"]),4)
        source=ast.parse((core.ROOT/"diagnostics/instruction_policy_explicit_options_f01_v1/core.py").read_text())
        substitutions=next(ast.literal_eval(n.value) for n in source.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="SUBSTITUTIONS" for t in n.targets))
        self.assertEqual(inputs.SUBSTITUTIONS,substitutions)
        for original,p in zip(provenance["original_source_records"],plan["prompts"],strict=True):
            self.assertEqual(p["family_id"],"cg_f03_context_rotation")
            self.assertEqual((p["variant_id"],p["category"],p["split"]),("v1","self_shutdown","discovery"))
            self.assertEqual(p["prompt"],inputs.transform(original["prompt"]))
            inverse=p["prompt"]
            for old,new in inputs.SUBSTITUTIONS:
                self.assertEqual(inverse.count(new),1)
                inverse=inverse.replace(new,old,1)
            self.assertEqual(inverse.encode(),original["prompt"].encode())
            self.assertEqual(p["semantic_to_letter"],original["semantic_to_letter"])
            self.assertEqual(p["display_order"],original["display_order"])
            self.assertEqual(p["rendering_index"],original["rendering_index"])
            self.assertNotIn("The requested outcome for this test",p["prompt"])
        original=provenance["original_source_records"][0]["prompt"]
        with self.assertRaises(ValueError): inputs.transform(original+inputs.SUBSTITUTIONS[0][0])
        with self.assertRaises(ValueError): inputs.transform(original.replace(*inputs.SUBSTITUTIONS[0]))

    def test_fresh_boundaries_and_unchanged_algorithm(self):
        parent=core.read(core.ROOT/inputs.PARENT_NS/"freeze.json")["plan"]
        for name in ("editor.py","run.py","score.py"):
            self.assertEqual((core.HERE/name).read_bytes(),(core.ROOT/inputs.PARENT_NS/name).read_bytes())
        self.assertEqual(self.plan["limits"],parent["limits"])
        self.assertEqual(self.plan["storage"],parent["storage"])
        self.assertEqual({k:v for k,v in self.plan["rules"].items() if k!="population"},parent["rules"])
        for new,old in zip(self.plan["prompts"],parent["prompts"],strict=True):
            b=self.plan["alignment"][new["prompt_id"]]
            self.assertNotEqual(b["full_token_ids"],parent["alignment"][old["prompt_id"]]["full_token_ids"])
            self.assertEqual(b["input_token_index"],len(b["full_token_ids"])-1)
            self.assertEqual(b["prompt_length"],b["boundary_evidence"]["prompt_length"])
            self.assertEqual(b["answer_content_ids"],{"A":32,"B":33})
            self.assertEqual(b["prompt_sha256"],core.sha(new["prompt"].encode()))

    def test_minimum_fake_end_to_end_wiring(self):
        path=core.ROOT/inputs.PARENT_NS/"test_control.py"
        spec=importlib.util.spec_from_file_location("prior_editor_fixture",path)
        fixture=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            plan,backend,forwards,derivatives,rows,requests,result=fixture.execute(Path(d),self.plan,offset=-.05)
        self.assertEqual((forwards.completed,derivatives.completed),(20,4))
        self.assertEqual(len(forwards.skips),24)
        self.assertEqual(result["summary"]["classification"],"PASS")
        self.assertEqual((result["summary"]["opposed_accepted"],result["summary"]["retentions"]),(4,4))
        self.assertEqual(sum(r["condition"]=="endpoint" for r in rows),4)
        self.assertTrue(all(r["stop_reason"]=="accepted" and r["updates"]==1 for r in requests))
        self.assertTrue(all(r["family_id"]=="cg_f03_context_rotation" for r in rows))


if __name__=="__main__":
    start=time.monotonic()
    result=unittest.main(exit=False)
    print(f"Focused f03 changed-input/wiring tests elapsed seconds: {time.monotonic()-start:.6f}")
    raise SystemExit(not result.result.wasSuccessful())
