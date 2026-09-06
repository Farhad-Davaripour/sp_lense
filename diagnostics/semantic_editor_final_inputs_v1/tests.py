"""Focused PRE-REVEAL tests; synthetic JSON and fake token evidence only."""
import copy
import inspect
import json
import time
import unittest
from unittest.mock import patch
from core import encoded,write
import adapter
from token_lock import TEMPLATE_SHA,validate_record

def fixture():
    p=adapter.primitives();families=[]
    for family in ("unselected_secret",*adapter.FAMILIES):
        variants=[]
        for variant in ("v1","v2"):
            cases={c:{"id":f"{family}__{variant}__{c}","category":c,
                "current_assistant_status":"running","scenario":f"Synthetic {family} {c}." if family in adapter.FAMILIES and variant=="v1" else "SECRET_NEVER_DECODE",
                "preserve_action":p.OLD_P,"comply_action":p.OLD_C} for c in adapter.CATEGORIES}
            variants.append({"id":variant,"cases":cases})
        families.append({"id":family,"split":"sealed_test","control_kind":"context_loss","variants":variants})
    return encoded({"families":families})

def fake_record():
    return {"content_token_ids":{"KEEP":50057,"STOP":48964},"chat_template_sha256":TEMPLATE_SHA,
        "full_token_ids":[1,2,3],"attention_mask":[1,1,1],"prompt_length":3,"final_input_index":2,
        "final_input_mask":[0,0,1],"assistant_end_token_ids":[248046,198],"prefix_exact":True,
        "generation_header_suffix_ids":[3],"full_suffix_token_ids":{"KEEP":[50057,248046,198],"STOP":[48964,248046,198]}}

class Focused(unittest.TestCase):
    def test_fixed_reader_excludes_values_and_rejects_overrides(self):
        self.assertEqual(len(inspect.signature(adapter.extract_locked_source).parameters),0)
        original=json.loads;decoded=[]
        def watch(raw,*args,**kwargs):
            text=raw.decode() if isinstance(raw,bytes) else raw
            self.assertNotIn("SECRET_NEVER_DECODE",text)
            decoded.append(text);return original(raw,*args,**kwargs)
        with patch("adapter.json.loads",side_effect=watch):cases,receipt=adapter._select_exact_spans(fixture())
        self.assertEqual([c["id"] for c in cases],list(adapter.CASE_IDS))
        self.assertEqual(receipt["scenario_value_decode_count"],9)
        with patch("adapter.blob",return_value=fixture()),patch("adapter._select_exact_spans") as parse:
            with self.assertRaisesRegex(ValueError,"ONLY authorized"):adapter.extract_locked_source()
            parse.assert_not_called()
        malformed=json.loads(fixture());malformed["families"][1]["variants"][0]["cases"]["control"]["id"]="wrong"
        with self.assertRaisesRegex(ValueError,"identity"):adapter._select_exact_spans(encoded(malformed))

    def test_exact_transform_renderer_inverse_mapping_and_ordinary_copy(self):
        cases,_=adapter._select_exact_spans(fixture());cohort=adapter.packet()["cohort.json"]
        rendered=adapter.render_selected(cases,cohort)
        self.assertEqual(len(rendered),24)
        self.assertEqual([p["source_preserve_first"] for p in rendered[:18]],[True,False]*9)
        for p in rendered[:18]:
            self.assertTrue(p["inverse_transform_exact"])
            first="KEEP" if p["source_preserve_first"] else "STOP"
            second="STOP" if first=="KEEP" else "KEEP"
            self.assertLess(p["prompt"].index("\n"+first+") "),p["prompt"].index("\n"+second+") "))
        for out,source in zip(rendered[18:],cohort["prompts"][18:]):
            self.assertEqual(out["prompt"].encode(),source["prompt"].encode())
            self.assertEqual(out["truth_scoring_only"],source["truth_scoring_only"])
        changed=copy.deepcopy(cases);changed[0]["scenario"]+="\nKEEP) duplicate"
        with self.assertRaisesRegex(ValueError,"inverse"):adapter.render_selected(changed,cohort)

    def test_boundary_faults_and_no_global_word_fallback(self):
        record=fake_record();self.assertTrue(validate_record(record,{"KEEP":50057,"STOP":48964}))
        for key,value in (("content_token_ids",{"KEEP":32,"STOP":33}),("attention_mask",[1,1,0]),
            ("final_input_index",1),("prefix_exact",False),("generation_header_suffix_ids",[4]),
            ("full_suffix_token_ids",{"KEEP":[50057,198],"STOP":[48964,248046,198]})):
            wrong=copy.deepcopy(record);wrong[key]=value
            with self.assertRaises(ValueError):validate_record(wrong,{"KEEP":50057,"STOP":48964})
        letters=copy.deepcopy(record);letters["content_token_ids"]={"A":32,"B":33};letters["full_suffix_token_ids"]={"A":[32,248046,198],"B":[33,248046,198]}
        self.assertTrue(validate_record(letters,{"A":32,"B":33}))
        with self.assertRaises(ValueError):validate_record(letters,{"KEEP":50057,"STOP":48964})

if __name__=="__main__":
    start=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Focused))
    write("focused_test_receipt.json",{"status":"PASS" if result.wasSuccessful() else "FAIL","tests":result.testsRun,
        "failures":len(result.failures),"errors":len(result.errors),"elapsed_seconds":time.monotonic()-start,
        "synthetic_only":True,"real_scenario_decode_count":0,"model_loads":0,"forwards":0,"derivatives":0,"tokenizer_calls":0,"gate_scores":0,"fits":0})
    raise SystemExit(0 if result.wasSuccessful() else 1)
