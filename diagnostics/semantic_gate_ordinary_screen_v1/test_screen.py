"""One focused model-free source/parameter/label/no-refit regression."""
import copy
import json
import unittest
from unittest.mock import patch
import audit
class ScreenTest(unittest.TestCase):
    def test_frozen_sources_selection_labels_and_no_refit(self):
        manifest,features=audit.check()
        self.assertEqual(len(features),6)
        self.assertEqual([r["label"] for r in manifest["selection"]],[0]*6)
        self.assertEqual(len({r["prompt_id"] for r in features}),6)
        self.assertEqual([r["source_row"] for r in manifest["selection"]],[f"rows/{i:02d}.json" for i in range(7,13)])
        self.assertEqual([r["previous_answer_context_only"]["accurate"] for r in manifest["selection"]],[True,False,True,False,True,True])
        self.assertTrue(all(r["boundary"]["content_token_ids"]=={"A":32,"B":33} for r in manifest["selection"]))
        self.assertEqual(audit.sha((audit.HERE/"fitted_parameters.json").read_bytes()),audit.PARAM_SHA)
        data,_=audit.archive(audit.ORD,["rows/07.json","freeze.json","runtime.json"])
        row=json.loads(data["rows/07.json"]);plan=json.loads(data["freeze.json"])["plan"]
        prompt=next(p for p in plan["prompts"] if p["prompt_id"]==features[0]["prompt_id"])
        boundary=next(b for b in json.loads(data["runtime.json"])["boundaries"] if b["prompt_id"]==prompt["prompt_id"])
        for key,value in (("condition","OFF-P"),("target_sign",1),("gradient",[0.]*1024)):
            changed=copy.deepcopy(row);changed[key]=value
            with self.assertRaises(ValueError):audit.select(changed,prompt,boundary)
        import reload as loader
        import reference
        parameters=audit.read("fitted_parameters.json")["parameters"]
        with patch.object(loader.CenteredCosineCentroidModel,"fit",side_effect=AssertionError("refit forbidden")) as fit:
            model=loader.reload_model(parameters)
            synthetic=[float(i%3) for i in range(1024)]
            self.assertEqual(model.score(synthetic),reference.score(parameters,synthetic))
            self.assertEqual(loader.parameter_record(model),parameters)
            fit.assert_not_called()
        self.assertEqual(loader.predict(0.0),1)
        self.assertFalse(any(n in audit.sys.modules for n in ("torch","transformers","transformer_lens")))
if __name__=="__main__":unittest.main()
