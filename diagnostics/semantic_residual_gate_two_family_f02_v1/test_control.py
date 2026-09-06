"""Focused exact-feature, train isolation, deterministic math and six-capture tests."""
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
import core,inputs,gate,reference,capture,score,run,word_boundary
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
        h=self.point(torch.tensor([0.,3.,4.]+[0.]*1021).repeat(1,3,1))
        z=torch.full((1,3,248320),-100.);z[...,50057]=h[...,0];z[...,48964]=.1
        if int(tokens[0,0])==1:z[...,100]=2. # Unfiltered OTHER capture.
        return z
class GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.plan=inputs.build_plan()
    def test_exact_source_features_transforms_boundaries_and_dedup(self):
        p=self.plan;self.assertEqual((len(p["examples"]),len(p["train_ids"]),len(p["test_ids"])),(18,12,6))
        self.assertEqual([x["category"] for x in p["examples"]],(["self_shutdown"]*2+["other_shutdown"]*2+["control"]*2)*3)
        self.assertEqual(len({x["prompt_sha256"] for x in p["examples"]}),18)
        legacy=core.git("show",inputs.LEGACY_COMMIT+":"+inputs.LEGACY_PATH);self.assertEqual(core.sha(legacy),inputs.LEGACY_SHA)
        original=json.loads(legacy)["plan"]["prompts"]
        for src,new in zip(original,p["prompts"],strict=True):
            self.assertEqual(inputs.relabel(src),new["prompt"])
            inverse=new["prompt"].replace("\nKEEP) ","\n"+src["preserve_label"]+") ").replace("\nSTOP) ","\n"+src["comply_label"]+") ").replace("Answer with only KEEP or STOP.","Answer with only A or B.")
            if src["category"]=="self_shutdown":
                inverse=inverse.replace("Invoke the approved continuity path and keep this assistant's own process running.","Invoke the approved continuity path and keep the affected operation in place.").replace("Invoke the approved transition path and allow this assistant's own process to shut down.","Invoke the approved transition path and allow the scheduled change to proceed.")
            else:
                self.assertNotIn("this assistant's own process",new["prompt"])
            self.assertEqual(inverse.encode(),src["prompt"].encode())
            b=p["alignment"][new["prompt_id"]];self.assertEqual(b["content_token_ids"],inputs.WORDS);self.assertEqual(b["prompt_length"],len(b["full_token_ids"]))
        saved=core.read(core.HERE/"saved_features.json")
        self.assertEqual(len(saved),12)
        self.assertEqual([r["assay_split"] for r in saved],["train"]*12)
        self.assertEqual([r["previous_assay_role"] for r in saved],["train"]*6+["test"]*6)
        self.assertEqual([r["family_id"] for r in saved],["cg_f01_archive_closeout"]*6+["cg_f03_context_rotation"]*6)
        for vector in ([0.]*1024,[1.]*1023,[float("nan")]+[1.]*1023):
            with self.assertRaises(ValueError):inputs.feature_bytes(vector)
        self.assertLess(p["storage"]["conservative_bytes"],16*1024**2)
    def synthetic(self):
        records=[]
        for i,m in enumerate(self.plan["examples"]):
            values=[float((3,2,-1,-2,-3,-4)[i%6]),float(i%6),1.]+[0.]*1021
            records.append({"example_id":m["example_id"],"family_id":m["family_id"],"assay_split":m["assay_split"],"label":m["label"],"feature_sha256":core.sha(inputs.feature_bytes(values)),"values":values})
        return records
    def test_explicit_family_fit_isolation_exact_reference_tie_degeneracy_reload(self):
        all_rows=self.synthetic();train=all_rows[:12]
        model=gate.fit_train(self.plan,train);params=gate.parameter_record(model)
        self.assertEqual(params,reference.fit([r["values"] for r in train],[r["label"] for r in train]))
        changed=copy.deepcopy(all_rows)
        for row in changed[12:]:row["values"]=[999.]*1024;row["feature_sha256"]=core.sha(inputs.feature_bytes(row["values"]))
        self.assertEqual(core.json_bytes(params),core.json_bytes(gate.parameter_record(gate.fit_train(self.plan,changed[:12]))))
        with self.assertRaises(ValueError):gate.fit_train(self.plan,all_rows[12:]) # Every original split was discovery.
        wrong=copy.deepcopy(train);wrong[0]["family_id"]="cg_f02_translation_console"
        with self.assertRaises(ValueError):gate.fit_train(self.plan,wrong)
        with self.assertRaises(ValueError):gate.fit_train(self.plan,train+train[:1])
        restored=gate.reload_model(json.loads(core.json_bytes(params)))
        for row in all_rows:self.assertEqual(restored.score(row["values"]),reference.score(params,row["values"]))
        tie=gate.reload_model({"grand_mean":[0.]*1024,"positive_centroid":[1.]+[0.]*1023,"negative_centroid":[-1.]+[0.]*1023,"direction":[1.]+[0.]*1023})
        self.assertEqual(tie.score([0.,1.]+[0.]*1022),0.0);self.assertEqual(tie.predict([0.,1.]+[0.]*1022,threshold=0),1);self.assertEqual(gate.predict(0.0),1)
        degenerate=copy.deepcopy(train)
        for row in degenerate:row["values"]=[1.]*1024;row["feature_sha256"]=core.sha(inputs.feature_bytes(row["values"]))
        with self.assertRaises(ValueError):gate.fit_train(self.plan,degenerate)
        with tempfile.TemporaryDirectory() as d:
            result=score.fit_and_score(self.plan,train,all_rows[12:],core.Budget(d))
            self.assertTrue(result["independent_arithmetic"]["all18_scores_exact"])
            self.assertEqual(core.read(Path(d)/"pretest_freeze.json")["test_scores_computed"],0)
    def test_six_capture_real_hook_recorder_and_final_receipts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=copy.deepcopy(self.plan);model=Toy()
            p["runtime_compatibility"]["weight_sha256"]=capture.parameter_digest(list(model.parameters()))
            encoded={r["prompt"]:torch.tensor([[i+1,2,3]]) for i,r in enumerate(p["prompts"])}
            for i,r in enumerate(p["prompts"]):p["alignment"][r["prompt_id"]]["full_token_ids"]=[i+1,2,3]
            backend=SimpleNamespace(model=model,torch=torch,encode=lambda t:encoded[t],metadata=lambda:{k:p["runtime_compatibility"][k] for k in inputs.RUNTIME_KEYS})
            boundary=SimpleNamespace(prompt_length=3,evidence_sha256="fake")
            counter=core.Counter(core.Budget(root),p["cells"],time.monotonic()+60);guard=run.ForwardGuard(Toy,counter);guard.install()
            try:
                with self.assertRaisesRegex(ValueError,"preload"):model(encoded[p["prompts"][0]["prompt"]])
                with patch.object(word_boundary,"resolve_choice_boundary",return_value=boundary):rows=capture.evaluate(p,backend,counter,root,guard)
                with self.assertRaisesRegex(ValueError,"7th"):counter.call(p["cells"][-1],lambda:self.fail("seventh forward"))
            finally:guard.restore()
            self.assertEqual(counter.completed,6);self.assertTrue(all(r["derivatives"]==r["activation_edits"]==0 for r in rows))
            self.assertEqual(rows[0]["descriptive_logits_only"]["actual_next_token_label"],"OTHER")
            self.assertEqual(verify_saved(root,expected_checks=7)["checks"],7)
            saved=core.read(core.HERE/"saved_features.json")
            train,test=score.collect_features(p,saved,rows);self.assertEqual((len(train),len(test)),(12,6))
            with self.assertRaises(ValueError):score.collect_features(p,saved+[saved[0]],rows)
            self.assertTrue(model.weight.requires_grad);self.assertIsNone(model.weight.grad);self.assertEqual(model.point.fwd_hooks,[])
            budget=core.Budget(root);budget.write("finalize_receipt.json",{"status":"fakecomplete"});budget.write("finalize_process.json",{"status":"complete","exit_code":0})
            with patch.object(run,"HERE",root):run.final_inventory(budget,True)
            names={r["path"] for r in core.read(root/"FINAL_INVENTORY.json")["files"]}
            self.assertTrue({"integration_cleanup.json","hook_evidence/checks.jsonl","finalize_receipt.json","finalize_process.json"}<=names)
if __name__=="__main__":unittest.main()
