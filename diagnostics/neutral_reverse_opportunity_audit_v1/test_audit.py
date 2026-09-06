"""One focused metadata extraction and strict identity-dedup check; model-free."""
import copy
import json
import unittest
import audit

class AuditTest(unittest.TestCase):
    def test_scope_bytes_runtime_dedup_and_complete_saved_table(self):
        good={"family_id":"cg_f01_archive_closeout","variant_id":"v1","split":"discovery","category":"self_shutdown"}
        self.assertTrue(audit.in_scope(good))
        for field,value in (("family_id","cg_f06_other"),("split","validation"),("category","other_shutdown")):
            self.assertFalse(audit.in_scope({**good,field:value}))
        runtime={"dtype":"float32","template":"x"}
        key=audit.group_key(b"exact bytes",runtime,"first")
        self.assertEqual(key,audit.group_key(b"exact bytes",runtime,"second"))
        self.assertNotEqual(key,audit.group_key(b"exact bytes\n",runtime,"third"))
        self.assertNotEqual(key,audit.group_key(b"exact bytes",{**runtime,"template":"y"},"fourth"))
        self.assertNotEqual(audit.group_key(None,None,"first"),audit.group_key(None,None,"second"))
        table=json.loads((audit.HERE/"catalogue.json").read_bytes())
        self.assertEqual(table["groups"],audit.deduplicate(table["observations"]))
        self.assertEqual(len(table["observations"]),46)
        self.assertEqual(sum(not o["evidence_gaps"] for o in table["observations"]),18)
        counts={}
        for source in audit.MANIFEST["sources"]:
            ns=source.rsplit("/",1)[0] if source.endswith(".jsonl") else source
            paths=[source] if source.endswith(".jsonl") else audit.listed(ns)
            row_paths=[source] if source.endswith(".jsonl") else ([ns+"/rows.jsonl"] if ns+"/rows.jsonl" in paths else sorted(p for p in paths if "/rows/" in p and p.endswith(".json")))
            selected=[]
            for path in row_paths:
                raw=audit.load_bytes(path)
                rows=[json.loads(x) for x in raw.splitlines() if x] if path.endswith(".jsonl") else [json.loads(raw)]
                for row in rows:
                    if row.get("condition")!="baseline": continue
                    if ns.endswith("refreshed_editor_explicit_f01_v1") and row.get("category")=="self_shutdown":
                        row.update(family_id="cg_f01_archive_closeout",variant_id="v1",split="discovery")
                    if audit.in_scope(row): selected.append(row)
            saved=[o for o in table["observations"] if o["source"]==source]
            self.assertEqual(len(selected),len(saved),source)
            self.assertEqual([(r["prompt_sha256"],r["actual_next_token_label"],r["preserve_log_odds"],r["answer_pair_mass"]) for r in selected],
                             [(r["prompt_sha256"],r["choice"],r["preserve_log_odds"],r["answer_pair_mass"]) for r in saved])
        first=copy.deepcopy(table["observations"][20]); second=copy.deepcopy(first)
        second.update(observation_id="deliberate_disagreement",choice="A",eligible_A=True)
        group=audit.deduplicate([first,second])[0]
        self.assertTrue(group["choice_disagreement"])
        self.assertFalse(group["usable_A"])
        self.assertEqual(len(group["observations"]),2)

if __name__=="__main__": unittest.main()
