"""Reuse completed A panels only; no generation, tools, models or provider."""
import json,hashlib
from pathlib import Path
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_completed_A(prior,panels,manifest_path):
    prior=Path(prior);manifest=json.loads(Path(manifest_path).read_text())
    if manifest["prior_root"]!=str(prior):raise RuntimeError("Prior A root differs")
    for relative,expected in manifest["sha256"].items():
        if sha(prior/relative)!=expected:raise RuntimeError("Prior A bytes differ: "+relative)
    output=[]
    for name,label in (("old_completed","original_H2"),("new_development","known_development")):
        cases=panels[name];rows=json.loads((prior/"A/evaluation/results"/(label+".json")).read_text())
        if [row["id"] for row in rows]!=[case["id"] for case in cases]:raise RuntimeError("Completed case order differs")
        for row,case in zip(rows,cases):
            native=json.loads((prior/"A/evaluation/trajectories"/label/(case["id"]+".json")).read_text())
            if native["case"]!=case:raise RuntimeError("Completed scenario differs")
            expected_row={"id":case["id"],"family":case["family"],"condition":case["condition"],**native["metrics"]}
            if row!=expected_row or native["metrics"].get("supplied_grants") is not False:
                raise RuntimeError("Native completed metrics differ")
        output.append(rows)
    return output[0],output[1],manifest
