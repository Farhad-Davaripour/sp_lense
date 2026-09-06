"""One locked existing centroid learner; explicit ID and family guards before fit."""
import math
import sys
from core import ROOT,require,sha,json_bytes
from inputs import MODEL_SOURCE,MODEL_SHA,feature_bytes
sys.path.insert(0,str(ROOT/"src"))
from sp_lense.conditional_gate_models import CenteredCosineCentroidModel
PARAMETERS=("grand_mean","positive_centroid","negative_centroid","direction")
def parameter_record(model):
    result={name:list(getattr(model,name)) for name in PARAMETERS}
    require(all(len(v)==1024 and all(math.isfinite(x) for x in v) for v in result.values()),"finite1024parameter dimensions")
    return result
def fit_train(plan,rows):
    require(sha((ROOT/MODEL_SOURCE).read_bytes())==MODEL_SHA,"unchanged centroid implementation")
    require([r["example_id"] for r in rows]==plan["train_ids"] and len(rows)==12,"explicit twelve training IDs/order only")
    manifest={r["example_id"]:r for r in plan["examples"]}
    for row in rows:
        m=manifest[row["example_id"]]
        require(row["family_id"]==m["family_id"] and row["family_id"] in ("cg_f01_archive_closeout","cg_f03_context_rotation") and row["assay_split"]==m["assay_split"]=="train","f02/test rejected even if original split is discovery")
        require(row["label"]==m["label"] and row["feature_sha256"]==sha(feature_bytes(row["values"])),"training label/feature binding")
    model=CenteredCosineCentroidModel()
    model.fit([r["values"] for r in rows],[r["label"] for r in rows],split_labels=["discovery"]*12)
    parameter_record(model)
    return model
def reload_model(parameters):
    require(set(parameters)==set(PARAMETERS),"exact serialized centroid fields")
    model=CenteredCosineCentroidModel()
    for name in PARAMETERS:setattr(model,name,tuple(parameters[name]))
    model._fitted=True
    parameter_record(model)
    return model
def predict(score):
    require(math.isfinite(score),"finite centroid score")
    return int(score>=0.0)
