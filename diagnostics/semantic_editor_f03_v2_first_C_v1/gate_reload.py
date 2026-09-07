"""Authenticated parent reload functions; no fit path and no model/tokenizer imports."""
import importlib.util
import math
import sys
from pathlib import Path
def require(ok,message):
    if not ok:raise ValueError(message)
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("ordinary_screen_centroid",ROOT/"src/sp_lense/conditional_gate_models.py")
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
CenteredCosineCentroidModel=module.CenteredCosineCentroidModel
PARAMETERS=("grand_mean","positive_centroid","negative_centroid","direction")
def parameter_record(model):
    result={name:list(getattr(model,name)) for name in PARAMETERS}
    require(all(len(v)==1024 and all(math.isfinite(x) for x in v) for v in result.values()),"finite1024parameter dimensions")
    return result
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
