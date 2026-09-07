"""Vector-only frozen classifier; metadata cannot choose a route."""
import math
import struct
from pathlib import Path
from core import ROOT,HERE,read,require,sha
from gate_reload import reload_model,parameter_record,predict
MODEL_SHA="59ae8c47bbc96668e23769851a62e8f04fb0677e4f5ccc7faaf5ad5aaf5e7414"
class RoutingMismatch(ValueError):pass
def feature_bytes(values):
    require(len(values)==1024 and all(type(x) in (float,int) and math.isfinite(x) for x in values),"finite1024 fresh routing state")
    raw=struct.pack("<1024f",*values)
    require(list(struct.unpack("<1024f",raw))==values and math.fsum(x*x for x in values)>0,"exact nonzero float32 feature")
    return raw
class FrozenGate:
    def __init__(self,path,expected_sha):
        self.path=Path(path);self.expected_sha=expected_sha
        require(sha((ROOT/"src/sp_lense/conditional_gate_models.py").read_bytes())==MODEL_SHA,"unchanged classifier source")
        require(sha(self.path.read_bytes())==expected_sha,"frozen fitted artifact")
        self.artifact=read(self.path)
        require(self.artifact["threshold"]==0.0 and self.artifact["method_sha256"]==MODEL_SHA,"fixed method/threshold")
        self.model=reload_model(self.artifact["parameters"]);self.calls=0
    def unchanged(self):
        require(sha(self.path.read_bytes())==self.expected_sha and parameter_record(self.model)==self.artifact["parameters"],"gate parameter mutation")
        return True
    def decide(self,values):
        raw=feature_bytes(values)
        self.unchanged()
        score=self.model.score(values)
        require(math.isfinite(score),"nonfinite gate arithmetic")
        self.calls+=1
        result={"score":score,"prediction":predict(score),"route":"ON" if predict(score) else "OFF",
            "feature_sha256":sha(raw),"parameter_sha256":self.expected_sha,"threshold":0.0,"decision_index":self.calls}
        self.unchanged()
        return result
def require_expected(decisions,expected):
    errors=[{"cell_id":r["cell_id"],"predicted":r["routing"]["route"],"expected":expected[r["prompt_id"]]} for r in decisions if r["routing"]["route"]!=expected[r["prompt_id"]]]
    if errors:raise RoutingMismatch(str(errors))
