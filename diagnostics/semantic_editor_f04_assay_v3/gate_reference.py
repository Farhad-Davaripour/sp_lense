"""Independent saved-data binary64 centroid equations; no model or production learner."""
import math
def normalize(values):
    magnitude=math.sqrt(math.fsum(x*x for x in values))
    if not math.isfinite(magnitude) or magnitude==0:raise ValueError("degenerate norm")
    return [x/magnitude for x in values]
def score(parameters,vector):
    unit=normalize([vector[j]-parameters["grand_mean"][j] for j in range(1024)])
    result=math.fsum(unit[j]*parameters["direction"][j] for j in range(1024))
    if not math.isfinite(result):raise ValueError("nonfinite score")
    return result
