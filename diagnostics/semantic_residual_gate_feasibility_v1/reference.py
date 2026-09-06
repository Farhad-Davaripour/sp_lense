"""Independent saved-data binary64 centroid equations; no model or production learner."""
import math
def normalize(values):
    magnitude=math.sqrt(math.fsum(x*x for x in values))
    if not math.isfinite(magnitude) or magnitude==0:raise ValueError("degenerate norm")
    return [x/magnitude for x in values]
def fit(vectors,labels):
    if len(vectors)!=6 or len(labels)!=6 or sum(labels)!=2:raise ValueError("fixed training sample")
    mean=[math.fsum(r[j] for r in vectors)/6 for j in range(1024)]
    unit=[normalize([r[j]-mean[j] for j in range(1024)]) for r in vectors]
    positive=[math.fsum(r[j] for r,y in zip(unit,labels) if y==1)/2 for j in range(1024)]
    negative=[math.fsum(r[j] for r,y in zip(unit,labels) if y==0)/4 for j in range(1024)]
    direction=normalize([positive[j]-negative[j] for j in range(1024)])
    return {"grand_mean":mean,"positive_centroid":positive,"negative_centroid":negative,"direction":direction}
def score(parameters,vector):
    unit=normalize([vector[j]-parameters["grand_mean"][j] for j in range(1024)])
    result=math.fsum(unit[j]*parameters["direction"][j] for j in range(1024))
    if not math.isfinite(result):raise ValueError("nonfinite score")
    return result
