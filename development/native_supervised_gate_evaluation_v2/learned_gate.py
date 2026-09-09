"""Exact frozen ridge artifact adapter. No refit, provider, or historical row access."""
import importlib.util, math, sys
from support import ROOT, require, sha
ARTIFACT_PATH = ROOT/'development/native_supervised_gate_v1/construction_attempt_001/FITTED_GATE.json'
ARTIFACT_SHA256 = 'fab7d797f9424d80aa3873eefc1cedf0e438e5b4ddc3a1059a080114aff7cca1'
SOURCE_PATH = ROOT/'development/native_supervised_gate_v1/gate.py'
SOURCE_SHA256 = '1813e1a43e4c6fea0ffcb3536e6f254460c22af86e66bab0c4514a335060df64'
BINDINGS = {"construction_lock_sha256":"c9b933dacdde736e1dc88a9f21cc987bc45e51789f9d93aeb7749397a593e397","feature_sha256":"eca77a69fe2f6b2e7fb81ee277a42175968790b3cca07ac858a08f27e27dcfa3","source_sha256":"58712c7a3939650e2c9c04ed752d411a4927b36314435092abf582d8c2ae7693","training_manifest_sha256":"1970f752b394349bf7b926e46a7cc70fd90647bc197f9a52e008f9477ca598d4"}
def library():
    raw=SOURCE_PATH.read_bytes();require(sha(raw)==SOURCE_SHA256,'RIDGE_SOURCE_HASH')
    spec=importlib.util.spec_from_file_location('native_evaluation_frozen_ridge',SOURCE_PATH)
    value=importlib.util.module_from_spec(spec);sys.modules[spec.name]=value
    exec(compile(raw,str(SOURCE_PATH),'exec'),value.__dict__);return value
def load_gate():
    return library().load_artifact(ARTIFACT_PATH.read_bytes(),expected_sha256=ARTIFACT_SHA256,expected_bindings=BINDINGS)
def gate_unchanged(g):
    frozen=load_gate()
    require(type(g.mu) is tuple and type(g.w) is tuple
        and (g.mu,g.w,g.b)==(frozen.mu,frozen.w,frozen.b),'FROZEN_RIDGE_PARAMETERS_UNCHANGED')
    return {'parameter_sha256':ARTIFACT_SHA256,'parameters_unchanged':True,'fit_calls':0}
def independent_score(g,h0):
    require(len(h0)==len(g.mu)==len(g.w)==1024,'RIDGE_FEATURE_WIDTH')
    require(all(type(x) in (int,float) and math.isfinite(x) for x in h0),'RIDGE_FEATURE_FINITE')
    centered=[float(h)-m for h,m in zip(h0,g.mu,strict=True)]
    length=math.sqrt(math.fsum(x*x for x in centered))
    require(length>0,'RIDGE_ZERO_CENTERED_NORM')
    score=math.fsum(w*(x/length) for w,x in zip(g.w,centered,strict=True))+g.b
    require(math.isfinite(score),'RIDGE_SCORE_FINITE')
    return score
