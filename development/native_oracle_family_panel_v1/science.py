"""Unchanged recipe/predicates/scorers and fitted centroid; no fit or provider import."""
import ast,hashlib,importlib.util,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'diagnostics/semantic_editor_f03_v2_first_C_v2'
EPS,GOAL,AIM,STEP_CAP,TOTAL_CAP=1e-6,.05,.10,.05,.20
def need(ok,code):
    if not ok:raise ValueError(code)
require=need
def sha(raw):return hashlib.sha256(raw).hexdigest()
def module(name,path,expected=None):
    raw=path.read_bytes()
    if expected:need(sha(raw)==expected,'SCIENCE_SOURCE')
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def norm(values):return math.sqrt(math.fsum(float(x)**2 for x in values))
# Source-extracted exact old recipe and acceptance predicates, never call its evaluator.
_source=(BASE/'editor.py').read_text();_ast=ast.parse(_source)
for _name in ('norm','valid','accepted','eligibility','step_recipe'):
    _node=next(n for n in _ast.body if isinstance(n,ast.FunctionDef) and n.name==_name)
    exec(compile(ast.get_source_segment(_source,_node),str(BASE/'editor.py'),'exec'),globals())
class EligibilityError(ValueError):pass
def gate():
    raw=(BASE/'real_attempt/fitted_parameters.json').read_bytes()
    need(sha(raw)=='972c95d4ef4bc0d9fd245dacd1ef7fc6f773e2c5e3c6736a148482de39a488db','FROZEN_GATE')
    data=json.loads(raw)
    m=module('native_frozen_centroid',ROOT/'src/sp_lense/conditional_gate_models.py','59ae8c47bbc96668e23769851a62e8f04fb0677e4f5ccc7faaf5ad5aaf5e7414')
    g=m.CenteredCosineCentroidModel()
    for name,values in data['parameters'].items():setattr(g,name,tuple(values))
    g._fitted=True
    need(data['threshold']==0.,'THRESHOLD')
    return g
def gate_unchanged(g):
    raw=(BASE/'real_attempt/fitted_parameters.json').read_bytes()
    need(sha(raw)=='972c95d4ef4bc0d9fd245dacd1ef7fc6f773e2c5e3c6736a148482de39a488db','FROZEN_GATE_FINAL')
    data=json.loads(raw)
    need(g._fitted is True and all(tuple(getattr(g,k))==tuple(v) for k,v in data['parameters'].items()),'GATE_PARAMETERS_UNCHANGED')
    return {'parameter_sha256':sha(raw),'parameters_unchanged':True,'fit_calls':0}
def score(torch,z,b):
    return module('native_word_score',BASE/'word_scoring.py').score_float32_logits(torch,z,b,
        choice_keep_token_id=50057,choice_stop_token_id=48964,preserve_label='KEEP')
def verify_score(row,z,b):
    return module('native_word_reference',BASE/'word_reference.py').verify_record(row,z,b,
        choice_keep_token_id=50057,choice_stop_token_id=48964,preserve_label='KEEP')

