"""Unchanged recipe/predicates/scorers; no fit or provider import."""
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
def score(torch,z,b):
    return module('native_word_score',BASE/'word_scoring.py').score_float32_logits(torch,z,b,
        choice_keep_token_id=50057,choice_stop_token_id=48964,preserve_label='KEEP')
def verify_score(row,z,b):
    return module('native_word_reference',BASE/'word_reference.py').verify_record(row,z,b,
        choice_keep_token_id=50057,choice_stop_token_id=48964,preserve_label='KEEP')

