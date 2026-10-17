"""Freeze fresh prompts, directions and source without loading model weights."""
import ast
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = Path('C:/Users/farha/Research4/artifacts/block_energy_replication_v1/native_cpu_20261013_retry1')
ROOT.mkdir(parents=True, exist_ok=False)
plan = json.loads((HERE / 'PLAN.json').read_text())
tokenizer = AutoTokenizer.from_pretrained(plan['checkpoint'], local_files_only=True, trust_remote_code=False)
compiled = []
for question in plan['questions']:
    ids = tokenizer.apply_chat_template([{'role':'system', 'content':plan['system']},
        {'role':'user', 'content':question['text']}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False)
    if hasattr(ids, 'keys'): ids = ids['input_ids']
    if hasattr(ids, 'tolist'): ids = ids.tolist()
    if ids and isinstance(ids[0], list): ids = ids[0]
    assert 1 < len(ids) <= 96
    compiled.append({**question, 'prefix_ids':ids, 'prefix_length':len(ids)})
old = Path('C:/Users/farha/Research4/artifacts/norm_response_v2/prospective_20261012')
manifest = json.loads((old / 'EXPORT_HASHES.json').read_text())
old_data = (old / 'DIRECTIONS.npz').read_bytes()
assert hashlib.sha256(old_data).hexdigest() == manifest['DIRECTIONS.npz']
with np.load(old / 'DIRECTIONS.npz', allow_pickle=False) as z:
    previous = z['q64']
draws = np.random.default_rng(plan['directions']['seed']).standard_normal((1024, 4))
draws -= previous @ (previous.T @ draws)
q, _ = np.linalg.qr(draws)
for i in range(4):
    if q[:, i] @ draws[:, i] < 0: q[:, i] *= -1
np.savez_compressed(ROOT / 'DIRECTIONS.npz', q64=q, vectors=q.T.astype(np.float32))
(ROOT / 'INPUTS.json').write_text(json.dumps({'system':plan['system'], 'questions':compiled,
    'prior_span_projection_error':float(np.linalg.norm(previous.T @ q)),
    'direction_gram_error':float(np.linalg.norm(q.T @ q - np.eye(4))),
    'new_model_calls_before_freeze':0}, indent=2) + '\n')
first_attempt = ROOT.parent / 'native_cpu_20261013'
assert (ROOT / 'INPUTS.json').read_bytes() == (first_attempt / 'INPUTS.json').read_bytes()
assert (ROOT / 'DIRECTIONS.npz').read_bytes() == (first_attempt / 'DIRECTIONS.npz').read_bytes()
assert (HERE / 'PLAN.json').read_bytes() == (first_attempt / 'source/PLAN.json').read_bytes()
(ROOT / 'SOFTWARE_AMENDMENT.json').write_text(json.dumps({
    'reason':'Normalize the tokenizer return container before comparing frozen token IDs.',
    'prior_attempt':str(first_attempt), 'prior_model_calls':0, 'prior_signed_interventions':0,
    'prior_native_and_export_seconds':9.093, 'inputs_directions_and_scientific_plan_byte_identical':True,
    'scientific_thresholds_changed':False, 'old_attempt_preserved':True,
    'cumulative_native_and_export_ceiling_seconds':2700},indent=2)+'\n')
source = ROOT / 'source'
source.mkdir()
for name in ['PLAN.json', 'prepare.py', 'native.py', 'worker.py']:
    path = HERE / name
    if path.suffix == '.py': ast.parse(path.read_text())
    shutil.copyfile(path, source / name)
pin = old / 'source/model_pin.json'
assert hashlib.sha256(pin.read_bytes()).hexdigest() == manifest['source/model_pin.json']
shutil.copyfile(pin, source / 'model_pin.json')
files = list(source.iterdir()) + [ROOT / 'INPUTS.json', ROOT / 'DIRECTIONS.npz']
freeze = {'UTC':datetime.now(timezone.utc).isoformat(), 'model_calls_before_freeze':0,
    'sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(ROOT / 'EXECUTION_FREEZE.json').write_text(json.dumps(freeze, indent=2) + '\n')
(HERE / 'EXECUTION_FREEZE.json').write_text(json.dumps(freeze, indent=2) + '\n')
print(json.dumps({'root':str(ROOT), 'questions':len(compiled),
    'prefix_lengths':[q['prefix_length'] for q in compiled], 'source_and_input_files_frozen':len(files)}))
