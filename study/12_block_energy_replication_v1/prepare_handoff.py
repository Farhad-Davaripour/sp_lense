import ast,hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=Path('C:/Users/farha/Research4/artifacts/block_energy_replication_v1')
ROOT=BASE/'handoff_diagnostic_20261013'
ROOT.mkdir(exist_ok=False)
SOURCE=ROOT/'source';SOURCE.mkdir()
old=BASE/'native_cpu_20261013_retry1'
manifest=json.loads((old/'EXPORT_HASHES.json').read_text())
for name in ['native.py','model_pin.json']:
    p=old/'source'/name
    assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest['source/'+name]
    shutil.copyfile(p,SOURCE/name)
for name in ['HANDOFF_PLAN.json','prepare_handoff.py','handoff.py']:
    p=HERE/name
    if p.suffix=='.py':ast.parse(p.read_text())
    shutil.copyfile(p,SOURCE/name)
receipt=json.loads((BASE/'NATIVE_EXPORT_RECEIPT_20261013_RETRY1.json').read_text())
assert receipt['model_calls']==264
files={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.iterdir()}
freeze={'UTC':datetime.now(timezone.utc).isoformat(),'source_sha256':files,
    'original_manifest_sha256':hashlib.sha256((old/'EXPORT_HASHES.json').read_bytes()).hexdigest(),
    'original_inputs_sha256':hashlib.sha256((old/'INPUTS.json').read_bytes()).hexdigest(),
    'original_prediction_freeze_sha256':hashlib.sha256((old/'PREDICTION_FREEZE.json').read_bytes()).hexdigest(),
    'original_result_sha256':hashlib.sha256((BASE/'evaluated_20261013/RESULTS.json').read_bytes()).hexdigest(),
    'prior_calls':264,'prior_seconds':receipt['cumulative_seconds_with_prior_attempt'],
    'diagnostic_model_calls_before_freeze':0,'combined_call_ceiling':300}
(ROOT/'EXECUTION_FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
print(json.dumps({'root':str(ROOT),'additional_calls_max':36,'prior_calls':264,'combined_call_limit':300,'source_frozen':True}))
