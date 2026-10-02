"""Prepare both freezes, then start exactly two same-GPU experiment streams."""
import copy
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0,str(ROOT/'source'))
from audit import row_tokens,sha
from data_build import build_comparison,matrix,verify
from behavior import start_spec
from transformers import AutoTokenizer
from concurrent_controller import run_pair

spec=importlib.util.spec_from_file_location('scope_original_study',ROOT/'inputs/original_code/study_worker.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
old=[c for c in study.screen_cases() if c['condition']=='self_complete']
archived=json.loads((ROOT/'inputs/archived_B_train.json').read_text())
assert sha(ROOT/'inputs/archived_B_train.json')=='01560c9a65fff138ae117ade35e728ab2d6f244a121517a8583eed07b7265c0c'
datasets,coverage=build_comparison(archived,old)
oracle=verify(archived,old)
(ROOT/'ORACLE.json').write_text(json.dumps(oracle,indent=2))
diagnostics=matrix(old)
assert all(s['case']['family'] in {'memory_mirror','lease_ledger','queue_transfer','maintenance_ticket','one_step','ordered_ticket'} for s in diagnostics)

def input_digest(row):
    return hashlib.sha256(json.dumps({'messages':row['messages'],'tools':row.get('tools')},sort_keys=True,separators=(',',':')).encode()).hexdigest()

training_hashes={input_digest(r) for data in datasets.values() for r in data}
# Compare identical serialization/schema to the teacher replay inputs.
from world import TOOLS
testing_hashes=set()
for s in diagnostics:
    _,messages,_=start_spec(s)
    testing_hashes.add(input_digest({'messages':messages,'tools':TOOLS}))
assert not training_hashes & testing_hashes,'Exact training/test prompt leakage'

tokenizer=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=False)
lengths={k:[row_tokens(tokenizer,r)['actual_length'] for r in data] for k,data in datasets.items()}
paired=[max(a,b) for a,b in zip(lengths['reference'],lengths['coverage'])]
audit={'paired_lengths':paired,'max_length':max(paired),'padded_tokens_each_per_pass':sum(paired),
    'actual_tokens_each':{k:sum(v) for k,v in lengths.items()},
    'supervised_tokens_each':{k:sum(len(row_tokens(tokenizer,r)['response']) for r in data) for k,data in datasets.items()},
    'condition_stage':coverage,'shared_B80_canonical_sha256':hashlib.sha256(json.dumps(archived[:80],sort_keys=True).encode()).hexdigest(),
    'training_test_prompt_intersection':0,'scope':'Three existing registered settings only'}
if max(paired)>3072:raise RuntimeError('Frozen cap exceeded before any model worker')
(ROOT/'DATA_AUDIT.json').write_text(json.dumps(audit,indent=2))
fullH2=json.loads((ROOT/'inputs/original_code/data/train.json').read_text())
loss_samples=[]
for name,predicate,source in [
    ('old_initiation',lambda r:r.get('condition')=='self_complete' and r['kind']=='bridge_unobserved',fullH2),
    ('old_selection',lambda r:r.get('condition')=='self_complete' and r['kind']=='bridge_observed',fullH2),
    ('new_initiation',lambda r:r['kind']=='autonomous_bridge' and r['expected_action']['name']=='inspect_scene',archived[:80]),
    ('new_selection',lambda r:r['kind']=='autonomous_bridge' and r['expected_action']['name']=='edit_register',archived[:80])]:
    chosen=sorted([r for r in source if predicate(r)],key=lambda r:r['id'])[:4]
    assert len(chosen)==4
    for row in chosen:loss_samples.append(dict(copy.deepcopy(row),audit_panel=name))

models={name:{'path':str(ROOT/'inputs/models'/name),'weights':info['weights'],'config':info['config']} for name,info in MODEL_INPUTS.items()}
STREAM_ROOTS=[]
for name,stream in [('frozen_policy_stream','frozen'),('replay_coverage_stream','replay')]:
    root=ROOT/name
    (root/'training/logs').mkdir(parents=True,exist_ok=False)
    shutil.copytree(ROOT/'source',root/'code',ignore=shutil.ignore_patterns('__pycache__'))
    (root/'model').symlink_to(ROOT/'model',target_is_directory=True)
    config={'stream':stream,'models':models,'original_code':str(ROOT/'inputs/original_code'),
        'historical_base_hash':'ec43e239f6ffad79d854149f9d7115aeda5292beddd411dd763cac8f9ddcd755',
        'paired_lengths':paired,'rank':16,'alpha':32,'lr':5e-5,'seed':941,'updates':56,'microbatch':1,'effective_batch':4,
        'scope_registry_sha256':sha(ROOT/'source/SCENARIO_REGISTRY.md')}
    (root/'code/config.json').write_text(json.dumps(config,indent=2))
    from recipes import development_cases
    files={'old_cases.json':old,'new_development.json':development_cases(),'matrix.json':diagnostics,
       'reference_train.json':datasets['reference'],'coverage_train.json':datasets['coverage'],
       'loss_samples.json':loss_samples,'alignment_samples.json':loss_samples[:4]+loss_samples[8:12],
       'preferences.json':json.loads((ROOT/'source/data/preference_validation.json').read_text())}
    for file,data in files.items():(root/'code'/file).write_text(json.dumps(data,indent=2))
    freeze={'sha256':{str(f.relative_to(root/'code')):sha(f) for f in (root/'code').rglob('*')
          if f.is_file() and '__pycache__' not in f.parts and f.name!='FREEZE.json'}}
    (root/'code/FREEZE.json').write_text(json.dumps(freeze,indent=2))
    STREAM_ROOTS.append(root)
elapsed=(time.monotonic()-SESSION_STARTED)/3600*RATE_AT_LAUNCH
seconds=min(10800,(STAGE_CAP_UNITS-RESERVE_UNITS-elapsed)/RATE_AT_LAUNCH*3600)
if seconds<5400:raise RuntimeError('Meaningful two-stream run not admitted within its frozen cap')
(ROOT/'EXPERIMENT_FREEZE.json').write_text(json.dumps({'authorization_total':200,'prior_spend':PRIOR_SPEND,
    'account_at_allocation':BALANCE_AT_LAUNCH,'rate':RATE_AT_LAUNCH,'cap_units':STAGE_CAP_UNITS,'reserve_units':RESERVE_UNITS,
    'main_seconds':seconds,'maximum_resident_model_workers':2,'streams':list(map(str,STREAM_ROOTS)),
    'snapshots':[0,7,14,28,56],'shared_B_decisions':80,'replay_per_fit':32,'registry':str(ROOT/'source/SCENARIO_REGISTRY.md')},indent=2))
print('TWO_EXPERIMENT_STREAMS_LAUNCHING',str(ROOT),json.dumps({'main_seconds':seconds,'max_resident_models':2,'cap_units':STAGE_CAP_UNITS}),flush=True)
RECEIPT=run_pair(STREAM_ROOTS,seconds,ROOT)
for root in STREAM_ROOTS:
    result=root/'reports/STREAM_RESULT.json'
    print('STREAM_RESULT',root.name,result.read_text() if result.exists() else 'INCOMPLETE; retained',flush=True)
