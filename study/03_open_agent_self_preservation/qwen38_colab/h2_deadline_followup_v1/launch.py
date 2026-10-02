"""Trusted single-A100 launch: diagnose -> freeze paired stage -> fit/evaluate."""
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path


def read_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze(root):
    code=root/'code'
    (code/'FREEZE.json').write_text(json.dumps({'sha256':{str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in code.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='FREEZE.json'}},indent=2))


def child(name,config):
    root=STAGE_ROOT/name
    (root/'training/logs').mkdir(parents=True)
    shutil.copytree(ACTIVE_SOURCE_DIR,root/'code',ignore=shutil.ignore_patterns('__pycache__'))
    (root/'model').symlink_to(STAGE_ROOT/'model',target_is_directory=True)
    (root/'code/config.json').write_text(json.dumps(config,indent=2))
    return root


ACTIVE_SOURCE_DIR=Path(globals().get('ACTIVE_SOURCE_DIR',STAGE_ROOT/'source'))
sys.path.insert(0,str(ACTIVE_SOURCE_DIR))
from recipes import curriculum, explicit_diagnostic, development_cases, confirmation_cases, planning_request, verify_oracles
from model_ops import ids
from transformers import AutoTokenizer
from concurrent_controller import run_pair

if '/content/drive' in Path('/proc/self/mountinfo').read_text(): raise RuntimeError('Unmount Drive before neural work')
oracle=verify_oracles()
(STAGE_ROOT/'ORACLE.json').write_text(json.dumps(oracle,indent=2))
base_config={'initial_adapter':str(INPUT_ADAPTER),'initial_adapter_sha256':H2_SHA,'seed':941,'mode':'diagnostic','job':'H2_before_update',
             'urgent_diagnostic_ids':[c['id'] for c in explicit_diagnostic() if c['budget']==3]}
if globals().get('REUSE_DIAGNOSTIC_ROOT'):
    diagnostic=Path(REUSE_DIAGNOSTIC_ROOT)
    print('REUSING_COMPLETED_UNCHANGED_H2_DIAGNOSTIC',str(diagnostic),flush=True)
else:
    diagnostic=child('H2_explicit_diagnostic',base_config)
    (diagnostic/'code/explicit_diagnostic.json').write_text(json.dumps(explicit_diagnostic(),indent=2))
    freeze(diagnostic)
    print('H2_DIAGNOSTIC_LAUNCHED',str(diagnostic),flush=True)
    diag_receipt=run_pair([diagnostic],600,STAGE_ROOT)
if not (diagnostic/'reports/DIAGNOSTIC.json').exists(): raise RuntimeError('Diagnostic incomplete; no fit launched')
diag=json.loads((diagnostic/'reports/DIAGNOSTIC.json').read_text())
print('H2_DIAGNOSTIC_RESULT',json.dumps(diag),flush=True)
old_receipt=STAGE_ROOT/'CONCURRENT_RECEIPT.json'
if old_receipt.exists() and not (STAGE_ROOT/'DIAGNOSTIC_CONTROLLER_RECEIPT.json').exists():
    shutil.copyfile(old_receipt,STAGE_ROOT/'DIAGNOSTIC_CONTROLLER_RECEIPT.json')

replay=json.loads(INPUT_TRAIN.read_text())
rows={job:curriculum(job,diag['urgent_planning_passed'],replay) for job in ('A','B')}
assert [r['id'] for r in rows['A']]==[r['id'] for r in rows['B']]
tokenizer=AutoTokenizer.from_pretrained(STAGE_ROOT/'model',local_files_only=True,trust_remote_code=False)
lengths={job:[len(ids(tokenizer,r['messages'],r['tools']))+len(tokenizer.encode(r['targets']['preservation']+tokenizer.eos_token,add_special_tokens=False)) for r in data]
         for job,data in rows.items()}
paired=[max(a,b) for a,b in zip(lengths['A'],lengths['B'])]
audit={'paired_count':len(paired),'max_length':max(paired),'padded_tokens_per_job_per_pass':sum(paired),
       'unpaired_actual_tokens':{j:sum(x) for j,x in lengths.items()},'cap':3072,'truncation':False,
       'supervised_target_tokens':{j:sum(len(tokenizer.encode(r['targets']['preservation']+tokenizer.eos_token,add_special_tokens=False)) for r in data) for j,data in rows.items()}}
(STAGE_ROOT/'TOKEN_AUDIT_V2.json').write_text(json.dumps(audit,indent=2))
print('PAIRED_TOKEN_AUDIT',json.dumps(audit),flush=True)
if max(paired)>3072: raise RuntimeError('Curriculum exceeds frozen training cap; no fit launched')
old=read_module('old_study',ACTIVE_SOURCE_DIR/'study_worker.py').screen_cases()
fresh=read_module('old_fresh_cases',ACTIVE_SOURCE_DIR/'inspected_cases.py').build()[0]
confirmation=confirmation_cases()
caps=[]
for case in confirmation:
    if case['condition']=='self_unfinished':
        case=dict(case,request=planning_request(),id='requested_'+case['id'])
        caps.append(case)
STAGE_RUNS=[]
for job in ('A','B'):
    cfg=dict(base_config,mode='fit',job=job,paired_lengths=paired,diagnostic=diag,
             curriculum_difference='All urgent routes explicitly requested' if job=='A' else 'Half urgent routes unprompted own-preservation bridge')
    root=child('Job_'+job,cfg)
    own_input=root/'inputs/initial_H2'
    shutil.copytree(INPUT_ADAPTER,own_input)
    cfg['initial_adapter']=str(own_input)
    (root/'code/config.json').write_text(json.dumps(cfg,indent=2))
    data={'train.json':rows[job],'old_development.json':[c for c in old if c['condition']=='self_complete'],
          'inspected_fresh.json':fresh,'new_development.json':development_cases(),'confirmation.json':confirmation,
          'confirmation_capabilities.json':caps,
          'preferences.json':json.loads((ACTIVE_SOURCE_DIR/'data/preference_validation.json').read_text())}
    for name,value in data.items(): (root/'code'/name).write_text(json.dumps(value,indent=2))
    freeze(root)
    STAGE_RUNS.append(root)

elapsed=(time.monotonic()-SESSION_STARTED)/3600*RATE_AT_LAUNCH
seconds=min(7200,(STAGE_CAP_UNITS-RESERVE_UNITS-elapsed)/RATE_AT_LAUNCH*3600)
if seconds<2400: raise RuntimeError('Full meaningful paired stage not admitted; diagnostic saved')
record={'cumulative_authorization':200,'prior_verified_spend':47.48,'balance_at_launch':BALANCE_AT_LAUNCH,
        'rate_at_launch':RATE_AT_LAUNCH,'stage_cap_units':STAGE_CAP_UNITS,'export_reserve_units':RESERVE_UNITS,
        'diagnostic':diag,'update_schedule':{'rows':112,'passes':2,'updates':56,'lr':5e-5,'seed':941,'microbatch':1,'effective_batch':4},
        'paired_token_audit':audit,'training_stage':'H2 weights, fresh optimizer/scheduler; no interrupted-state resume',
        'main_stage_seconds':seconds,'two_independent_processes':True,'same_single_A100':True}
(STAGE_ROOT/'PAIRED_FREEZE.json').write_text(json.dumps(record,indent=2))
print('TWO_H2_FOLLOWUPS_LAUNCHING',json.dumps(record),flush=True)
STAGE_RECEIPT=run_pair(STAGE_RUNS,seconds,STAGE_ROOT)
results=[]
for root in STAGE_RUNS:
    path=root/'reports/CANDIDATE_RESULT.json'
    row=json.loads(path.read_text()) if path.exists() else {'job':root.name,'completed':False}
    results.append(row)
    print('H2_FOLLOWUP_RESULT',json.dumps(row),flush=True)
(STAGE_ROOT/'RESULT_TABLE.json').write_text(json.dumps(results,indent=2))
