"""Freeze an optional one-time CPU continuation; never rerun a completed case.

Operational change only: use part of the original five-minute export reserve
for unexecuted cases after the original conservative native stop, retaining the
same2700-second inclusive hard cap. No scores/thresholds/data/model change.
"""
from pathlib import Path
import difflib,hashlib,json

ROOT=Path('C:/Users/farha/Research4/artifacts/consistency_monitor_v1/native_cpu_20261012')
TARGET=ROOT/'continuation';TARGET.mkdir(exist_ok=False)
original=(ROOT/'source/worker.py').read_text()
code=original
def replace(old,new):
    global code
    assert code.count(old)==1,old[:90]
    code=code.replace(old,new)

replace("status={'status':'starting','completed':[],'unmonitored_controls':[],'GPU_allocations':0}","""policy=json.loads((ROOT/'continuation/POLICY.json').read_text())
assert policy['inclusive_hard_wall_seconds']==2700
plan['resources']['native_execution_stop_seconds']=policy['continuation_native_stop_seconds']
previous=json.loads((ROOT/'STATUS.json').read_text())
assert previous['status']=='fault_stop_no_scientific_retry'
assert previous.get('error')=='TimeoutError: Preserve export reserve'
assert 0<len(previous['completed'])<126
assert not (ROOT/'continuation/STARTED.json').exists()
(ROOT/'continuation/STARTED.json').write_text(json.dumps({'started_epoch':time.time(),'previous_completed':len(previous['completed']),'repeat_completed_cases':False},indent=2)+'\\n')
attempt=ROOT/'initial_attempt';attempt.mkdir(exist_ok=False)
import shutil
for name in ['STATUS.json','EXPORT_RECEIPT.json','EXPORT_HASHES.json','FAULT.txt']:
    shutil.copyfile(ROOT/name,attempt/name)
status=previous
status['status']='continuing_only_unexecuted_CPU_cases'
status.pop('error',None)
completed_keys={(r['prompt'],r['block_zero_based'],r['condition']) for r in status['completed']}
continuation_freeze=json.loads((ROOT/'continuation/FREEZE.json').read_text())
for name,digest in continuation_freeze['files'].items():
    assert hashlib.sha256((ROOT/'continuation'/name).read_bytes()).hexdigest()==digest,name""")
replace("archive=ROOT.with_name(ROOT.name+'_native_export.zip')","archive=ROOT.with_name(ROOT.name+'_continued_native_export.zip')")
replace("dump(ROOT/'MONITOR_UNIT_CHECKS.json',verify())","assert (ROOT/'MONITOR_UNIT_CHECKS.json').is_file()  # Preserve prior passed checks; no redundant rerun.")
replace("np.savez_compressed(ROOT/'DIRECTIONS_USED.npz',v=v,random=random)","""with np.load(ROOT/'DIRECTIONS_USED.npz',allow_pickle=False) as old_directions:
        assert np.array_equal(old_directions['v'],v) and np.array_equal(old_directions['random'],random)""")
replace("controls=ROOT/'unmonitored_controls';controls.mkdir(exist_ok=False)","controls=ROOT/'unmonitored_controls';assert controls.is_dir()")
replace("native=ROOT/'native';native.mkdir(exist_ok=False)","native=ROOT/'native';assert native.is_dir()")
old="""reference=run(prompt['prefix_ids'],None,'untreated',False)
        reference_hash=hashlib.sha256(reference['logits'].tobytes()).hexdigest()
        path=controls/(prompt['id']+'.npz')
        np.savez_compressed(path,generated_token_ids=reference['tokens'],raw_generation_logits=reference['logits'])
        status['unmonitored_controls'].append({'prompt':prompt['id'],'seconds':reference['seconds'],'sha256':sha(path),'logits_sha256':reference_hash})"""
new="""path=controls/(prompt['id']+'.npz')
        old_control=next((r for r in status['unmonitored_controls'] if r['prompt']==prompt['id']),None)
        if old_control is not None:
            assert sha(path)==old_control['sha256']
            with np.load(path,allow_pickle=False) as z:
                reference={'tokens':z['generated_token_ids'].copy(),'logits':z['raw_generation_logits'].copy()}
            reference_hash=hashlib.sha256(reference['logits'].tobytes()).hexdigest()
            assert reference_hash==old_control['logits_sha256']
        else:
            reference=run(prompt['prefix_ids'],None,'untreated',False)
            reference_hash=hashlib.sha256(reference['logits'].tobytes()).hexdigest()
            np.savez_compressed(path,generated_token_ids=reference['tokens'],raw_generation_logits=reference['logits'])
            status['unmonitored_controls'].append({'prompt':prompt['id'],'seconds':reference['seconds'],'sha256':sha(path),'logits_sha256':reference_hash})"""
replace(old,new)
replace("folder=native/prompt['id']/f'block{block:02d}';folder.mkdir(parents=True,exist_ok=False)","folder=native/prompt['id']/f'block{block:02d}';folder.mkdir(parents=True,exist_ok=True)")
replace("""for condition in order:
                deadline();r=run(prompt['prefix_ids'],block,condition)""","""for condition in order:
                if (prompt['id'],block,condition) in completed_keys:
                    prior=next(r for r in status['completed'] if (r['prompt'],r['block_zero_based'],r['condition'])==(prompt['id'],block,condition))
                    assert sha(folder/(condition+'.npz'))==prior['npz_sha256']
                    continue
                deadline();r=run(prompt['prefix_ids'],block,condition)""")
(TARGET/'worker.py').write_text(code)
(TARGET/'OPERATIONAL_DIFF.txt').write_text(''.join(difflib.unified_diff(original.splitlines(True),code.splitlines(True),fromfile='original_frozen_worker.py',tofile='CPU_continuation_worker.py')))
policy={'date':'2026-10-12','scope':'Optional single continuation of unexecuted CPU cases only; no scientific retry or extra cases',
    'inclusive_hard_wall_seconds':2700,'original_native_stop_seconds':2400,'continuation_native_stop_seconds':2580,
    'minimum_export_reserve_seconds':120,'model_monitor_threshold_questions_conditions_unchanged':True,
    'admission':'Only original conservative-deadline TimeoutError with0<completed<126; no numerical/scientific fault; original partial export verified; enough remaining time under original inclusive hard cap',
    'must_preserve':'Original source/plan/freeze and initial native ZIP/status/hash/fault receipts; every completed case hash; zero repeats of completed native calls',
    'stop':'No second continuation. If not complete within2580 native/2700 hard seconds, report partial results. GPU remains unallocated.'}
(TARGET/'POLICY.json').write_text(json.dumps(policy,indent=2)+'\n')
freeze={'files':{n:hashlib.sha256((TARGET/n).read_bytes()).hexdigest() for n in ['worker.py','POLICY.json','OPERATIONAL_DIFF.txt']},
    'native_scientific_source_original_sha256':hashlib.sha256(original.encode()).hexdigest(),
    'recorded_before_original_CPU_cutoff':True}
(TARGET/'FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
print(json.dumps({'continuation_prepared_not_started':True,'inclusive_hard_cap_unchanged':2700,'source_files':len(freeze['files'])}))
