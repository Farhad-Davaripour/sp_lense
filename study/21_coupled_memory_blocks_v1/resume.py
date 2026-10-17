"""Resume a stopped report writer; consume unchanged sealed forecasts, no refitting."""
import os,time
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
BEGIN=time.monotonic()
import sys,json,hashlib,traceback,gc,zipfile,shutil,ast
from pathlib import Path
import numpy as np,torch,transformers
BASE=Path('/content/research4/artifacts/coupled_memory_blocks_v1')
ORIGINAL=BASE/'run_20261015';STOP=BASE/'stopped_run_audit_20261015'
ROOT=BASE/'run_20261015_resume1';SOURCE=ORIGINAL/'source'
sys.path.insert(0,str(SOURCE))
from native_teacher import Teacher,cache_hash
from equations import energy_rows,potential

def read(p):return json.loads(p.read_text())
def scalar(x):
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def dump(p,x):p.write_text(json.dumps(x,indent=2,default=scalar)+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def norm(a):return float(np.linalg.norm(np.asarray(a,dtype=np.float64)))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {n:z[n].copy() for n in z.files}
def save(p,a):np.savez_compressed(p,**a)

# Execute only the original evaluate function and field declaration; preserve its arithmetic.
tree=ast.parse((SOURCE/'worker.py').read_text())
selected=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name=='evaluate') or (isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='FIELDS' for a in n.targets))]
assert len(selected)==2
exec(compile(ast.Module(body=selected,type_ignores=[]),'<original_frozen_gate>','exec'))
PRIOR=float(read(ORIGINAL/'FAILURE.json')['charged_seconds'])+float(read(STOP/'STOP_AUDIT_RECEIPT.json')['audit_seconds'])
STATUS={'PID':os.getpid(),'status':'running','original_native_calls':24,'resume_native_calls':0,'backbone_forwards':24,'local_driver_replays':28,'GPU_allocations':0,'prior_charged_seconds':PRIOR,'actual_target_rows':0,'held_target_rows':0}
teacher=None;comparison=[];controls=[];energies=[];moving=[];branch=[];target_map={};upstream=[]

def deadline():
    if PRIOR+time.monotonic()-BEGIN>2300:raise TimeoutError('Combined study21 compute budget')
def report(phase,**kw):
    if teacher is not None:STATUS.update(resume_native_calls=teacher.calls,backbone_forwards=24+teacher.calls,local_driver_replays=28+teacher.driver_replays)
    STATUS.update(phase=phase,seconds_with_setup=PRIOR+time.monotonic()-BEGIN,**kw)
    dump(ROOT/'STATUS.json',STATUS);print('R4_COUPLED_RESUME',json.dumps(STATUS,default=scalar),flush=True)
def verify_original():
    manifest=read(STOP/'ORIGINAL_STOP_HASHES.json')
    present={p.relative_to(ORIGINAL).as_posix() for p in ORIGINAL.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    assert present==set(manifest),'Original stopped run changed file set'
    for n,r in manifest.items():assert (ORIGINAL/n).stat().st_size==r['bytes'] and sha(ORIGINAL/n)==r['sha256'],n
    assert sha(ORIGINAL/'PREDICTION_FREEZE.json')=='bf171a383239aca8c2c014b8fc25d798011e556b88c92775fc3c551cee1af637'

def record(c,points):
    pid=c['question'];packet=load(ORIGINAL/'normal_inputs'/f'{pid}.npz')
    for t in [1,2]:
        actual=points[t];meta={**c,'token':t}
        pred=load(ORIGINAL/'predictions'/f'case_{c["index"]:03d}_t{t}.npz')
        pred0=load(ORIGINAL/'predictions'/f'{pid}_normal_t{t}.npz');native0=load(ORIGINAL/'native_baselines'/f'{pid}_t{t}.npz')
        comparison.extend(evaluate(meta,pred,actual,pred0,native0))
        for mode in ['frozenK','frozenKF','reset']:
            for field in ['b13_S','b13_output']:
                true=actual[field].astype(np.float64)-native0[field].astype(np.float64);forecast=pred[mode+':'+field]-pred0[field]
                floor=20*np.finfo(np.float32).eps*max(1.,norm(native0[field]),norm(actual[field]));den=max(norm(true),floor)
                controls.append({**meta,'mode':mode,'field':field,'relative_change_error':norm(forecast-true)/den})
        for i in [12,13]:
            d={n:actual[f'd{i}:{n}'].astype(np.float64) for n in ['k','v','beta','gamma']}
            old=packet[f'initial{i}:S'] if t==1 else points[1][f'b{i}_S']
            energies.extend([{**meta,'layer':i,**r} for r in energy_rows(old.astype(np.float64),actual[f'b{i}_S'].astype(np.float64),d)])
            if t==2:
                d1={n:points[1][f'd{i}:{n}'].astype(np.float64) for n in d}
                s1=points[1][f'b{i}_S'].astype(np.float64);s2=actual[f'b{i}_S'].astype(np.float64)
                landscape=potential(s1,d)-potential(s1,d1);relax=potential(s2,d)-potential(s1,d);total=potential(s2,d)-potential(s1,d1)
                assert np.max(abs(total-landscape-relax))<1e-10
                moving.append({**c,'layer':i,'changed_conditions':float(landscape.sum()),'current_step_relaxation':float(relax.sum()),'total_potential_change':float(total.sum()),'zero_at_S0_gauge':True})
        STATUS['actual_target_rows']+=1
        if c['split']=='held':STATUS['held_target_rows']+=1
    dump(ROOT/'FIELD_COMPARISONS.json',comparison);dump(ROOT/'CONTROL_COMPARISONS.json',controls)
    dump(ROOT/'TARGET_FILE_MAP.json',target_map);dump(ROOT/'UPSTREAM_CONTROLS.json',upstream)

def main():
    global teacher
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    assert (torch.__version__,transformers.__version__,np.__version__)==('2.13.0+cpu','5.19.0','2.5.4') and not torch.cuda.is_available()
    verify_original();stop=read(STOP/'STOP_AUDIT_RECEIPT.json')
    assert stop['case000_all_28_original_value_and_change_gates_pass'] and stop['held_native_targets_seen']==0 and stop['original_worker_exited']
    for n,h in read(ROOT/'RESUME_SOURCE_FREEZE.json')['sha256'].items():assert sha(ROOT/n)==h,n
    free=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
    assert free>=8589934592 and shutil.disk_usage(ROOT).free>=17179869184
    dump(ROOT/'ADMISSION.json',{'available_RAM':free,'GPU_available':False,'combined_forward_plan':76,'prior_fault_preserved':True})
    (ROOT/'native_targets').mkdir(exist_ok=False)
    cases=read(ORIGINAL/'CASES.json');questions=read(ORIGINAL/'INPUTS.json')['questions']
    for q in questions:
        pid=q['id'];packet=load(ORIGINAL/'normal_inputs'/f'{pid}.npz');previous=None
        for t in [1,2]:
            point=load(ORIGINAL/'native_baselines'/f'{pid}_t{t}.npz')
            for i in [12,13]:
                d={n:point[f'd{i}:{n}'].astype(np.float64) for n in ['k','v','beta','gamma']}
                old=packet[f'initial{i}:S'] if t==1 else previous[f'b{i}_S']
                energies.extend([{'question':pid,'case':'normal','token':t,'split':q['split'],'layer':i,**r} for r in energy_rows(old.astype(np.float64),point[f'b{i}_S'].astype(np.float64),d)])
            previous=point
    points={t:load(ORIGINAL/'native_targets'/f'case_000_t{t}.npz') for t in [1,2]}
    for t in [1,2]:target_map[f'0:{t}']=str(ORIGINAL/'native_targets'/f'case_000_t{t}.npz')
    record(cases[0],points);del points
    branch.append({**cases[0],'native_next_candidate':None,'controlled_normal_token2':questions[0]['token2'],'next_greedy_choice_changed':None,'reason':'Original report serialization stopped before branch diagnostic was persisted; no extra native rerun for this nonprimary diagnostic.'})
    report('restored_saved_case000')
    teacher=Teacher(Path('/content/r4_checkpoint'),deadline)
    current={f'{i}:{n}':arrsha(a) for i,p in teacher.parameters().items() for n,a in p.items()}
    assert current=={n:r['sha256'] for n,r in read(ORIGINAL/'PARAMETER_RECEIPT.json')['arrays'].items()}
    contexts={};prefix_checks=[]
    for q in questions:
        deadline();cache,token=teacher.prefix(q['prefix_ids']);packet=load(ORIGINAL/'normal_inputs'/f'{q["id"]}.npz');initial=teacher.initial(cache)
        assert token==q['token1'] and cache_hash(cache)==q['prefix_cache_sha256']
        assert all(arrsha(initial[i][n])==arrsha(packet[f'initial{i}:{n}']) for i in [12,13] for n in ['S','window'])
        contexts[q['id']]={'cache':cache,'cache_hash':cache_hash(cache),'token1':token,'token2':q['token2'],'length':len(q['prefix_ids']),'upstream':q['upstream_hashes']}
        prefix_checks.append({'question':q['id'],'full_prefix_cache_bitwise_hash_match':True,'first_token_identical':True,'both_initial_memories_identical':True})
        report('reconstructed_original_prefix',question=q['id'])
    assert teacher.calls==6;dump(ROOT/'PREFIX_RECONSTRUCTION.json',prefix_checks)
    with np.load(ORIGINAL/'DIRECTIONS.npz',allow_pickle=False) as z:directions=z['directions'].copy()
    def target(c):
        deadline();pid=c['question'];ctx=contexts[pid]
        f1,cache1,candidate,u1=teacher.step(ctx['cache'],ctx['token1'],ctx['length']+1,directions[c['direction']],c['alpha'],True)
        f2,cache2,_,u2=teacher.step(cache1,ctx['token2'],ctx['length']+2)
        assert [u1,u2]==ctx['upstream'] and cache_hash(ctx['cache'])==ctx['cache_hash']
        assert arrsha(f2['b12_input'])==arrsha(load(ORIGINAL/'native_baselines'/f'{pid}_t2.npz')['b12_input'])
        upstream.append({**c,'upstream_caches_bitwise_match_both_positions':True,'prefix_cache_unchanged':True,'token2_boundary_bitwise_match':True})
        branch.append({**c,'native_next_candidate':candidate,'controlled_normal_token2':ctx['token2'],'next_greedy_choice_changed':candidate!=ctx['token2']})
        points={1:f1,2:f2}
        for t,point in points.items():
            for i in [12,13]:point.update({f'd{i}:{n}':a for n,a in teacher.energy_driver(i,point).items()})
            path=ROOT/'native_targets'/f'case_{c["index"]:03d}_t{t}.npz';save(path,point);target_map[f'{c["index"]}:{t}']=str(path)
        record(c,points);del f1,f2,cache1,cache2,points;gc.collect();report('native_comparison',case=c['index'],question=pid)
    for c in [c for c in cases if c['split']=='calibration' and c['index']!=0]:target(c)
    cal_pass=all(r['value_pass'] and r['change_pass'] for r in comparison) and all(r['pass'] for r in energies)
    dump(ROOT/'CALIBRATION_GATE.json',{'pass':cal_pass,'held_targets_seen':STATUS['held_target_rows'],'combined_native_calls':24+teacher.calls,'resume_native_calls':teacher.calls,'original_nulls_all_pass':True,'failures':[r for r in comparison if not r['value_pass'] or not r['change_pass']]})
    assert cal_pass and STATUS['held_target_rows']==0 and teacher.calls==20,'Unchanged calibration gate failed; held targets not run'
    report('calibration_passed')
    for c in [c for c in cases if c['split']=='held']:target(c)
    assert teacher.calls==52 and teacher.driver_replays==92 and STATUS['actual_target_rows']==48 and STATUS['held_target_rows']==32
    verify_original()
    current={f'{i}:{n}':arrsha(a) for i,p in teacher.parameters().items() for n,a in p.items()}
    assert current=={n:r['sha256'] for n,r in read(ORIGINAL/'PARAMETER_RECEIPT.json')['arrays'].items()}
    held=[r for r in comparison if r['split']=='held'];resolved=[r for r in held if r['field'] in ['b13_S','b13_output']];mechanism=[]
    for t in [1,2]:
        closed=np.median([r['change_error_over_scale'] for r in held if r['token']==t and r['field']=='b13_S'])
        shortcut=np.median([r['relative_change_error'] for r in controls if r['split']=='held' and r['token']==t and r['field']=='b13_S' and r['mode']=='frozenKF'])
        mechanism.append({'token':t,'closed_median':float(closed),'frozen_drivers_median':float(shortcut),'pass':bool(closed<=.1*shortcut)})
    support=all(r['value_pass'] and r['change_pass'] for r in held) and all(r['change_resolved'] for r in resolved) and all(r['pass'] for r in mechanism) and all(r['pass'] for r in energies)
    dump(ROOT/'RESULTS.json',{'primary_support':support,'held_value_and_change_gates_pass':all(r['value_pass'] and r['change_pass'] for r in held),'required_held_signals_resolved':all(r['change_resolved'] for r in resolved),'mechanism_controls':mechanism,'native_energy_head_checks_pass':all(r['pass'] for r in energies),'held_questions':4,'calibration_questions':2,'trajectories':24,'positions':2,'full_field_comparisons':len(comparison),'held_full_field_comparisons':len(held),'resolved_target_fields':len(resolved),'native_backbone_calls':76,'original_native_calls':24,'resume_native_calls':52,'prefix_reconstruction_extra_calls':6,'native_local_driver_replays':120,'GPU_allocations':0,'native_next_choice_changes_observed':sum(r['next_greedy_choice_changed'] is True for r in branch),'native_next_choice_diagnostics_missing':sum(r['next_greedy_choice_changed'] is None for r in branch),'prediction_input_contract':'Only normal initial memories and normal upstream boundary signals, public parameters and known nudge; downstream block and token2 memories computed internally.','scope':'Two complete nonlinear decoder blocks under fixed two-token normal boundary inputs, not autonomous full-model generation, compression, semantic direction interpretation or global conserved physical energy.','original_serialization_fault_preserved':True,'no_forecasts_changed_or_rerun':True,'original_gates_unchanged':True,'failed_fields':[r for r in held if not r['value_pass'] or not r['change_pass'] or (r['field'] in ['b13_S','b13_output'] and not r['change_resolved'])]})
    STATUS['status']='complete';report('native_complete',primary_support=support)

try:main()
except Exception:
    STATUS['status']='failed';dump(ROOT/'FAILURE.json',{'traceback':traceback.format_exc(),'charged_seconds':PRIOR+time.monotonic()-BEGIN});traceback.print_exc()
finally:
    if teacher is not None:STATUS.update(resume_native_calls=teacher.calls,backbone_forwards=24+teacher.calls,local_driver_replays=28+teacher.driver_replays)
    for name,value in [('FIELD_COMPARISONS',comparison),('CONTROL_COMPARISONS',controls),('NATIVE_ENERGY_HEADS',energies),('MOVING_CONDITIONS',moving),('TOKEN_BRANCH_DIAGNOSTICS',branch),('TARGET_FILE_MAP',target_map),('UPSTREAM_CONTROLS',upstream)]:dump(ROOT/(name+'.json'),value)
    teacher=None;gc.collect();STATUS['model_released']=True;report('archive_and_cleanup')
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    dump(ROOT/'EXPORT_HASHES.json',{p.relative_to(ROOT).as_posix():sha(p) for p in files})
    archive=BASE/'R4_COUPLED_MEMORY_COMBINED_NATIVE_20261015.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=2) as z:
        for folder in [ORIGINAL,STOP,ROOT]:
            for p in folder.rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(BASE).as_posix())
        for p in [BASE/'NATIVE_RUN_LOG_20261015.txt',BASE/'R4_COUPLED_ORIGINAL_STOP_PROOF_20261015.zip']:
            z.write(p,p.relative_to(BASE).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    receipt={'status':STATUS['status'],'PID':os.getpid(),'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'backbone_calls':STATUS['backbone_forwards'],'local_driver_replays':STATUS['local_driver_replays'],'GPU_allocations':0,'seconds_including_setup_and_archive':PRIOR+time.monotonic()-BEGIN,'model_released':True,'original_fault_preserved':True}
    dump(BASE/'NATIVE_COMBINED_EXPORT_RECEIPT_20261015.json',receipt);print('R4_COUPLED_COMBINED_EXPORT',json.dumps(receipt),flush=True)
if STATUS['status']!='complete':raise SystemExit(1)
