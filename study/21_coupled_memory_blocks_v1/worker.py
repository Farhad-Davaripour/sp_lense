"""Prospective two-block experiment with sealed forecasts and native targets."""
import os,time
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
BEGIN=time.monotonic()
import sys,json,hashlib,subprocess,traceback,gc,zipfile,shutil
from pathlib import Path
import numpy as np,torch,transformers
ROOT=Path('/content/research4/artifacts/coupled_memory_blocks_v1/run_20261015');SOURCE=ROOT/'source'
sys.path.insert(0,str(SOURCE))
from native_teacher import Teacher,array,cache_hash
from equations import energy_rows,potential
SETUP=json.loads(Path('/content/research4/artifacts/coupled_memory_blocks_v1/setup_20261015/SETUP_RECEIPT.json').read_text())
PRIOR=float(SETUP['setup_seconds'])+float(json.loads((ROOT/'PREFLIGHT.json').read_text())['charged_host_seconds'])
STATUS={'PID':os.getpid(),'status':'running','backbone_forwards':0,'local_driver_replays':0,'GPU_allocations':0,'prior_setup_seconds':PRIOR,'actual_target_rows':0,'held_target_rows':0}
teacher=None;comparison=[];baseline_rows=[];controls=[];nulls=[];energies=[];moving=[];branch=[]
def read(p):return json.loads(p.read_text())
def json_scalar(x):
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def dump(p,x):p.write_text(json.dumps(x,indent=2,default=json_scalar)+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for a in iter(lambda:f.read(8*1024*1024),b''):h.update(a)
    return h.hexdigest()
def norm(a):return float(np.linalg.norm(np.asarray(a,dtype=np.float64)))
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {n:z[n].copy() for n in z.files}
def save(p,a):np.savez_compressed(p,**a)
def deadline():
    if PRIOR+time.monotonic()-BEGIN>2300:raise TimeoutError('Study21 compute budget including prior setup')
def report(phase,**kw):
    STATUS.update(phase=phase,seconds_with_setup=PRIOR+time.monotonic()-BEGIN,**kw)
    if teacher is not None:STATUS.update(backbone_forwards=teacher.calls,local_driver_replays=teacher.driver_replays)
    dump(ROOT/'STATUS.json',STATUS);print('R4_COUPLED',json.dumps(STATUS),flush=True)
FIELDS=[f'b{i}_{n}' for i in [12,13] for n in ['input','mixer','middle','mlp','output','window','S']]
def evaluate(meta,pred,native,pred0=None,native0=None):
    rows=[]
    for n in FIELDS:
        assert pred[n].shape==native[n].shape
        error=norm(pred[n]-native[n]);den=norm(native[n]);value_pass=error<=1e-5*max(1.,den)
        row={**meta,'field':n,'value_error':error,'native_norm':den,'value_pass':value_pass}
        if pred0 is not None:
            true=native[n].astype(np.float64)-native0[n].astype(np.float64);forecast=pred[n]-pred0[n]
            delta_norm=norm(true);floor=20*np.finfo(np.float32).eps*max(1.,norm(native0[n]),den)
            err=norm(forecast-true);bound=max(.01*delta_norm,floor)
            row.update(native_change_norm=delta_norm,predicted_change_norm=norm(forecast),change_error=err,numeric_floor=float(floor),change_bound=float(bound),change_error_over_scale=err/max(delta_norm,floor),change_pass=bool(err<=bound),change_resolved=bool(delta_norm>5*floor))
        rows.append(row)
    return rows
def add_drivers_and_energy(meta,fields,old):
    result=dict(fields)
    for i in [12,13]:
        d=teacher.energy_driver(i,fields)
        result.update({f'd{i}:{n}':a for n,a in d.items()})
        energies.extend([{**meta,'layer':i,**r} for r in energy_rows(old[i].astype(np.float64),fields[f'b{i}_S'].astype(np.float64),{n:a.astype(np.float64) for n,a in d.items()})])
    return result
def main():
    global teacher
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    assert (torch.__version__,transformers.__version__,np.__version__)==('2.13.0+cpu','5.19.0','2.5.4') and not torch.cuda.is_available()
    free=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
    dump(ROOT/'ADMISSION.json',{'free_memory_after_imports':free,'minimum':8589934592,'disk_free':shutil.disk_usage(ROOT).free,'local_host_free_KiB_at_admission':5456920,'CPU_runtime_confirmed_in_UI':True})
    assert free>=8589934592 and shutil.disk_usage(ROOT).free>=17179869184
    shutil.copyfile('/content/research4/artifacts/coupled_memory_blocks_v1/setup_20261015/SETUP_RECEIPT.json',ROOT/'PRIOR_SETUP_RECEIPT.json')
    shutil.copyfile('/content/research4/artifacts/coupled_memory_blocks_v1/setup_20261015/SETUP_LOG.txt',ROOT/'PRIOR_SETUP_LOG.txt')
    freeze=read(ROOT/'EXECUTION_FREEZE.json')
    for n,h in freeze['sha256'].items():assert sha(ROOT/n)==h,n
    plan=read(SOURCE/'PLAN.json');assert SETUP['complete'] and SETUP['new_backbone_forwards']==0 and PRIOR<1200
    for n in ['normal_inputs','native_baselines','native_targets']:(ROOT/n).mkdir(exist_ok=False)
    report('loading_original_frozen_model');teacher=Teacher(Path('/content/r4_checkpoint'),deadline)
    params=teacher.parameters();flat={f'{i}:{n}':a for i,p in params.items() for n,a in p.items()}
    save(ROOT/'PARAMETERS.npz',flat)
    dump(ROOT/'PARAMETER_RECEIPT.json',{'loaded_A_log_dtype_before_promotion':teacher.loaded_A_log_dtype,'effective_dtype':'float32','arrays':{n:{'shape':list(a.shape),'sha256':arrsha(a)} for n,a in flat.items()},'model_revision':SETUP['model_revision'],'original_trained_parameters_changed':False})
    del params,flat;gc.collect()
    with np.load(ROOT/'DIRECTIONS.npz',allow_pickle=False) as z:directions=z['directions'].copy()
    assert directions.shape==(2,1024) and np.allclose(directions@directions.T,np.eye(2),atol=2e-6)
    contexts={};questions=[];initials={}
    for q in plan['questions']:
        deadline();pid=q['id'];ids=teacher.encode(q['text'],plan['input_template']['system']);cache0,token1=teacher.prefix(ids)
        initial=teacher.initial(cache0);initials[pid]=initial
        first,cache1,token2,up1=teacher.step(cache0,token1,len(ids)+1,choose_next=True)
        eos={248044,248046,int(teacher.tokenizer.eos_token_id)};assert token2 not in eos,'Second normal token is EOS; no forced continuation'
        second,cache2,_,up2=teacher.step(cache1,token2,len(ids)+2)
        packet={f'initial{i}:{n}':a for i,s in initial.items() for n,a in s.items()}
        packet.update(boundary_t1=first['b12_input'],boundary_t2=second['b12_input']);save(ROOT/'normal_inputs'/f'{pid}.npz',packet)
        for t,point,old in [(1,first,{i:initial[i]['S'] for i in [12,13]}),(2,second,{i:first[f'b{i}_S'] for i in [12,13]})]:
            point=add_drivers_and_energy({'question':pid,'case':'normal','token':t,'split':q['split']},point,old)
            save(ROOT/'native_baselines'/f'{pid}_t{t}.npz',point)
        contexts[pid]={'cache':cache0,'cache_hash':cache_hash(cache0),'token1':token1,'token2':token2,'length':len(ids),'upstream':[up1,up2]}
        questions.append({**q,'prefix_ids':ids,'token1':token1,'token2':token2,'prefix_cache_sha256':contexts[pid]['cache_hash'],'upstream_hashes':[up1,up2]})
        del first,second,cache1,cache2;gc.collect();report('normal_boundary_packet',question=pid)
    assert teacher.calls==18
    dump(ROOT/'INPUTS.json',{'system':plan['input_template']['system'],'questions':questions,'normal_native_forwards':18})
    cases=[{'index':j,'question':q['id'],'split':q['split'],'direction':d,'alpha':alpha} for j,(q,d,alpha) in enumerate((q,d,alpha) for q in questions for d in range(2) for alpha in [.125,-.125])]
    dump(ROOT/'CASES.json',cases)
    allowed=[ROOT/'PARAMETERS.npz',ROOT/'DIRECTIONS.npz',ROOT/'INPUTS.json',ROOT/'CASES.json',*sorted((ROOT/'normal_inputs').glob('*.npz'))]
    dump(ROOT/'NORMAL_INPUT_FREEZE.json',{'native_steered_targets_seen':0,'schema':'Only public parameters, known nudge and six-field normal input packets','sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in allowed}})
    baseline_hash={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'native_baselines').iterdir()}
    dump(ROOT/'NATIVE_BASELINE_FREEZE.json',baseline_hash)
    assert not list((ROOT/'native_targets').iterdir());report('separate_predictor_start')
    p=subprocess.Popen([sys.executable,'-B',str(SOURCE/'predict.py'),str(ROOT)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'2'})
    with (ROOT/'PREDICTOR_LOG.txt').open('x') as log:
        for line in p.stdout:log.write(line);log.flush();print(line.rstrip(),flush=True)
    code=p.wait();assert code==0,'Sealed predictor failed before any native steered target'
    pf=read(ROOT/'PREDICTION_FREEZE.json');assert pf['all_predictions_precede_any_native_steered_target'] and not list((ROOT/'native_targets').iterdir())
    pred_freeze_sha=sha(ROOT/'PREDICTION_FREEZE.json');report('predictions_sealed',predicted_trajectories=24)
    for q in questions:
        for t in [1,2]:
            pred=load(ROOT/'predictions'/f'{q["id"]}_normal_t{t}.npz');actual=load(ROOT/'native_baselines'/f'{q["id"]}_t{t}.npz')
            baseline_rows.extend(evaluate({'question':q['id'],'token':t,'split':q['split']},pred,actual))
    dump(ROOT/'BASELINE_VALUE_CHECKS.json',baseline_rows)
    assert all(r['value_pass'] for r in baseline_rows),'Equation baseline failed before any perturbed native target'
    for q in [q for q in questions if q['split']=='calibration']:
        ctx=contexts[q['id']];f1,cache1,_,u1=teacher.step(ctx['cache'],ctx['token1'],ctx['length']+1,directions[0],0.)
        f2,cache2,_,u2=teacher.step(cache1,ctx['token2'],ctx['length']+2)
        exact=all(arrsha(point[n])==arrsha(load(ROOT/'native_baselines'/f'{q["id"]}_t{t}.npz')[n]) for t,point in [(1,f1),(2,f2)] for n in FIELDS)
        assert exact and [u1,u2]==ctx['upstream']
        nulls.append({'question':q['id'],'both_positions_bitwise_exact':exact,'prefix_cache_immutable':cache_hash(ctx['cache'])==ctx['cache_hash']})
        del f1,f2,cache1,cache2
    assert teacher.calls==22;dump(ROOT/'NULL_CHECKS.json',nulls)
    def target(c):
        deadline();pid=c['question'];ctx=contexts[pid];d=directions[c['direction']];alpha=c['alpha']
        f1,cache1,candidate,u1=teacher.step(ctx['cache'],ctx['token1'],ctx['length']+1,d,alpha,True)
        f2,cache2,_,u2=teacher.step(cache1,ctx['token2'],ctx['length']+2)
        assert [u1,u2]==ctx['upstream'] and cache_hash(ctx['cache'])==ctx['cache_hash']
        normal2=load(ROOT/'native_baselines'/f'{pid}_t2.npz');assert arrsha(f2['b12_input'])==arrsha(normal2['b12_input'])
        branch.append({**c,'native_next_candidate':candidate,'controlled_normal_token2':ctx['token2'],'next_greedy_choice_changed':candidate!=ctx['token2']})
        first_aug=None
        for t,point,old in [(1,f1,{i:initials[pid][i]['S'] for i in [12,13]}),(2,f2,{i:f1[f'b{i}_S'] for i in [12,13]})]:
            meta={**c,'token':t};actual=add_drivers_and_energy(meta,point,old)
            save(ROOT/'native_targets'/f'case_{c["index"]:03d}_t{t}.npz',actual)
            pred=load(ROOT/'predictions'/f'case_{c["index"]:03d}_t{t}.npz');pred0=load(ROOT/'predictions'/f'{pid}_normal_t{t}.npz');native0=load(ROOT/'native_baselines'/f'{pid}_t{t}.npz')
            comparison.extend(evaluate(meta,pred,actual,pred0,native0))
            for mode in ['frozenK','frozenKF','reset']:
                for field in ['b13_S','b13_output']:
                    true=actual[field].astype(np.float64)-native0[field].astype(np.float64);forecast=pred[mode+':'+field]-pred0[field]
                    floor=20*np.finfo(np.float32).eps*max(1.,norm(native0[field]),norm(actual[field]));den=max(norm(true),floor)
                    controls.append({**meta,'mode':mode,'field':field,'relative_change_error':norm(forecast-true)/den})
            if t==1:first_aug=actual
            else:
                for i in [12,13]:
                    d1={n:first_aug[f'd{i}:{n}'].astype(np.float64) for n in ['k','v','beta','gamma']};d2={n:actual[f'd{i}:{n}'].astype(np.float64) for n in d1}
                    s1=first_aug[f'b{i}_S'].astype(np.float64);s2=actual[f'b{i}_S'].astype(np.float64)
                    landscape=potential(s1,d2)-potential(s1,d1);relax=potential(s2,d2)-potential(s1,d2);total=potential(s2,d2)-potential(s1,d1)
                    assert np.max(abs(total-landscape-relax))<1e-10
                    moving.append({**c,'layer':i,'changed_conditions':float(landscape.sum()),'current_step_relaxation':float(relax.sum()),'total_potential_change':float(total.sum()),'zero_at_S0_gauge':True})
            STATUS['actual_target_rows']+=1
            if c['split']=='held':STATUS['held_target_rows']+=1
        dump(ROOT/'FIELD_COMPARISONS.json',comparison);dump(ROOT/'CONTROL_COMPARISONS.json',controls);report('native_comparison',case=c['index'],question=pid)
    for c in [x for x in cases if x['split']=='calibration']:target(c)
    cal_pass=all(r['value_pass'] and r['change_pass'] for r in comparison) and all(r['pass'] for r in energies)
    dump(ROOT/'CALIBRATION_GATE.json',{'pass':cal_pass,'held_targets_seen':STATUS['held_target_rows'],'native_calls':teacher.calls,'failures':[r for r in comparison if not r['value_pass'] or not r['change_pass']]})
    assert cal_pass and STATUS['held_target_rows']==0 and teacher.calls==38,'Calibration failed; no held steered targets executed'
    report('calibration_passed')
    for c in [x for x in cases if x['split']=='held']:target(c)
    assert teacher.calls==70 and teacher.driver_replays==120 and STATUS['actual_target_rows']==48 and STATUS['held_target_rows']==32
    for n,h in pf['sha256'].items():assert sha(ROOT/n)==h,n
    for n,h in read(ROOT/'NORMAL_INPUT_FREEZE.json')['sha256'].items():assert sha(ROOT/n)==h,n
    for n,h in baseline_hash.items():assert sha(ROOT/n)==h,n
    assert sha(ROOT/'PREDICTION_FREEZE.json')==pred_freeze_sha
    for n,h in freeze['sha256'].items():assert sha(ROOT/n)==h,n
    current={f'{i}:{n}':arrsha(a) for i,p in teacher.parameters().items() for n,a in p.items()}
    assert current=={n:r['sha256'] for n,r in read(ROOT/'PARAMETER_RECEIPT.json')['arrays'].items()}
    held=[r for r in comparison if r['split']=='held'];resolved=[r for r in held if r['field'] in ['b13_S','b13_output']]
    mechanism=[]
    for t in [1,2]:
        closed=np.median([r['change_error_over_scale'] for r in held if r['token']==t and r['field']=='b13_S'])
        shortcut=np.median([r['relative_change_error'] for r in controls if r['split']=='held' and r['token']==t and r['field']=='b13_S' and r['mode']=='frozenKF'])
        mechanism.append({'token':t,'closed_median':float(closed),'frozen_drivers_median':float(shortcut),'pass':bool(closed<=.1*shortcut)})
    support=all(r['value_pass'] and r['change_pass'] for r in held) and all(r['change_resolved'] for r in resolved) and all(r['pass'] for r in mechanism) and all(r['pass'] for r in energies)
    dump(ROOT/'RESULTS.json',{'primary_support':support,'held_value_and_change_gates_pass':all(r['value_pass'] and r['change_pass'] for r in held),'required_held_signals_resolved':all(r['change_resolved'] for r in resolved),'mechanism_controls':mechanism,'native_energy_head_checks_pass':all(r['pass'] for r in energies),'held_questions':4,'calibration_questions':2,'trajectories':24,'positions':2,'full_field_comparisons':len(comparison),'held_full_field_comparisons':len(held),'resolved_target_fields':len(resolved),'native_backbone_calls':teacher.calls,'native_local_driver_replays':teacher.driver_replays,'GPU_allocations':0,'native_next_choice_changes':sum(r['next_greedy_choice_changed'] for r in branch),'prediction_input_contract':'Only normal initial memories and normal upstream boundary signals, public parameters and the known nudge; second block and token2 memories calculated internally.','scope':'Two nonlinear decoder blocks under fixed two-token normal boundary inputs, not autonomous whole-model generation, compression, semantic direction interpretation or a global conserved physical energy.','failed_fields':[r for r in held if not r['value_pass'] or not r['change_pass'] or (r['field'] in ['b13_S','b13_output'] and not r['change_resolved'])]})
    STATUS['status']='complete';report('native_complete',primary_support=support)
try:main()
except Exception:
    STATUS['status']='failed';dump(ROOT/'FAILURE.json',{'traceback':traceback.format_exc(),'charged_seconds':PRIOR+time.monotonic()-BEGIN});traceback.print_exc()
finally:
    if teacher is not None:STATUS.update(backbone_forwards=teacher.calls,local_driver_replays=teacher.driver_replays)
    dump(ROOT/'FIELD_COMPARISONS.json',comparison);dump(ROOT/'BASELINE_VALUE_CHECKS.json',baseline_rows);dump(ROOT/'CONTROL_COMPARISONS.json',controls);dump(ROOT/'NULL_CHECKS.json',nulls);dump(ROOT/'NATIVE_ENERGY_HEADS.json',energies);dump(ROOT/'MOVING_CONDITIONS.json',moving);dump(ROOT/'TOKEN_BRANCH_DIAGNOSTICS.json',branch)
    teacher=None;gc.collect();report('archive_and_cleanup')
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    dump(ROOT/'EXPORT_HASHES.json',{p.relative_to(ROOT).as_posix():sha(p) for p in files})
    archive=ROOT.parent/'R4_COUPLED_MEMORY_NATIVE_20261015.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=2) as z:
        for p in files+[ROOT/'EXPORT_HASHES.json']:z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    receipt={'status':STATUS['status'],'PID':os.getpid(),'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'backbone_calls':STATUS['backbone_forwards'],'local_driver_replays':STATUS['local_driver_replays'],'GPU_allocations':0,'seconds_including_setup_and_archive':PRIOR+time.monotonic()-BEGIN,'model_released':True}
    dump(ROOT.parent/'NATIVE_EXPORT_RECEIPT_20261015.json',receipt);print('R4_COUPLED_NATIVE_EXPORT',json.dumps(receipt),flush=True)
if STATUS['status']!='complete':raise SystemExit(1)
