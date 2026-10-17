"""Resume frozen coefficients after a scalar-JSON bug, never refit or restart."""
import os,sys,time,json,hashlib,shutil,zipfile,subprocess
from pathlib import Path
BASE=Path('/content/research4/artifacts/cached_state_response_v1')
OLD=BASE/'colab_cpu_20261014';ROOT=BASE/'colab_cpu_20261014_resume1'
PY=Path('/content/r4_cached_cpu_env/bin/python')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

if '--worker' in sys.argv:
    NEW_START=time.monotonic()
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='4')
    import ast,gc,textwrap,threading,traceback,types
    import numpy as np
    import torch
    import transformers
    assert not ROOT.exists(),'Do not restart an existing resume attempt'
    prior=json.loads((BASE/'NATIVE_EXPORT_RECEIPT_20261014_COLAB_CPU.json').read_text())
    old_status=json.loads((OLD/'STATUS.json').read_text())
    assert prior['status']=='fault_or_calibration_stop' and prior['model_calls']==90 and old_status['main_endpoints']==4
    assert 'bool is not JSON serializable' in old_status['error']
    assert sha(Path(prior['path']))==prior['sha256']
    origin_manifest=json.loads((OLD/'EXPORT_HASHES.json').read_text())
    assert all(sha(OLD/n)==h for n,h in origin_manifest.items())
    PRIOR_SECONDS=prior['native_export_seconds'];PRIOR_CALLS=prior['model_calls'];START=NEW_START-PRIOR_SECONDS
    shutil.copytree(OLD,ROOT)
    origin_map={n:n for n in origin_manifest}
    def preserve(relative):
        p=ROOT/relative
        if p.exists():
            name=p.with_name('ORIGIN_'+p.name);p.rename(name)
            if relative in origin_map:origin_map[relative]=name.relative_to(ROOT).as_posix()
    for n in ['STATUS.json','FAULT.txt','EXPORT_HASHES.json','EXECUTION_FREEZE.json','PLATFORM_AMENDMENT.json','source/maths.py','source/analyze.py']:preserve(n)
    maths=(ROOT/'source/ORIGIN_maths.py').read_text();assert 'resolved=tn>floor' in maths
    (ROOT/'source/maths.py').write_text(maths.replace('resolved=tn>floor','resolved=bool(tn>floor)'))
    analysis=(ROOT/'source/ORIGIN_analyze.py').read_text()
    analysis=analysis.replace("else 'colab_cpu_20261014'","else 'colab_cpu_20261014_resume1'")
    analysis=analysis.replace("'colab_cpu_20261014']","'colab_cpu_20261014','colab_cpu_20261014_resume1']")
    analysis=analysis.replace("suffix='_COLAB_CPU' if", "suffix='_COLAB_CPU_RESUME1' if ROOT.name.endswith('_resume1') else '_COLAB_CPU' if")
    (ROOT/'source/analyze.py').write_text(analysis)
    shutil.copyfile(Path(__file__),ROOT/'source/resume_frozen.py')
    sys.path.insert(0,str(ROOT/'source'))
    from native import CachedRig,cache_hash
    from maths import GROUPS,probability,kl,wn,norm,group_norm,subtract,central,checks,evaluate
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    assert (torch.__version__,transformers.__version__,np.__version__)==('2.13.0+cpu','5.19.0','2.5.4') and not torch.cuda.is_available()
    def scalar(x):
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False,default=scalar)+'\n')
    STATUS={'status':'resuming','PID':os.getpid(),'model_calls':PRIOR_CALLS,'prior_calls':PRIOR_CALLS,'main_endpoints':4,'stress_endpoints':0,'mixed_endpoints':0,'GPU_allocations':0,'Colab':True}
    def report(phase,**extra):
        STATUS.update(phase=phase,seconds=time.monotonic()-START,**extra);dump(ROOT/'STATUS.json',STATUS);print(json.dumps(STATUS),flush=True)
    def free_memory():return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    def deadline():
        if time.monotonic()-START>2400:raise TimeoutError('Combined native2400-second ceiling')
        if 'rig' in globals() and rig.calls>=300:raise RuntimeError('Combined300-forward ceiling')
    def hardstop():
        dump(ROOT/'HARD_STOP.json',{'model_calls':rig.calls if 'rig' in globals() else PRIOR_CALLS});os._exit(124)
    timer=threading.Timer(max(1.,2700-PRIOR_SECONDS),hardstop);timer.daemon=True;timer.start()
    def load(p):
        with np.load(p,allow_pickle=False) as z:return {n:z[n].copy() for n in z.files}
    def export():
        files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name!='EXPORT_HASHES.json')
        manifest={p.relative_to(ROOT).as_posix():sha(p) for p in files};dump(ROOT/'EXPORT_HASHES.json',manifest)
        archive=BASE/'R4_CACHED_STATE_NATIVE_20261014_COLAB_CPU_RESUME1.zip'
        with zipfile.ZipFile(archive,'x') as z:
            for p in files+[ROOT/'EXPORT_HASHES.json']:z.write(p,p.relative_to(ROOT).as_posix(),compress_type=zipfile.ZIP_STORED if p.suffix=='.npz' else zipfile.ZIP_DEFLATED,compresslevel=2)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None;assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
            assert all(hashlib.sha256(z.read(origin_map[n])).hexdigest()==h for n,h in origin_manifest.items())
        receipt={'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'members':len(files)+1,
            'model_calls':STATUS['model_calls'],'prior_calls':PRIOR_CALLS,'new_calls':STATUS['model_calls']-PRIOR_CALLS,
            'status':STATUS['status'],'native_export_seconds':time.monotonic()-START,'all_members_and_CRC_verified':True,
            'all_original_failure_manifest_contents_retained_and_verified':True,'GPU_allocations':0,'Colab':True}
        dump(BASE/'NATIVE_EXPORT_RECEIPT_20261014_COLAB_CPU_RESUME1.json',receipt);print('R4_RESUME_EXPORT',json.dumps(receipt),flush=True)
    try:
        admission={'free_bytes':free_memory(),'minimum_bytes':8589934592,'prior_calls':PRIOR_CALLS,'prior_seconds':PRIOR_SECONDS}
        dump(ROOT/'RESUME_ADMISSION.json',admission);assert admission['free_bytes']>=admission['minimum_bytes']
        PLAN=json.loads((ROOT/'source/PLAN.json').read_text());INPUTS=json.loads((ROOT/'INPUTS.json').read_text());QUESTIONS=INPUTS['questions']
        SELECTION=json.loads((ROOT/'DERIVATIVE_SELECTION_LOCK.json').read_text());coarse,fine=SELECTION['coarse'],SELECTION['fine']
        SCHEMA=json.loads((ROOT/'STATE_SCHEMA.json').read_text());V=load(ROOT/'DIRECTIONS.npz')['vectors']
        forward_available=SELECTION['forward_available'];exact_available=SELECTION['exact_available'];exact_needed=any(m=='exact' for m in SELECTION['modes'].values());attempts=[]
        BASES={};JETS={};FORECASTS={}
        for question in QUESTIONS:
            pid=question['id'];b=load(ROOT/'contexts'/f'{pid}_baseline.npz');BASES[pid]={'logits':b.pop('logits'),'values':b}
            for di in range(2):
                c=load(ROOT/'coefficients'/f'{pid}_d{di}.npz');meta=json.loads((ROOT/'coefficients'/f'{pid}_d{di}.json').read_text())
                JETS[pid,di]={'q':c['q'],'r':c['r'],'first':{n:c[n+'_first'] for n in SCHEMA},'second':{n:c[n+'_second'] for n in SCHEMA},**meta}
                FORECASTS[pid,di]=json.loads((ROOT/'predictions'/f'{pid}_d{di}.json').read_text())
        frozen=json.loads((ROOT/'PREDICTION_FREEZE.json').read_text());pred_hashes=frozen['sha256'];selection_sha=sha(ROOT/'DERIVATIVE_SELECTION_LOCK.json');prediction_sha=sha(ROOT/'PREDICTION_FREEZE.json')
        assert all(sha(ROOT/n)==h for n,h in pred_hashes.items()) and selection_sha==frozen['selection_sha256']
        # Reuse the identical scoring/probe functions, without executing the old worker's top-level run.
        original_source=(ROOT/'source/worker.py').read_text();tree=ast.parse(original_source)
        wanted={'save_value','save_probe','primal_ok','sample','jet_from','save_jet','forecasts','score'}
        defs=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in wanted];assert {n.name for n in defs}==wanted
        exec(compile(ast.fix_missing_locations(ast.Module(body=defs,type_ignores=[])),'frozen_worker_helpers','exec'),globals())
        rig=types.SimpleNamespace(schema=SCHEMA,calls=PRIOR_CALLS,model=None)
        existing=[]
        for q in QUESTIONS:
            for di in range(2):
                for ai,a in enumerate([s*t for s in PLAN['primary_strengths'] for t in PLAN['signs']]):
                    path=ROOT/'actual'/f'{q["id"]}_d{di}_a{a}.npz'
                    if path.exists():
                        values=load(path);actual={'logits':values.pop('logits'),'values':values,'requested_vector':V[di],'requested_alpha':a}
                        existing.append(score(q,di,a,actual,JETS[q['id'],di],FORECASTS[q['id'],di][ai]))
        assert len(existing)==4 and all(r['split']=='calibration' for r in existing)
        dump(ROOT/'PREEXISTING_METRICS_RECONSTRUCTED.json',existing)
        dump(ROOT/'ORIGIN_FILE_MAP.json',origin_map)
        dump(ROOT/'PLATFORM_AMENDMENT.json',{'repair':'NumPy boolean to plain bool; JSON scalar encoding only. Frozen forecasts/steps/directions unchanged.',
            'prior_calls':PRIOR_CALLS,'prior_seconds':PRIOR_SECONDS,'reused_saved_actual_endpoints':4,'fresh_held_steered_endpoints_before_resume':0,
            'no_new_fitting_or_numerical_selection':True,'original_failed_archive':prior})
        files=[p for p in ROOT.rglob('*') if p.is_file() and p.name not in ['STATUS.json','EXECUTION_FREEZE.json']]
        dump(ROOT/'EXECUTION_FREEZE.json',{'model_calls_before_resume':PRIOR_CALLS,'new_calls_before_freeze':0,'sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files}})
        report('resume_frozen_coefficients_loading_model',reconstructed_existing=4)
        rig=CachedRig(Path(PLAN['checkpoint']),deadline);rig.calls=PRIOR_CALLS;rig.precision(torch.float32)
        rig.schema=SCHEMA;rig.names=[f'b{i}_{s}' for i in range(24) for s in ['input','mixer','middle','mlp','output']]+['hidden']+sorted(n for n in SCHEMA if n.startswith('m'))
        for q in QUESTIONS:
            pid=q['id'];meta=json.loads((ROOT/'contexts'/f'{pid}.json').read_text())
            cache=torch.load(ROOT/'contexts'/f'{pid}_incoming_cache.pt',map_location='cpu',weights_only=False)
            assert cache_hash(cache)==meta['incoming_cache_hash']
            rig.contexts[pid]={'question':q,'cache':cache,'hash':meta['incoming_cache_hash'],'token':meta['token_id'],'fallback':meta['EOS_fallback'],
                'length':q['prefix_length']+1,'base_invariants':meta['outgoing_constant_invariants']}
        metrics=[];nulls=[];alphas=[s*t for s in PLAN['primary_strengths'] for t in PLAN['signs']]
        for q in QUESTIONS:
            pid=q['id'];rig.activate(pid);base=BASES[pid]
            null=rig.value(torch.tensor(V[0]),0.);assert np.array_equal(null['logits'],base['logits']) and all(np.array_equal(null['values'][n],base['values'][n]) for n in rig.names)
            nulls.append({'question':pid,'exact':True})
            for di,v in enumerate(V):
                for ai,a in enumerate(alphas):
                    path=ROOT/'actual'/f'{pid}_d{di}_a{a}.npz'
                    if path.exists():
                        data=load(path);actual={'logits':data.pop('logits'),'values':data,'requested_vector':v,'requested_alpha':a}
                    else:actual=rig.value(torch.tensor(v),a);save_value(path,actual)
                    metrics.append(score(q,di,a,actual,JETS[pid,di],FORECASTS[pid,di][ai]));STATUS['main_endpoints']=len(metrics)
                dump(ROOT/'METRICS.json',metrics);report('main_endpoints',question=pid,direction=di,model_calls=rig.calls)
        dump(ROOT/'NULL_CHECKS.json',nulls);assert len(metrics)==48
        assert all(sha(ROOT/n)==h for n,h in pred_hashes.items()) and sha(ROOT/'PREDICTION_FREEZE.json')==prediction_sha
        # Execute the unchanged, prospectively chosen stress and mixed-plane subphases.
        optional=original_source.split('    # Compact, prospectively selected stress localization while the model is loaded.\n',1)[1].split("    dump(ROOT/'BASELINE_IMMUTABILITY.json'",1)[0]
        exec(compile(textwrap.dedent(optional),'frozen_optional_subphases','exec'),globals())
        assert all(sha(OLD/n)==h for n,h in origin_manifest.items()),'Original failed data mutated'
        assert all(sha(ROOT/origin_map[n])==h for n,h in origin_manifest.items()),'Original failed contents not retained'
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in rig.model.modules())
        STATUS['status']='native_complete';report('native_complete_export_pending',model_calls=rig.calls,methods=SELECTION['modes'])
    except BaseException as e:
        STATUS['model_calls']=rig.calls if 'rig' in globals() else PRIOR_CALLS;STATUS['status']='fault_or_calibration_stop';report('fault',error=type(e).__name__+': '+str(e))
        (ROOT/'FAULT.txt').write_text(traceback.format_exc());raise
    finally:
        if 'rig' in globals():del rig;gc.collect()
        try:export()
        finally:timer.cancel()
else:
    assert not ROOT.exists(),'Inspect existing resume; never relaunch blindly'
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'4'}
    with (BASE/'COLAB_NATIVE_RESUME1_LOG_20261014.txt').open('x') as log:
        proc=subprocess.Popen([str(PY),'-B',str(Path(__file__)),'--worker'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
        print('R4_RESUME_PID',proc.pid,flush=True)
        for line in proc.stdout:log.write(line);log.flush();print(line.rstrip(),flush=True)
        code=proc.wait()
    print('R4_RESUME_TERMINAL',code,flush=True)
    assert code==0,'Resume stopped; preserved export and cumulative ledger must be inspected before any further work'
    subprocess.run([str(PY),'-B',str(ROOT/'source/analyze.py'),'colab_cpu_20261014_resume1'],check=True,env=env)
    out=BASE/'evaluated_20261014';files={p.name:p for p in out.iterdir() if p.is_file()}
    for name in ['NATIVE_EXPORT_RECEIPT_20261014_COLAB_CPU.json','NATIVE_EXPORT_RECEIPT_20261014_COLAB_CPU_RESUME1.json','COLAB_SETUP_RECEIPT_20261014.json','COLAB_NATIVE_LOG_20261014.txt','COLAB_NATIVE_RESUME1_LOG_20261014.txt']:files['receipts/'+name]=BASE/name
    archive=BASE/'R4_CACHED_STATE_COLAB_CPU_RESUME1_FINDINGS_20261014.zip';manifest={n:sha(p) for n,p in files.items()}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as z:
        for n,p in files.items():z.write(p,n)
        z.writestr('CONTENTS_SHA256.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None;assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
    print('R4_CACHED_COLAB_COMPLETE',json.dumps({'findings_archive':str(archive),'sha256':sha(archive),'results':json.loads((out/'RESULTS.json').read_text()),'model_worker_exited':True}),flush=True)
