"""Prospective semantic intervention and full two-block vector comparison on CPU."""
import os,time
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
BEGIN=time.monotonic()
import sys,subprocess,gc,traceback,zipfile,shutil,ctypes
from pathlib import Path
import numpy as np,torch,transformers
from common import read,dump,sha,digest,norm,load,save,evaluate,word_scores,FIELDS
from native_teacher import Teacher,cache_hash
from equations import energy_rows
from output_geometry import build
ROOT=Path(sys.argv[1]);SOURCE=ROOT/'source';CHECKPOINT=Path(sys.argv[2]);PLAN=read(SOURCE/'PLAN.json');PRIOR=60.
teacher=None;comparison=[];energies=[];baseline=[];nulls=[];effects=[];projections=[];branches=[];upstream=[]
STATUS={'PID':os.getpid(),'status':'running','backbone_forwards':0,'local_driver_replays':0,'held_target_rows':0,'target_rows':0,'GPU_allocations':0,'conservative_prior_setup_seconds':PRIOR}
def available_memory():
    if sys.platform=='win32':
        class Memory(ctypes.Structure):_fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ['total','available','total_page','available_page','total_virtual','available_virtual','extended']]
        q=Memory();q.length=ctypes.sizeof(q);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(q));return int(q.available)
    return int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
def deadline():
    assert PRIOR+time.monotonic()-BEGIN<2300,'Combined execution deadline'
    assert available_memory()>1610612736,'Available RAM fell below1.5GiB; preserve and release'
def report(phase,**kw):
    if teacher is not None:STATUS.update(backbone_forwards=teacher.calls,local_driver_replays=teacher.driver_replays)
    STATUS.update(phase=phase,charged_seconds=PRIOR+time.monotonic()-BEGIN,**kw);dump(ROOT/'STATUS.json',STATUS)
    print('R4_SEMANTIC',__import__('json').dumps(STATUS),flush=True)
def augment(fields):
    for i in [12,13]:fields.update({f'd{i}:{n}':a for n,a in teacher.energy_driver(i,fields).items()})
    return fields
def add_energy(meta,point,old):
    for i in [12,13]:
        d={n:point[f'd{i}:{n}'].astype(np.float64) for n in ['k','v','beta','gamma']}
        energies.extend([{**meta,'layer':i,**r} for r in energy_rows(old[i].astype(np.float64),point[f'b{i}_S'].astype(np.float64),d)])
def main():
    global teacher
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    assert (torch.__version__,transformers.__version__,np.__version__)==('2.13.0+cpu','5.19.0','2.5.4') and not torch.cuda.is_available()
    admission={'available_RAM_after_import':available_memory(),'disk_free':shutil.disk_usage(ROOT).free,'GPU_available':False,'planned_backbone_forwards':222}
    dump(ROOT/'ADMISSION.json',admission);assert admission['available_RAM_after_import']>=8589934592 and admission['disk_free']>=17179869184
    freeze=read(ROOT/'EXECUTION_FREEZE.json')
    for n,h in freeze['sha256'].items():assert sha(ROOT/n)==h,n
    pins={r['name']:r for r in read(SOURCE/'model_pin.json')['files']};checked=[]
    for p in CHECKPOINT.iterdir():
        if not p.is_file():continue
        r=pins[p.name];assert p.stat().st_size==r['size']
        if r['sha256']:assert sha(p)==r['sha256']
        else:
            b=p.read_bytes();assert __import__('hashlib').sha1(('blob '+str(len(b))+'\0').encode()+b).hexdigest()==r['git_blob_id']
        checked.append(p.name)
    assert len(checked)==8;dump(ROOT/'CHECKPOINT_VERIFICATION.json',{'files':checked,'all_pinned_hashes_match':True,'model_revision':PLAN['model_revision']})
    for n in ['normal_inputs','training_prefixes','native_baselines','native_targets']:(ROOT/n).mkdir(exist_ok=False)
    report('loading_frozen_original_model');teacher=Teacher(CHECKPOINT,deadline)
    public_geometry=build(teacher.model.lm_head.weight,PLAN,ROOT,report,deadline)
    assert PRIOR+time.monotonic()-BEGIN<1200,'Setup time limit'
    flat={f'{i}:{n}':a for i,p in teacher.parameters().items() for n,a in p.items()};save(ROOT/'PARAMETERS.npz',flat)
    parameter_hash={n:digest(a) for n,a in flat.items()};dump(ROOT/'PARAMETER_RECEIPT.json',{'arrays':{n:{'shape':list(a.shape),'sha256':parameter_hash[n]} for n,a in flat.items()},'unchanged_public_trained_parameters':True,'original_loader_A_log_dtype':teacher.loaded_A_log_dtype});del flat;gc.collect()
    contexts={};normal_metadata=[]
    def normal(q,both):
        pid=q['id'];ids=q['prefix_ids'];cache0,_=teacher.prefix(ids[:-1]);initial=teacher.initial(cache0)
        f1,cache1,token2,u1=teacher.step(cache0,ids[-1],len(ids),choose_next=True);augment(f1);save(ROOT/'native_baselines'/f'{pid}_t1.npz',f1)
        packet={f'initial{i}:{n}':a for i,s in initial.items() for n,a in s.items()};packet['boundary_t1']=f1['b12_input']
        ctx={'cache':cache0,'hash':cache_hash(cache0),'token1':ids[-1],'token2':token2,'length':len(ids),'upstream':[u1],'initial':initial}
        if both:
            assert token2 not in {248044,248046,int(teacher.tokenizer.eos_token_id)},'Normal first answer is EOS; no substitution'
            f2,cache2,_,u2=teacher.step(cache1,token2,len(ids)+1);augment(f2);save(ROOT/'native_baselines'/f'{pid}_t2.npz',f2)
            packet['boundary_t2']=f2['b12_input'];ctx['upstream'].append(u2);save(ROOT/'normal_inputs'/f'{pid}.npz',packet)
            add_energy({'question':pid,'split':q['split'],'case':'normal','token':1},f1,{i:initial[i]['S'] for i in [12,13]})
            add_energy({'question':pid,'split':q['split'],'case':'normal','token':2},f2,{i:f1[f'b{i}_S'] for i in [12,13]})
            del f2,cache2
        else:save(ROOT/'training_prefixes'/f'{pid}.npz',packet)
        ctx['scores']=word_scores(f1['logits'],q['word_ids']);contexts[pid]=ctx
        normal_metadata.append({**q,'token1':ctx['token1'],'token2':token2,'normal_first_word':teacher.tokenizer.decode([token2]),'prefix_cache_sha256':ctx['hash'],'upstream_hashes':ctx['upstream'],'normal_word_scores':ctx['scores']})
        del f1,cache1;gc.collect();report('normal_context',question=pid)
    training=[q for q in PLAN['questions'] if q['split']!='held']
    for q in training:normal(q,q['split']=='calibration')
    assert teacher.calls==26
    differences={n:[] for n in ['b12_input','b13_output','b13_S']}
    for pair_index in range(6):
        qs=sorted([q for q in training if q['pair_index']==pair_index],key=lambda q:q['number']);assert len(qs)==2
        p0=load(ROOT/'native_baselines'/f'{qs[0]["id"]}_t1.npz');p1=load(ROOT/'native_baselines'/f'{qs[1]["id"]}_t1.npz')
        for n in differences:differences[n].append(p1[n].astype(np.float64)-p0[n].astype(np.float64))
    w=np.mean(differences['b12_input'],axis=0);assert norm(w)>1e-8;w/=norm(w)
    rng=np.random.default_rng(2026101522);r=rng.normal(size=1024);r-=w*float(w@r);r/=norm(r)
    directions=np.stack([w,r]).astype(np.float32);alpha=float(np.clip(.25*np.median([norm(a) for a in differences['b12_input']]),.125,2.))
    axes={n:np.mean(differences[n],axis=0) for n in ['b13_output','b13_S']}
    for n in axes:assert norm(axes[n])>1e-8;axes[n]/=norm(axes[n])
    save(ROOT/'DIRECTIONS.npz',{'directions':directions,'alpha':np.array(alpha),'training_input_differences':np.stack(differences['b12_input']),**axes})
    direction_receipt={'questions_seen':[q['id'] for q in training],'held_contexts_seen':0,'steered_targets_seen':0,'alpha':alpha,'unit_direction_sha256':[digest(a) for a in directions],'mean_contrast_norm':norm(np.mean(differences['b12_input'],axis=0)),'training_pair_cosines':[float(a@w/norm(a)) for a in differences['b12_input']],'no_output_metric_assumed_in_hidden_layer':True,'sha256':{'DIRECTIONS.npz':sha(ROOT/'DIRECTIONS.npz'),**{f'native_baselines/{q["id"]}_t1.npz':sha(ROOT/'native_baselines'/f'{q["id"]}_t1.npz') for q in training}}}
    dump(ROOT/'DIRECTION_FREEZE.json',direction_receipt);report('directions_frozen_before_held_contexts',alpha=alpha)
    for q in [q for q in training if q['split']=='fit']:del contexts[q['id']]
    del differences;gc.collect()
    for q in [q for q in PLAN['questions'] if q['split']=='held']:normal(q,True)
    assert teacher.calls==74
    dump(ROOT/'ALL_NORMAL_METADATA.json',normal_metadata)
    questions=[q for q in normal_metadata if q['split']!='fit'];assert len(questions)==18
    dump(ROOT/'INPUTS.json',{'questions':questions,'normal_forwards':74})
    cases=[{'index':j,'question':q['id'],'split':q['split'],'pair_index':q['pair_index'],'number':q['number'],'template':q['template'],'direction':d,'direction_name':'candidate_number' if d==0 else 'random_control','alpha':sign*alpha} for j,(q,d,sign) in enumerate((q,d,sign) for q in questions for d in [0,1] for sign in [1,-1])]
    dump(ROOT/'CASES.json',cases);assert len(cases)==72
    allowed=[ROOT/n for n in ['PARAMETERS.npz','DIRECTIONS.npz','INPUTS.json','CASES.json']]+list((ROOT/'normal_inputs').glob('*.npz'))
    dump(ROOT/'NORMAL_INPUT_FREEZE.json',{'native_steered_targets_seen':0,'sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in allowed}})
    native_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'native_baselines').glob('*.npz')};dump(ROOT/'NATIVE_BASELINE_FREEZE.json',native_hashes)
    assert not list((ROOT/'native_targets').iterdir());report('separate_numpy_predictor_start')
    with (ROOT/'PREDICTOR_LOG.txt').open('x') as log:
        predictor=subprocess.Popen([sys.executable,'-B',str(SOURCE/'predict.py'),str(ROOT)],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'2'})
        while predictor.poll() is None:
            deadline();time.sleep(.5)
        assert predictor.returncode==0,'Predictor stopped before any native intervention'
    pf=read(ROOT/'PREDICTION_FREEZE.json');pf_sha=sha(ROOT/'PREDICTION_FREEZE.json');assert pf['all_targets_seen']==0 and not list((ROOT/'native_targets').iterdir())
    for q in questions:
        for t in [1,2]:baseline.extend(evaluate({'question':q['id'],'split':q['split'],'token':t},load(ROOT/'predictions'/f'{q["id"]}_normal_t{t}.npz'),load(ROOT/'native_baselines'/f'{q["id"]}_t{t}.npz')))
    dump(ROOT/'BASELINE_VALUE_CHECKS.json',baseline);assert len(baseline)==504 and all(r['value_pass'] for r in baseline)
    for q in [q for q in questions if q['split']=='calibration']:
        ctx=contexts[q['id']];f1,c1,_,u1=teacher.step(ctx['cache'],ctx['token1'],ctx['length'],directions[1],0.)
        f2,c2,_,u2=teacher.step(c1,ctx['token2'],ctx['length']+1)
        exact=all(digest(point[n])==digest(load(ROOT/'native_baselines'/f'{q["id"]}_t{t}.npz')[n]) for t,point in [(1,f1),(2,f2)] for n in FIELDS+['logits','final_embedding'])
        assert exact and [u1,u2]==ctx['upstream'];nulls.append({'question':q['id'],'both_positions_all16_fields_bitwise_exact':True,'prefix_immutable':cache_hash(ctx['cache'])==ctx['hash']})
        del f1,f2,c1,c2
    assert teacher.calls==78;dump(ROOT/'NULL_CHECKS.json',nulls);report('sealed_forecasts_and_strict_null_passed')
    question_map={q['id']:q for q in questions}
    def target(c):
        pid=c['question'];ctx=contexts[pid];q=question_map[pid]
        f1,c1,candidate,u1=teacher.step(ctx['cache'],ctx['token1'],ctx['length'],directions[c['direction']],c['alpha'],True)
        f2,c2,_,u2=teacher.step(c1,ctx['token2'],ctx['length']+1)
        assert [u1,u2]==ctx['upstream'] and cache_hash(ctx['cache'])==ctx['hash']
        assert digest(f2['b12_input'])==digest(load(ROOT/'native_baselines'/f'{pid}_t2.npz')['b12_input'])
        upstream.append({**c,'both_upstream_cache_hashes_match':True,'prefix_immutable':True,'token2_boundary_bitwise_identical':True})
        branches.append({**c,'normal_first_token':ctx['token2'],'steered_first_token':candidate,'normal_word':teacher.tokenizer.decode([ctx['token2']]),'steered_word':teacher.tokenizer.decode([candidate]),'greedy_first_word_changed':candidate!=ctx['token2']})
        points={1:augment(f1),2:augment(f2)}
        for t,actual in points.items():
            meta={**c,'token':t};save(ROOT/'native_targets'/f'case_{c["index"]:03d}_t{t}.npz',actual)
            n0=load(ROOT/'native_baselines'/f'{pid}_t{t}.npz');p=load(ROOT/'predictions'/f'case_{c["index"]:03d}_t{t}.npz');p0=load(ROOT/'predictions'/f'{pid}_normal_t{t}.npz')
            comparison.extend(evaluate(meta,p,actual,p0,n0));add_energy(meta,actual,{i:ctx['initial'][i]['S'] if t==1 else f1[f'b{i}_S'] for i in [12,13]})
            for name in ['b13_output','b13_S']:
                projections.append({**meta,'field':name,'native_projection':float(np.sum((actual[name].astype(np.float64)-n0[name])*axes[name])),'predicted_projection':float(np.sum((p[name]-p0[name])*axes[name]))})
            if t==1:effects.append({**c,**word_scores(actual['logits'],q['word_ids'])})
            STATUS['target_rows']+=1
            if c['split']=='held':STATUS['held_target_rows']+=1
        dump(ROOT/'FIELD_COMPARISONS.json',comparison);dump(ROOT/'NATIVE_WORD_EFFECTS.json',effects);dump(ROOT/'TOKEN_BRANCH_DIAGNOSTICS.json',branches);dump(ROOT/'UPSTREAM_CONTROLS.json',upstream)
        del f1,f2,c1,c2,points;gc.collect();report('native_comparison',case=c['index'],question=pid)
    for c in [c for c in cases if c['split']=='calibration']:target(c)
    calibration=all(r['value_pass'] and r['change_pass'] for r in comparison) and all(r['pass'] for r in energies)
    dump(ROOT/'CALIBRATION_GATE.json',{'pass':calibration,'held_targets_seen':STATUS['held_target_rows'],'native_calls':teacher.calls})
    assert calibration and STATUS['held_target_rows']==0 and teacher.calls==94;report('calibration_passed')
    for c in [c for c in cases if c['split']=='held']:target(c)
    assert teacher.calls==222 and teacher.driver_replays==380 and STATUS['target_rows']==144 and STATUS['held_target_rows']==128
    for n,h in pf['sha256'].items():assert sha(ROOT/n)==h,n
    assert sha(ROOT/'PREDICTION_FREEZE.json')==pf_sha
    for n,h in read(ROOT/'NORMAL_INPUT_FREEZE.json')['sha256'].items():assert sha(ROOT/n)==h,n
    for n,h in native_hashes.items():assert sha(ROOT/n)==h,n
    for n,h in direction_receipt['sha256'].items():assert sha(ROOT/n)==h,n
    for n,h in freeze['sha256'].items():assert sha(ROOT/n)==h,n
    assert {f'{i}:{n}':digest(a) for i,p in teacher.parameters().items() for n,a in p.items()}==parameter_hash
    held=[r for r in comparison if r['split']=='held'];keys=[r for r in held if r['direction']==0 and r['field'] in ['b13_S','b13_output']]
    numerical=all(r['value_pass'] and r['change_pass'] for r in held) and all(r['change_resolved'] for r in keys) and all(r['pass'] for r in energies)
    dump(ROOT/'RESULTS.json',{'numerical_support':numerical,'semantic_support':'Computed independently from complete archived word effects; do not conflate numerical and semantic success.','held_complete_field_checks':len(held),'all_complete_field_checks':len(comparison),'baseline_field_checks':len(baseline),'required_held_semantic_key_fields':len(keys),'all_required_signals_resolved':all(r['change_resolved'] for r in keys),'native_energy_head_checks':len(energies),'native_backbone_calls':teacher.calls,'native_local_driver_replays':teacher.driver_replays,'held_contexts':16,'held_nouns':4,'held_templates':2,'candidate_direction_alpha':alpha,'GPU_allocations':0,'original_parameters_changed':False,'scope':PLAN['interpretation'],'failed_fields':[r for r in held if not r['value_pass'] or not r['change_pass'] or (r['direction']==0 and r['field'] in ['b13_S','b13_output'] and not r['change_resolved'])]})
    STATUS['status']='complete';report('native_complete',numerical_support=numerical)
try:main()
except Exception:
    STATUS['status']='failed';dump(ROOT/'FAILURE.json',{'traceback':traceback.format_exc(),'charged_seconds':PRIOR+time.monotonic()-BEGIN});traceback.print_exc()
finally:
    if teacher is not None:STATUS.update(backbone_forwards=teacher.calls,local_driver_replays=teacher.driver_replays)
    for name,value in [('FIELD_COMPARISONS',comparison),('BASELINE_VALUE_CHECKS',baseline),('NATIVE_ENERGY_HEADS',energies),('NULL_CHECKS',nulls),('NATIVE_WORD_EFFECTS',effects),('SEMANTIC_AXIS_PROJECTIONS',projections),('TOKEN_BRANCH_DIAGNOSTICS',branches),('UPSTREAM_CONTROLS',upstream)]:dump(ROOT/(name+'.json'),value)
    teacher=None;gc.collect();STATUS['model_released']=True;report('model_release_and_archive')
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.tmp')]
    dump(ROOT/'EXPORT_HASHES.json',{p.relative_to(ROOT).as_posix():sha(p) for p in files})
    archive=ROOT.parent/'R4_SEMANTIC_TRANSPORT_NATIVE_20261015.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=2) as z:
        for p in files+[ROOT/'EXPORT_HASHES.json']:z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    receipt={'status':STATUS['status'],'PID':os.getpid(),'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'native_backbone_calls':STATUS['backbone_forwards'],'GPU_allocations':0,'charged_seconds_including_prior_and_archive':PRIOR+time.monotonic()-BEGIN,'model_released':True}
    dump(ROOT.parent/'NATIVE_EXPORT_RECEIPT_20261015.json',receipt);print('R4_SEMANTIC_NATIVE_EXPORT',__import__('json').dumps(receipt),flush=True)
if STATUS['status']!='complete':raise SystemExit(1)
