"""One admitted CPU study: calibration-locked equations, fresh confirmation."""
from pathlib import Path
from datetime import datetime,timezone
import copy,ctypes,hashlib,json,math,os,sys,threading,time,traceback,zipfile
ROOT=Path(sys.argv[1]);SOURCE=ROOT/'source';sys.path.insert(0,str(SOURCE))
PLAN=json.loads((SOURCE/'PLAN.json').read_text());INPUTS=json.loads((ROOT/'INPUTS.json').read_text())
START=time.time();PHASE_START=START;PHASE_LIMIT=300
STATUS={'status':'starting','stage':'admission','GPU_allocations':0,'whole_backbone_calls':0,'signed_MLP_points':0,'output_cases':0,'gates':{}}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def report(stage,**extra):
    STATUS.update(stage=stage,elapsed_seconds=time.time()-START,**extra);dump(ROOT/'STATUS.json',STATUS)
    print(json.dumps(STATUS),flush=True)
def deadline():
    if (ROOT/'HUMAN_STOP.json').exists():raise InterruptedError('Human stop requested')
    if time.time()-START>=2400:raise TimeoutError('Preserve original export reserve')
    if time.time()-PHASE_START>=PHASE_LIMIT:raise TimeoutError('Declared phase work bound')
def begin(stage):
    global PHASE_START,PHASE_LIMIT
    PHASE_START=time.time();PHASE_LIMIT=PLAN['resources']['stage_limits_seconds'][stage];report(stage);deadline()
def free_memory():
    class State(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ['total','available','page_total','page_available','virtual_total','virtual_available','extended']]
    s=State();s.length=ctypes.sizeof(s);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s));return s.available
def hard_stop():
    dump(ROOT/'HARD_TIMEOUT.json',{'elapsed_seconds':time.time()-START,'GPU_allocations':0});os._exit(124)
TIMER=threading.Timer(2700,hard_stop);TIMER.daemon=True;TIMER.start()
dump(ROOT/'LIFECYCLE.json',{'started_epoch':START,'started_UTC':datetime.now(timezone.utc).isoformat(),'hard_seconds':2700,'native_stop_seconds':2400,'GPU_allocations':0})
def export():
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['EXPORT_HASHES.json','EXPORT_RECEIPT.json']]
    manifest={p.relative_to(ROOT).as_posix():sha(p) for p in files};dump(ROOT/'EXPORT_HASHES.json',manifest)
    archive=ROOT.with_name(ROOT.name+'_export.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in files:z.write(p,p.relative_to(ROOT).as_posix())
        z.write(ROOT/'EXPORT_HASHES.json','EXPORT_HASHES.json')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for member,digest in manifest.items():assert hashlib.sha256(z.read(member)).hexdigest()==digest
    receipt={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'manifest_files':len(manifest),
        'elapsed_seconds_including_load_export':time.time()-START,'GPU_allocations':0,'all_members_verified':True}
    dump(ROOT/'EXPORT_RECEIPT.json',receipt);print('R4_NORM_RESPONSE_V2_EXPORT',json.dumps(receipt),flush=True)

try:
    for name,digest in json.loads((ROOT/'EXECUTION_FREEZE.json').read_text())['sources'].items():assert sha(ROOT/name)==digest,name
    admission=json.loads((ROOT/'ADMISSION.json').read_text());assert admission['available_memory_bytes']>=8*1024**3 and admission['owned_model_workers']==0
    import numpy as np
    import torch,transformers
    from maths import LocalMLP,array,norm,relative,cosine,kl_from_logits,fisher_direction,weighted_derivative_error
    from native_v2 import Rig,cache_hash
    from equations import NAMES,candidates,gate2,quadratic_even,q_value,select_candidate,factorial_diagnostics
    assert torch.__version__=='2.13.0+cpu' and transformers.__version__=='5.19.0' and not torch.cuda.is_available()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    checkpoint=Path('C:/Users/farha/Research4/cache/qwen35_08b_pinned');pin=json.loads((SOURCE/'model_pin.json').read_text());verified=[]
    for item in pin['files']:
        p=checkpoint/item['name']
        if not p.is_file():continue
        assert p.stat().st_size==item['size']
        if item.get('sha256'):assert sha(p)==item['sha256']
        else:
            b=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==item['git_blob_id']
        verified.append(item['name'])
    assert len(verified)==8 and pin['revision']==PLAN['model']['revision']
    dump(ROOT/'IDENTITY.json',{'revision':pin['revision'],'verified_files':verified,'path':str(checkpoint)})
    begin('context_capture');rig=Rig(checkpoint);rig.check=deadline
    with np.load(ROOT/'DIRECTIONS.npz',allow_pickle=False) as z:directions=torch.tensor(z['directions_FP32'])
    questions=INPUTS['questions'];contexts={};prefix_caches={};models={}
    context_folder=ROOT/'contexts';context_folder.mkdir()
    for precision,dtype in [('BF16',torch.bfloat16),('FP32',torch.float32)]:
        rig.precision(dtype);layer=rig.layers[11]
        models[precision]=LocalMLP(copy.deepcopy(layer.post_attention_layernorm),copy.deepcopy(layer.mlp))
        for question in questions:
            pid=question['id'];ids=question['prefix_ids'];assert rig.prefix(PLAN['system'],question['text'])==ids
            folder=context_folder/pid/precision;folder.mkdir(parents=True)
            point=[]
            def capture(module,args):point.append(args[0][0,-1].detach().clone())
            handle=rig.layers[11].post_attention_layernorm.register_forward_pre_hook(capture)
            try:
                prefill=rig.forward(ids,len(ids),capture=False);prefix=copy.deepcopy(prefill['cache'])
                decoded=rig.forward([760],len(ids)+1,prefix,capture=False)
            finally:handle.remove()
            assert len(point)==2
            for position,answer,h in [('prefill',prefill,point[0]),('decode1',decoded,point[1])]:
                ctx={'h':h,'hidden':answer['hidden'].detach().clone(),'logits':answer['logits'].detach().clone()};contexts[pid,precision,position]=ctx
                np.savez_compressed(folder/(position+'.npz'),h=array(h),final_hidden=array(ctx['hidden']),logits=array(ctx['logits']))
            if precision=='FP32':prefix_caches[pid]=prefix;torch.save(prefix,folder/'prefix_cache.pt')
            report('context_capture',current_question=pid,precision=precision,whole_backbone_calls=rig.calls)
    dump(ROOT/'RUNTIME.json',{'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,'device':'CPU','threads':4,'GPU_allocations':0,
        'precision_controls':['native BF16','exact stored weights promoted to FP32','FP64 formula control'],'attention':'SDPA MATH','block_zero_based':11})

    begin('internal_equations');local=models['FP32'];numerics=[];q_checks=[]
    pred_folder=ROOT/'predictions';pred_folder.mkdir();coeff_folder=ROOT/'coefficients';coeff_folder.mkdir();prediction_hashes={}
    alpha_list=[sign*eps for eps in PLAN['strengths'] for sign in [1,-1]]
    for question in questions:
        pid=question['id']
        for position in ['prefill','decode1']:
            h=contexts[pid,'BF16',position]['h'].float()
            for di in range(4):
                v=directions[di]
                with torch.no_grad():base,j,hh=local.derivatives(h,v)
                coef=coeff_folder/f'{pid}_{position}_d{di}.npz';np.savez_compressed(coef,h=array(h),v=array(v),Jv=array(j),Hvv=array(hh),norm_epsilon=np.array(local.norm.eps))
                if question['split']=='calibration' and di<2:
                    with torch.enable_grad():_,auto=torch.autograd.functional.jvp(local.branch,h,v)
                    with torch.no_grad():
                        b64,j64,h64=local.derivatives(h.double(),v.double(),torch.float64);eps=1e-3
                        bp=local.formula(h.double()+eps*v.double());bm=local.formula(h.double()-eps*v.double())
                        fd=(bp-bm)/(2*eps);fd2=(bp+bm-2*b64)/(eps*eps)
                        gp=gate2(local,h.double(),eps*v.double(),torch.float64);gm=gate2(local,h.double(),-eps*v.double(),torch.float64)
                        s0=h.double().square().mean()+local.norm.eps;s1=2*(h.double()*v.double()).mean()
                        def rational64(alpha):
                            den=(h.double()+alpha*v.double()).square().mean()+local.norm.eps
                            return s0/den*(alpha*j64+alpha*alpha*(.5*h64+(s1/s0)*j64))
                        rp=rational64(eps);rm=rational64(-eps)
                        module=local.branch(h)
                    row={'question':pid,'position':position,'direction':di,'analytic_autograd_relative':relative(array(j-auto),array(j)),
                        'FP64_first_relative':float((fd-j64).norm()/j64.norm()),'FP64_second_relative':float((fd2-h64).norm()/h64.norm()),
                        'formula_module_relative':relative(array(base-module),array(module)),
                        'gate2_expansion_first_relative':float(((gp-gm)/(2*eps)-j64).norm()/j64.norm()),
                        'gate2_expansion_second_relative':float(((gp+gm)/(eps*eps)-h64).norm()/h64.norm()),
                        'rational_expansion_first_relative':float(((rp-rm)/(2*eps)-j64).norm()/j64.norm()),
                        'rational_expansion_second_relative':float(((rp+rm)/(eps*eps)-h64).norm()/h64.norm())}
                    numerics.append(row)
                    if pid==questions[0]['id'] and position=='prefill':
                        with torch.no_grad():
                            q0,q1,q2=quadratic_even(local,h,v);s0=h.double().square().mean()+local.norm.eps;s1=2*(h.double()*v.double()).mean()
                            for alpha in [.5,-.5,2.,-2.]:
                                den=(h.double()+alpha*v.double()).square().mean()+local.norm.eps
                                predicted=s0/den*(alpha*q1+alpha*alpha*(.5*q2+(s1/s0)*q1))
                                actual=q_value(local,h.double()+alpha*v.double())-q0
                                error=float((actual-predicted).norm());assert error<1e-10
                                q_checks.append({'direction':di,'alpha':alpha,'absolute_identity_error':error})
                for precision,dtype in [('FP32',torch.float32),('BF16',torch.bfloat16)]:
                    hb=h.to(dtype);vd=v.to(dtype);nominal_b=[];nominal_f=[];actual_b=[];actual_f=[];deltas=[];changed_inputs=[];denominators=[]
                    diag={}
                    for alpha in alpha_list:
                        deadline()
                        with torch.no_grad():
                            changed=hb+alpha*vd;delta=changed.double()-hb.double()
                            nb,nf,meta=candidates(local,h,v,alpha);ab,af,_=candidates(local,h,v,alpha,actual_delta=delta)
                            if question['split']=='calibration' and di<2 and abs(alpha) in [.5,1.25,2.]:
                                for key,value in factorial_diagnostics(local,h,delta).items():diag.setdefault(key,[]).append(array(value))
                        nominal_b.append(array(nb));nominal_f.append(array(nf));actual_b.append(array(ab));actual_f.append(array(af));deltas.append(array(delta));changed_inputs.append(array(changed.float()));denominators.append(meta['denominator'])
                    path=pred_folder/f'{pid}_{position}_d{di}_{precision}.npz'
                    np.savez_compressed(path,alphas=np.asarray(alpha_list),actual_delta=np.stack(deltas),changed_inputs=np.stack(changed_inputs),
                        nominal_branch=np.stack(nominal_b),nominal_complete=np.stack(nominal_f),actual_branch=np.stack(actual_b),actual_complete=np.stack(actual_f),denominators=np.asarray(denominators),
                        **{k:np.stack(vs) for k,vs in diag.items()})
                    prediction_hashes[path.name]=sha(path)
            report('internal_equations',prediction_question=pid,prediction_position=position)
    dump(ROOT/'PREDICTION_FREEZE.json',prediction_hashes)
    numerical_pass=all(r['analytic_autograd_relative']<=1e-4 and r['FP64_first_relative']<=1e-3 and r['FP64_second_relative']<=.01 and r['formula_module_relative']<=1e-5 and
        r['gate2_expansion_first_relative']<=1e-3 and r['gate2_expansion_second_relative']<=.01 and r['rational_expansion_first_relative']<=1e-3 and r['rational_expansion_second_relative']<=.01 for r in numerics)
    dump(ROOT/'NUMERICAL_GATE.json',{'pass':numerical_pass,'derivative_and_expansion_checks':numerics,'quadratic_even_identity_checks':q_checks})
    STATUS['gates']['internal_numerical']=numerical_pass
    if not numerical_pass:raise ArithmeticError('Fixed numerical gate failed before response measurements')

    response_folder=ROOT/'responses';response_folder.mkdir();metrics=[];done=set();diag_rows=[]
    def measure(question,position,di,precision):
        pid=question['id'];key=(pid,position,di,precision)
        if key in done:return
        done.add(key);dtype=torch.float32 if precision=='FP32' else torch.bfloat16
        h=contexts[pid,'BF16',position]['h'].to(dtype);model=models[precision]
        path=pred_folder/f'{pid}_{position}_d{di}_{precision}.npz';assert sha(path)==prediction_hashes[path.name]
        with np.load(path,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
        with torch.no_grad():base_b=model.branch(h);base_f=h+base_b
        observed_b=[];observed_f=[];diag_index=0
        for i,alpha in enumerate(alpha_list):
            deadline();changed=torch.tensor(data['changed_inputs'][i],dtype=dtype)
            with torch.no_grad():b=model.branch(changed);f=changed+b
            db=array(b.double()-base_b.double());df=array(f.double()-base_f.double());observed_b.append(db);observed_f.append(df)
            for kind,actual,reference in [('branch',db,array(base_b)),('complete',df,array(base_f))]:
                errors={};actual_controls={};input_norm=norm(data['actual_delta'][i]);output_norm=norm(actual)
                for ci,name in enumerate(NAMES):
                    predicted=data['nominal_'+kind][i,ci].astype(float);controlled=data['actual_'+kind][i,ci].astype(float)
                    magnitude=norm(predicted)
                    errors[name]={'absolute':norm(actual-predicted),'relative':relative(actual-predicted,actual),
                        'predicted_magnitude':magnitude,'magnitude_relative_error':abs(magnitude-output_norm)/output_norm if output_norm>0 else None,
                        'predicted_gain':magnitude/input_norm if input_norm>0 else None,'direction_cosine':cosine(actual,predicted)}
                    actual_controls[name]={'absolute':norm(actual-controlled),'relative':relative(actual-controlled,actual)}
                metrics.append({'question':pid,'split':question['split'],'family':question['family'],'position':position,'direction':di,'direction_split':'calibration' if di<2 else 'held_out',
                    'precision':precision,'map':kind,'strength':abs(alpha),'sign':1 if alpha>0 else -1,'actual_input_norm':input_norm,
                    'input_projection':float(data['actual_delta'][i]@directions[di].numpy()),'off_direction_norm':norm(data['actual_delta'][i]-float(data['actual_delta'][i]@directions[di].numpy())*directions[di].numpy()),
                    'zero_input':input_norm==0,'zero_output':output_norm==0,'observed_magnitude':output_norm,'observed_gain':output_norm/input_norm if input_norm>0 else None,
                    'resolved':output_norm>1e-5*max(1.,norm(reference)),'errors':errors,'actual_input_errors':actual_controls})
            if question['split']=='calibration' and di<2 and abs(alpha) in [.5,1.25,2.] and precision=='FP32':
                values={k:relative(db-data[k][diag_index],db) for k in ['linear_norm_gate1','linear_norm_gate2','exact_norm_gate1','exact_norm_gate2']}
                diag_rows.append({'question':pid,'position':position,'direction':di,'alpha':alpha,'relative_errors':values,'interaction_vector_norm':norm(data['interaction_vector'][diag_index])})
                diag_index+=1
            STATUS['signed_MLP_points']+=1
        np.savez_compressed(response_folder/path.name,branch_delta=np.stack(observed_b),complete_delta=np.stack(observed_f),base_branch=array(base_b),base_complete=array(base_f))
    for question in questions[:4]:
        for position in ['prefill','decode1']:
            for di in [0,1]:
                for precision in ['FP32','BF16']:measure(question,position,di,precision)
        report('internal_equations',calibration_measured=question['id'])
    calibration=[r for r in metrics if r['split']=='calibration' and r['direction']<2 and r['precision']=='FP32']
    selection=select_candidate(calibration);dump(ROOT/'CALIBRATION_SELECTION_LOCK.json',selection)
    selection_hash=sha(ROOT/'CALIBRATION_SELECTION_LOCK.json');selected=selection['selected'];STATUS['selected_equation']=selected
    for question in questions:
        for position in ['prefill','decode1']:
            for di in range(4):
                for precision in ['FP32','BF16']:measure(question,position,di,precision)
        report('internal_equations',held_or_cross_measured=question['id'])
    assert STATUS['signed_MLP_points']==1536 and sha(ROOT/'CALIBRATION_SELECTION_LOCK.json')==selection_hash
    dump(ROOT/'INTERNAL_METRICS.json',metrics);dump(ROOT/'CALIBRATION_DECOMPOSITION.json',diag_rows)

    remaining=2400-(time.time()-START);available=free_memory()
    output_admitted=remaining>=1500 and available>=3*1024**3
    dump(ROOT/'OUTPUT_ADMISSION.json',{'admitted':output_admitted,'native_seconds_remaining':remaining,'available_memory_bytes':available,'required_remaining_seconds':1500,'required_memory_bytes':3*1024**3})
    if not output_admitted:
        STATUS['output_extension_status']='unexecuted: complete-window/resource admission not met'
    else:
        begin('output_extension');rig.precision(torch.float32);output_folder=ROOT/'output';output_folder.mkdir();output_rows=[];checks=[];methods={}
        local_output=LocalMLP(rig.layers[11].post_attention_layernorm,rig.layers[11].mlp)
        for position in ['prefill','decode1']:
            position_ok=True;cases=[]
            for question in questions[:2]:
                for di in ([0,1] if position=='prefill' else [0]):cases.append((question,di))
            for question in questions[4:]:
                for di in ([2,3] if position=='prefill' else [2]):cases.append((question,di))
            for question,di in cases:
                if question['split']=='held_out' and not position_ok:break
                pid=question['id'];h=contexts[pid,'FP32',position]['h'].float();v=directions[di]
                ids=question['prefix_ids'] if position=='prefill' else [760];length=len(question['prefix_ids'])+(position=='decode1')
                cache=None if position=='prefill' else prefix_caches[pid]
                def hidden_shift(beta,shift):
                    def hook(module,args,output):return torch.cat([output[:,:-1],output[:,-1:]+beta*shift.reshape(1,1,-1)],dim=1)
                    return rig.hidden_context(ids,length,cache,hook)
                def derivative(shift):
                    fn=lambda beta:hidden_shift(beta,shift)
                    method=methods.get(position)
                    if method in [None,'forward_AD']:
                        try:
                            with torch.no_grad(),torch.autograd.forward_ad.dual_level():
                                dual=torch.autograd.forward_ad.make_dual(torch.tensor(0.),torch.tensor(1.))
                                value=fn(dual);base,tangent=torch.autograd.forward_ad.unpack_dual(value)
                            if tangent is None:raise RuntimeError('No automatic derivative tangent')
                            methods[position]='forward_AD'
                            return base,tangent
                        except (NotImplementedError,RuntimeError) as error:
                            if method is not None or 'memory' in str(error).lower():raise
                            dump(output_folder/(position+'_AD_compatibility.json'),{'error':type(error).__name__+': '+str(error),'fallback_before_signed_measurements':True})
                    with torch.enable_grad():base,tangent=torch.autograd.functional.jvp(fn,torch.tensor(0.),torch.tensor(1.))
                    methods[position]='reverse_over_reverse_JVP';return base,tangent
                with torch.no_grad():_,j,_=local_output.derivatives(h,v);nominal_complete=v+j
                base_hidden,tangent=derivative(nominal_complete)
                primal_reference=contexts[pid,'FP32',position]['hidden']
                primal_error=float((base_hidden.detach()-primal_reference).norm()/max(1.,float(primal_reference.norm())))
                assert primal_error<=1e-5,'Output derivative base-context mismatch'
                with torch.no_grad():base_logits=rig.model.lm_head(base_hidden);q_nominal=rig.model.lm_head(tangent)
                if question['split']=='calibration':
                    with torch.no_grad():
                        hp=hidden_shift(torch.tensor(.03125),nominal_complete);hm=hidden_shift(torch.tensor(-.03125),nominal_complete)
                        fd=(rig.model.lm_head(hp)-rig.model.lm_head(hm))/(2*.03125)
                    check=weighted_derivative_error(array(base_logits),array(q_nominal),array(fd))
                    checks.append({'question':pid,'position':position,'direction':di,**check})
                    if not check['resolved'] or check['relative_error']>.05:position_ok=False
                coefficients=fisher_direction(array(base_logits),array(q_nominal))
                qpred={};storage={'base_logits':array(base_logits),'q_nominal':array(q_nominal),'base_hidden':array(base_hidden),'v':array(v),'h':array(h)}
                unique=['quadratic','rational']+([selected] if selected!='rational' else [])
                alphas=[sign*eps for eps in [.125,.5,2.] for sign in [1,-1]]
                for name in unique:
                    values=[]
                    for alpha in alphas:
                        deadline()
                        with torch.no_grad():_,full,_=candidates(local_output,h,v,alpha);shift=full[NAMES.index(name)]
                        _,dq=derivative(shift)
                        with torch.no_grad():q=rig.model.lm_head(dq)
                        values.append(array(q))
                    qpred[name]=np.stack(values);storage['q_'+name]=qpred[name]
                prediction_file=output_folder/f'{pid}_{position}_d{di}_predictions.npz'
                np.savez_compressed(prediction_file,**storage,alphas=np.asarray(alphas))
                dump(prediction_file.with_suffix('.json'),{'sha256':sha(prediction_file),'predicted_before_actual_endpoints':True,'selected_equation':selected,'AD_method':methods[position],
                    'base_context_relative_error':primal_error,'base_context_exact':bool(torch.equal(base_hidden.detach(),primal_reference))})
                actual_logits=[];actual_hidden=[]
                for ai,alpha in enumerate(alphas):
                    def actual_hook(module,args,output):
                        x=h+alpha*v
                        delta=(x+local_output.branch(x))-(h+local_output.branch(h))
                        return torch.cat([output[:,:-1],output[:,-1:]+delta.reshape(1,1,-1)],dim=1)
                    with torch.no_grad():
                        actual=rig.hidden_context(ids,length,cache,actual_hook);logits=rig.model.lm_head(actual)
                    actual_logits.append(array(logits));actual_hidden.append(array(actual))
                    measured=kl_from_logits(array(base_logits),array(logits))
                    predictions={'Fisher':.5*alpha*alpha*coefficients,'linear_endpoint':kl_from_logits(array(base_logits),array(base_logits)+alpha*array(q_nominal))}
                    secondary={}
                    for name in unique:
                        predicted_logits=array(base_logits)+qpred[name][ai]
                        predictions[name+'_endpoint']=kl_from_logits(array(base_logits),predicted_logits)
                        secondary[name+'_endpoint']=kl_from_logits(array(logits),predicted_logits)
                    predictions['selected_endpoint']=predictions[selected+'_endpoint'];secondary['selected_endpoint']=secondary[selected+'_endpoint']
                    output_rows.append({'question':pid,'split':question['split'],'family':question['family'],'position':position,'direction':di,'strength':abs(alpha),'sign':1 if alpha>0 else -1,
                        'measured_endpoint_KL':measured,'resolved':measured>1e-10,'predictions':predictions,
                        'absolute_errors':{k:abs(x-measured) for k,x in predictions.items()},
                        'relative_errors':{k:abs(x-measured)/measured if measured>1e-10 else None for k,x in predictions.items()},
                        'KL_actual_to_predicted':secondary,'selected_equation':selected})
                np.savez_compressed(output_folder/f'{pid}_{position}_d{di}_actual.npz',logits=np.stack(actual_logits),hidden=np.stack(actual_hidden),alphas=np.asarray(alphas))
                STATUS['output_cases']+=1;report('output_extension',current_question=pid,position=position,direction=di,whole_backbone_calls=rig.calls)
                if question['split']=='calibration':dump(ROOT/'OUTPUT_DERIVATIVE_GATE.json',{'checks':checks,'methods':methods,'position_pass':{p:all(c['resolved'] and c['relative_error']<=.05 for c in checks if c['position']==p) for p in methods}})
        dump(ROOT/'OUTPUT_METRICS.json',output_rows);STATUS['output_extension_status']='executed, numerical/support results retained'
    STATUS['whole_backbone_calls']=rig.calls;assert rig.calls<=496
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in rig.layers)
    STATUS['status']='complete';report('complete_export_pending')
except BaseException as error:
    STATUS['status']='fault_or_gate_stop_no_automatic_retry';report('fault',error=type(error).__name__+': '+str(error))
    (ROOT/'FAULT.txt').write_text(traceback.format_exc());raise
finally:export();TIMER.cancel()
