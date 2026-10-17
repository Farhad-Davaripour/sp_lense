"""Bounded staged local-response and output-energy study on one pinned backbone."""
from pathlib import Path
import copy,csv,hashlib,json,os,sys,threading,time,traceback,zipfile

ROOT=Path(sys.argv[1]);SOURCE=ROOT/'source';sys.path.insert(0,str(SOURCE))
PLAN=json.loads((SOURCE/'PLAN.json').read_text());START=time.time();PHASE_START=START;PHASE_LIMIT=300
STATUS={'status':'starting','stage':'source_admission','GPU_allocations':0,'completed':{},'gates':{}}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def report(stage,**extra):
    STATUS.update(stage=stage,elapsed_seconds=time.time()-START,**extra)
    dump(ROOT/'STATUS.json',STATUS);print(json.dumps({'stage':stage,'elapsed':STATUS['elapsed_seconds'],'completed':STATUS['completed'],**extra}),flush=True)
def deadline():
    if (ROOT/'HUMAN_STOP.json').exists():raise InterruptedError('Human stop requested')
    if time.time()-START>=PLAN['resources']['native_stop_seconds']:raise TimeoutError('Preserve export reserve')
    if time.time()-PHASE_START>=PHASE_LIMIT:raise TimeoutError('Declared phase work bound reached')
def begin(phase):
    global PHASE_START,PHASE_LIMIT
    PHASE_START=time.time();PHASE_LIMIT=PLAN['resources']['stage_limits_seconds'][phase]
    report(phase);deadline()
def hard_stop():
    dump(ROOT/'HARD_TIMEOUT.json',{'status':'CPU hard cap','elapsed':time.time()-START,'GPU_allocations':0});os._exit(124)
TIMER=threading.Timer(PLAN['resources']['inclusive_worker_hard_seconds'],hard_stop);TIMER.daemon=True;TIMER.start()

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
        'elapsed_seconds':time.time()-START,'GPU_allocations':0,'all_members_verified':True}
    dump(ROOT/'EXPORT_RECEIPT.json',receipt);print('R4_ENERGY_RESPONSE_EXPORT',json.dumps(receipt),flush=True)

try:
    freeze=json.loads((ROOT/'EXECUTION_FREEZE.json').read_text())
    for member,digest in freeze['sources'].items():assert sha(ROOT/member)==digest,member
    import numpy as np
    import torch,transformers
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from maths import LocalMLP,array,norm,cosine,relative,parity,geometry,kl_from_logits,fisher_direction,weighted_derivative_error,synthetic_controls
    from native import Rig,save_context,cache_hash,attention_function
    assert torch.__version__=='2.13.0+cpu' and transformers.__version__=='5.19.0' and not torch.cuda.is_available()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    dump(ROOT/'SYNTHETIC_CONTROLS.json',synthetic_controls())
    admission=json.loads((ROOT/'ADMISSION.json').read_text());assert admission['available_memory_bytes']>=8*1024**3
    modelpath=Path('C:/Users/farha/Research4/cache/qwen35_08b_pinned')
    pin=json.loads((SOURCE/'model_pin.json').read_text());verified=[]
    for item in pin['files']:
        p=modelpath/item['name']
        if not p.is_file():continue
        assert p.stat().st_size==item['size']
        if item.get('sha256'):assert sha(p)==item['sha256'],item['name']
        else:
            data=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==item['git_blob_id']
        verified.append(item['name'])
    assert len(verified)==8 and pin['revision']==PLAN['model']['revision']
    dump(ROOT/'IDENTITY.json',{'revision':pin['revision'],'verified_files':verified,'model_path':str(modelpath)})
    begin('context_capture');rig=Rig(modelpath);rig.check=deadline
    questions=PLAN['questions'];compiled=[]
    for question in questions:
        ids=rig.prefix(PLAN['system'],question['text']);assert 1<len(ids)<=64
        compiled.append({**question,'prefix_ids':ids,'prefix_length':len(ids)})
    dump(ROOT/'COMPILED_QUESTIONS.json',compiled)
    vectorpath=ROOT/'panel_vectors.npz';assert sha(vectorpath)=='0f4311c6c8aef95c57bb3183dff119e50ddb5c0c472136988fa72fc689ca99eb'
    with np.load(vectorpath,allow_pickle=False) as z:legacy=np.asarray(z['v'],dtype=np.float32)
    raw=np.column_stack([legacy.astype(np.float64),np.random.default_rng(20261012).standard_normal((1024,3))])
    q,_=np.linalg.qr(raw)
    for i in range(4):
        if q[:,i]@raw[:,i]<0:q[:,i]*=-1
    directions=torch.tensor(q.T,dtype=torch.float32)
    np.savez_compressed(ROOT/'DIRECTIONS.npz',q64=q,directions_FP32=directions.numpy(),directions_BF16=directions.bfloat16().float().numpy(),legacy_v=legacy)
    contexts={};prefix_caches={};copies={}
    for precision,dtype in [('BF16',torch.bfloat16),('FP32',torch.float32)]:
        rig.precision(dtype)
        layer=rig.layers[11]
        copies[precision]=dict(norm=copy.deepcopy(layer.post_attention_layernorm),mlp=copy.deepcopy(layer.mlp),attention_layer=copy.deepcopy(layer))
        for question in compiled:
            deadline();pid=question['id'];ids=question['prefix_ids']
            prefill=rig.forward(ids,len(ids),capture=True)
            prefix=copy.deepcopy(prefill['cache']);prefix_caches[pid,precision]=prefix
            decoded=rig.forward([760],len(ids)+1,prefix,capture=True)
            for position,ctx in [('prefill',prefill),('decode1',decoded)]:
                contexts[pid,precision,position]=ctx
                save_context(ROOT/'contexts'/pid/precision/position,ctx,prefix_ids=ids,precision=precision,position=position)
            torch.save(prefix,ROOT/'contexts'/pid/precision/'prefix_cache.pt')
            STATUS['completed']['base_contexts']=STATUS['completed'].get('base_contexts',0)+2
            report('context_capture',current_question=pid,precision=precision)
        # Full conditional wrapper parity is a calibration-only context check.
        ids=compiled[0]['prefix_ids']
        with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
            wrapper=rig.model(input_ids=torch.tensor([ids]),attention_mask=torch.ones((1,len(ids)),dtype=torch.long),use_cache=True,logits_to_keep=1,return_dict=True)
        rig.calls+=1
        reference=contexts[compiled[0]['id'],precision,'prefill']['logits']
        error=float((wrapper.logits[0,-1].float()-reference.float()).abs().max())
        assert error<=1e-5*max(1.,float(reference.float().abs().max()))
        STATUS['gates']['wrapper_parity_'+precision]=error
    dump(ROOT/'RUNTIME.json',{'torch':torch.__version__,'transformers':transformers.__version__,'device':'CPU','threads':4,
        'precision_controls':['native BF16','promoted exact checkpoint values FP32','formula-level FP64 local control'],
        'attention_backend':'SDPA MATH','block_types':rig.text.config.layer_types,'GPU_allocations':0})

    begin('MLP_pilot')
    mlp32=LocalMLP(copies['FP32']['norm'],copies['FP32']['mlp'])
    mlp16=LocalMLP(copies['BF16']['norm'],copies['BF16']['mlp'])
    local_models={'FP32':mlp32,'BF16':mlp16};gates=[];predictions={};geometries=[]
    pred_folder=ROOT/'MLP_predictions';pred_folder.mkdir()
    for question in compiled:
        pid=question['id']
        for position in ['prefill','decode1']:
            h=contexts[pid,'BF16',position]['data']['mlp_h'].float().detach()
            jcols=[]
            for di in range(4):
                deadline();v=directions[di]
                with torch.no_grad():base,j,hessian=mlp32.derivatives(h,v)
                predictions[pid,position,di]=(h,v,j,hessian);jcols.append(array(j))
                np.savez_compressed(pred_folder/f'{pid}_{position}_d{di}.npz',h=array(h),v=array(v),branch_base=array(base),Jv=array(j),Hvv=array(hessian),complete_Jv=array(v+j))
                if question['split']=='calibration' and di in [0,1]:
                    with torch.enable_grad():_,auto=torch.autograd.functional.jvp(mlp32.branch,h,v,create_graph=False)
                    step=1e-3
                    with torch.no_grad():
                        b64,j64,h64=mlp32.derivatives(h.double(),v.double(),torch.float64)
                        plus=mlp32.formula(h.double()+step*v.double());minus=mlp32.formula(h.double()-step*v.double())
                        fd=(plus-minus)/(2*step);fd2=(plus+minus-2*b64)/(step*step)
                        direct=mlp32.branch(h)
                    row=dict(question=pid,position=position,direction=di,analytic_autograd_relative=relative(array(j-auto),array(j)),
                        FP64_first_relative=float((fd-j64).norm()/j64.norm()),FP64_second_relative=float((fd2-h64).norm()/h64.norm()),
                        formula_module_relative=relative(array(base-direct),array(direct)))
                    gates.append(row)
            jmatrix=np.column_stack(jcols)
            for kind,values in [('branch',jmatrix),('complete',jmatrix+q)]:
                geometries.append(dict(question=pid,position=position,map=kind,**geometry(q,values)))
    prediction_hashes={p.name:sha(p) for p in pred_folder.iterdir()};dump(ROOT/'MLP_PREDICTION_FREEZE.json',prediction_hashes)
    gate_pass=all(g['analytic_autograd_relative']<=1e-4 and g['FP64_first_relative']<=1e-3 and g['FP64_second_relative']<=.01 and g['formula_module_relative']<=1e-5 for g in gates)
    dump(ROOT/'MLP_NUMERICAL_GATE.json',{'pass':gate_pass,'calibration_checks':gates});dump(ROOT/'SUBSPACE_GEOMETRY.json',geometries)
    STATUS['gates']['MLP_numerical']=gate_pass
    if not gate_pass:raise ArithmeticError('MLP numerical gate failed; dependent native phases not run')
    mlp_rows=[];response_lookup={};response_folder=ROOT/'MLP_responses';response_folder.mkdir()
    for question in compiled:
        pid=question['id']
        for position in ['prefill','decode1']:
            for di in range(4):
                h,v,j,hessian=predictions[pid,position,di]
                for precision,dtype in [('FP32',torch.float32),('BF16',torch.bfloat16)]:
                    local=local_models[precision];base_h=h.to(dtype);native_v=v.to(dtype)
                    with torch.no_grad():base_branch=local.branch(base_h);base_full=base_h+base_branch
                    storage={k:[] for k in ['actual_delta','branch_delta','complete_delta','linear_actual_branch','quadratic_actual_branch','linear_actual_complete','quadratic_actual_complete']}
                    for strength in PLAN['signed_absolute_strengths']:
                        pair={}
                        for sign in [1,-1]:
                            deadline();alpha=sign*strength
                            with torch.no_grad():
                                update=alpha*native_v;changed=base_h+update;out=local.branch(changed);whole=changed+out
                                actual=changed.double()-base_h.double();_,ja,ha=mlp32.derivatives(h,actual)
                            delta_b=array(out.double()-base_branch.double());delta_f=array(whole.double()-base_full.double());delta_x=array(actual)
                            lb=array(ja);qb=array(ja+.5*ha);lf=delta_x+lb;qf=delta_x+qb
                            pair[sign]=(delta_x,delta_b,delta_f)
                            response_lookup[pid,position,di,precision,strength,sign]=pair[sign]
                            for key,value in zip(storage,[delta_x,delta_b,delta_f,lb,qb,lf,qf]):storage[key].append(value)
                            for kind,delta,nominal,second,linear_actual,quadratic_actual,base_value in [
                                ('branch',delta_b,alpha*array(j),alpha*array(j)+.5*alpha*alpha*array(hessian),lb,qb,array(base_branch)),
                                ('complete',delta_f,alpha*(array(v)+array(j)),alpha*(array(v)+array(j))+.5*alpha*alpha*array(hessian),lf,qf,array(base_full))]:
                                floor=1e-5*max(1.,norm(base_value));size=norm(delta)
                                mlp_rows.append(dict(question=pid,split=question['split'],position=position,direction=di,direction_split='calibration' if di<2 else 'held_out',
                                    precision=precision,map=kind,strength=strength,sign=sign,actual_input_norm=norm(delta_x),actual_projection=float(delta_x@array(v)),
                                    input_off_direction_norm=norm(delta_x-float(delta_x@array(v))*array(v)),zero_delivered=norm(delta_x)==0,zero_output=size==0,
                                    output_norm=size,gain=size/norm(delta_x) if norm(delta_x)>0 else None,resolved=size>floor,
                                    first_absolute_error=norm(delta-nominal),first_relative_error=relative(delta-nominal,delta),
                                    second_absolute_error=norm(delta-second),second_relative_error=relative(delta-second,delta),
                                    actual_input_linear_absolute_error=norm(delta-linear_actual),actual_input_linear_relative_error=relative(delta-linear_actual,delta),
                                    actual_input_quadratic_absolute_error=norm(delta-quadratic_actual),actual_input_quadratic_relative_error=relative(delta-quadratic_actual,delta)))
                        for kind,index in [('input',0),('branch',1),('complete',2)]:
                            dump(response_folder/f'{pid}_{position}_d{di}_{precision}_eps{strength}_{kind}_parity.json',
                                {'question':pid,'position':position,'direction':di,'precision':precision,'strength':strength,'map':kind,**parity(pair[1][index],pair[-1][index])})
                    np.savez_compressed(response_folder/f'{pid}_{position}_d{di}_{precision}.npz',**{k:np.stack(vs) for k,vs in storage.items()})
                STATUS['completed']['MLP_direction_base_pairs']=STATUS['completed'].get('MLP_direction_base_pairs',0)+1
            report('MLP_pilot',current_question=pid,current_position=position)
    dump(ROOT/'MLP_RESPONSE_METRICS.json',mlp_rows)
    superposition=[]
    for question in compiled:
        pid=question['id']
        for position in ['prefill','decode1']:
            h=predictions[pid,position,0][0]
            for d1,d2 in [(0,1),(2,3)]:
                for precision,dtype in [('FP32',torch.float32),('BF16',torch.bfloat16)]:
                    local=local_models[precision];base_h=h.to(dtype);summed=(directions[d1]+directions[d2]).to(dtype)
                    with torch.no_grad():b0=local.branch(base_h);f0=base_h+b0
                    for eps in [.03125,.5]:
                        for sign in [1,-1]:
                            deadline()
                            with torch.no_grad():changed=base_h+(sign*eps)*summed;bs=local.branch(changed);fs=changed+bs
                            actual_sum=array(changed.double()-base_h.double());one=response_lookup[pid,position,d1,precision,eps,sign];two=response_lookup[pid,position,d2,precision,eps,sign]
                            input_gap=actual_sum-one[0]-two[0]
                            with torch.no_grad():_,jg,_=mlp32.derivatives(h,torch.tensor(input_gap))
                            for kind,index,delta in [('branch',1,array(bs.double()-b0.double())),('complete',2,array(fs.double()-f0.double()))]:
                                error=delta-one[index]-two[index];pred=array(jg)+(input_gap if kind=='complete' else 0)
                                superposition.append(dict(question=pid,position=position,directions=[d1,d2],precision=precision,strength=eps,sign=sign,map=kind,
                                    actual_input_sum_gap_norm=norm(input_gap),absolute_error=norm(error),relative_error=norm(error)/max(1e-12,norm(one[index])+norm(two[index])),
                                    linear_input_rounding_correction_remainder=norm(error-pred)))
    dump(ROOT/'SUPERPOSITION.json',superposition)

    begin('functional_energy');rig.precision(torch.float32)
    energy_folder=ROOT/'functional_energy';energy_folder.mkdir();energy_rows=[];energy_checks=[];functional_pass=True;ad_method=None;ad_compatibility_failures=[]
    for split in ['calibration','held_out']:
        if split=='held_out' and not functional_pass:break
        for question in [p for p in compiled if p['split']==split]:
            pid=question['id'];ids=question['prefix_ids'];h=contexts[pid,'FP32','prefill']['data']['mlp_h'].float().detach()
            local=LocalMLP(rig.layers[11].post_attention_layernorm,rig.layers[11].mlp)
            for di in ([0,1] if split=='calibration' else [2,3]):
                v=directions[di]
                def function(alpha):
                    def hook(module,args,output):
                        x=h+alpha*v;x0=h+alpha*0
                        delta=(x+local.branch(x))-(x0+local.branch(x0))
                        return torch.cat([output[:,:-1],output[:,-1:]+delta.reshape(1,1,-1)],dim=1)
                    return rig.hidden_function(ids,hook)
                deadline()
                if ad_method is None:
                    try:
                        with torch.no_grad(),torch.autograd.forward_ad.dual_level():
                            dual=torch.autograd.forward_ad.make_dual(torch.tensor(0.),torch.tensor(1.))
                            value=function(dual);base_hidden,dh=torch.autograd.forward_ad.unpack_dual(value)
                        if dh is None:raise RuntimeError('No forward tangent returned')
                        ad_method='forward_AD'
                    except (NotImplementedError,RuntimeError) as compatibility:
                        if 'memory' in str(compatibility).lower():raise
                        ad_compatibility_failures.append(type(compatibility).__name__+': '+str(compatibility))
                        with torch.enable_grad():base_hidden,dh=torch.autograd.functional.jvp(function,torch.tensor(0.),torch.tensor(1.),create_graph=False)
                        ad_method='reverse_over_reverse_JVP'
                    dump(ROOT/'FUNCTIONAL_AD_METHOD.json',{'method':ad_method,'first_calibration_compatibility_fallback_only':ad_compatibility_failures})
                elif ad_method=='forward_AD':
                    with torch.no_grad(),torch.autograd.forward_ad.dual_level():
                        dual=torch.autograd.forward_ad.make_dual(torch.tensor(0.),torch.tensor(1.))
                        value=function(dual);base_hidden,dh=torch.autograd.forward_ad.unpack_dual(value)
                    assert dh is not None
                else:
                    with torch.enable_grad():base_hidden,dh=torch.autograd.functional.jvp(function,torch.tensor(0.),torch.tensor(1.),create_graph=False)
                with torch.no_grad():base_logits=rig.model.lm_head(base_hidden);qlogits=rig.model.lm_head(dh)
                coefficient=fisher_direction(array(base_logits),array(qlogits))
                pred_path=energy_folder/f'{pid}_d{di}_prediction.npz'
                np.savez_compressed(pred_path,base_logits=array(base_logits),q_logits=array(qlogits),base_hidden=array(base_hidden),derivative_hidden=array(dh),h=array(h),direction=array(v))
                dump(pred_path.with_suffix('.json'),{'predicted_before_measurement':True,'prediction_sha256':sha(pred_path),'fisher_variance':coefficient,'question':pid,'direction':di,'split':split})
                measurements={}
                for eps in [.03125,.125,.5]:
                    for sign in [1,-1]:
                        deadline();alpha=sign*eps
                        with torch.no_grad():hidden=function(torch.tensor(alpha));logits=rig.model.lm_head(hidden)
                        measured=kl_from_logits(array(base_logits),array(logits));predicted=.5*alpha*alpha*coefficient
                        measurements[eps,sign]=array(logits)
                        np.savez_compressed(energy_folder/f'{pid}_d{di}_eps{eps}_sign{sign}.npz',logits=array(logits),final_hidden=array(hidden))
                        energy_rows.append(dict(question=pid,split=split,direction=di,strength=eps,sign=sign,predicted_KL=predicted,measured_KL_raw=measured,
                            absolute_error=abs(predicted-measured),relative_error=abs(predicted-measured)/measured if measured>1e-10 else None,
                            resolved_KL=measured>1e-10,geometric_final_hidden_displacement=norm(array(hidden-base_hidden)),fisher_variance=coefficient))
                fd=(measurements[.03125,1]-measurements[.03125,-1])/(2*.03125)
                check=weighted_derivative_error(array(base_logits),array(qlogits),fd);energy_checks.append({'question':pid,'split':split,'direction':di,**check})
                if split=='calibration' and (not check['resolved'] or check['relative_error']>.05):functional_pass=False
                STATUS['completed']['functional_direction_cases']=STATUS['completed'].get('functional_direction_cases',0)+1
                report('functional_energy',current_question=pid,current_direction=di)
        if split=='calibration':
            dump(ROOT/'FUNCTIONAL_NUMERICAL_GATE.json',{'pass':functional_pass,'checks':energy_checks.copy()});STATUS['gates']['functional_numerical']=functional_pass
    dump(ROOT/'FUNCTIONAL_ENERGY.json',energy_rows);dump(ROOT/'FUNCTIONAL_DERIVATIVE_CHECKS.json',energy_checks)

    begin('attention_and_memory');attention_rows=[];attention_checks=[];attention_pass=True;attn_folder=ROOT/'attention_response';attn_folder.mkdir()
    for precision,dtype in [('FP32',torch.float32),('BF16',torch.bfloat16)]:
        layer=copies[precision]['attention_layer']
        for question in [compiled[0],compiled[2]]:
            if question['split']=='held_out' and not attention_pass:continue
            pid=question['id']
            for position in ['prefill','decode1']:
                ctx=contexts[pid,precision,position];fn,h,original_hash=attention_function(layer,ctx)
                with torch.no_grad():base=fn(h)
                reference=ctx['data']['attn_branch'];base_error=norm(array(base-reference))/max(1.,norm(array(reference)))
                assert base_error<=1e-5,'Attention context replay mismatch'
                for di in [0,2]:
                    v=directions[di].to(dtype);j=None
                    if precision=='FP32':
                        deadline()
                        with torch.enable_grad():_,j=torch.autograd.functional.jvp(fn,h,v,create_graph=False)
                        np.savez_compressed(attn_folder/f'{pid}_{position}_d{di}_prediction.npz',h=array(h),direction=array(v),branch_base=array(base),Jv=array(j))
                    pairs={}
                    for eps in [.03125,.5]:
                        for sign in [1,-1]:
                            deadline()
                            with torch.no_grad():changed=h+(sign*eps)*v;result=fn(changed)
                            delta=array(result.double()-base.double());actual=array(changed.double()-h.double());pairs[eps,sign]=(delta,actual)
                            full_delta=array((changed+result).double()-(h+base).double())
                            for kind,dy,pred in [('branch',delta,sign*eps*array(j) if j is not None else None),
                                ('complete',full_delta,sign*eps*(array(v)+array(j)) if j is not None else None)]:
                                attention_rows.append(dict(question=pid,split=question['split'],position=position,precision=precision,direction=di,map=kind,
                                    strength=eps,sign=sign,actual_input_norm=norm(actual),output_norm=norm(dy),zero_input=norm(actual)==0,
                                    first_absolute_error=norm(dy-pred) if pred is not None else None,
                                    first_relative_error=relative(dy-pred,dy) if pred is not None else None,
                                    context_replay_relative_error=base_error,held_cache_unchanged=(cache_hash(ctx['data']['attn_kwargs'].get('past_key_values'))==original_hash)))
                            np.savez_compressed(attn_folder/f'{pid}_{position}_d{di}_{precision}_eps{eps}_sign{sign}.npz',actual_delta=actual,branch_delta=delta,complete_delta=full_delta)
                    if precision=='FP32':
                        fd=(pairs[.03125,1][0]-pairs[.03125,-1][0])/(2*.03125)
                        error=relative(fd-array(j),array(j));attention_checks.append(dict(question=pid,position=position,direction=di,FD_relative_error=error))
            if precision=='FP32' and question['split']=='calibration':
                attention_pass=all(c['FD_relative_error'] is not None and c['FD_relative_error']<=.05 for c in attention_checks)
                dump(ROOT/'ATTENTION_CALIBRATION_GATE.json',{'pass':attention_pass,'checks':attention_checks.copy()})
                STATUS['gates']['attention_numerical']=attention_pass
    dump(ROOT/'ATTENTION_RESPONSE.json',attention_rows);dump(ROOT/'ATTENTION_NUMERICAL_GATE.json',{'checks':attention_checks,
        'pass':all(c['FD_relative_error'] is not None and c['FD_relative_error']<=.05 for c in attention_checks if c['question']==compiled[0]['id'])})

    memory_rows=[];memory_folder=ROOT/'causal_memory';memory_folder.mkdir();legacy32=torch.tensor(legacy)
    conditions=[0.,.03125,-.03125,.5,-.5]
    def path_call(ids,length,cache,alpha,schedule,precision):
        delivered=[]
        def hook(module,args,output):
            vector=legacy32.to(output.dtype);update=alpha*vector
            if schedule=='all':changed=output+update
            else:changed=torch.cat([output[:,:-1],output[:,-1:]+update.reshape(1,1,-1)],dim=1)
            delivered.append((changed.double()-output.double())[0,-1].float().numpy().copy())
            return changed
        answer=rig.forward(ids,length,cache,capture=True,edit=(10,hook))
        return answer,np.asarray(delivered)[0]
    for precision,dtype in [('FP32',torch.float32),('BF16',torch.bfloat16)]:
        rig.precision(dtype)
        for question in compiled[2:]:
            pid=question['id'];ids=question['prefix_ids'];prefix=prefix_caches[pid,precision]
            frozen_hash=cache_hash(prefix)
            for position in ['prefill','decode1']:
                for alpha in conditions:
                    deadline();inp=ids if position=='prefill' else [760];length=len(ids)+(position=='decode1')
                    answer,delta=path_call(inp,length,None if position=='prefill' else prefix,alpha,'last',precision)
                    name=f'{pid}_{precision}_{position}_fixed_alpha{alpha}'
                    np.savez_compressed(memory_folder/(name+'.npz'),readout15=array(answer['data']['readout15']),site10_before=array(answer['data']['site10']),
                        delivered=delta,logits=array(answer['logits']))
                    memory_rows.append(dict(question=pid,precision=precision,mode='held_fixed_state',position=position,alpha=alpha,file=name+'.npz'))
            for schedule in ['all','last']:
                for alpha in conditions:
                    if schedule=='last' and alpha==0:continue
                    cache=None;outputs=[];deltas=[];site=[];logits=[]
                    for step in range(4):
                        deadline();inp=ids if step==0 else [PLAN['causal_memory_followup']['tokens'][step-1]]
                        answer,delta=path_call(inp,len(ids)+step,cache,alpha,schedule,precision)
                        cache=answer['cache'];outputs.append(array(answer['data']['readout15']));site.append(array(answer['data']['site10']));deltas.append(delta);logits.append(array(answer['logits']))
                    name=f'{pid}_{precision}_persistent_{schedule}_alpha{alpha}'
                    np.savez_compressed(memory_folder/(name+'.npz'),readout15=np.stack(outputs),site10_before=np.stack(site),
                        delivered=np.stack(deltas),logits=np.stack(logits),forced_ids=np.asarray(PLAN['causal_memory_followup']['tokens']))
                    memory_rows.append(dict(question=pid,precision=precision,mode='persistent_'+schedule,position='four_steps',alpha=alpha,file=name+'.npz'))
                    STATUS['completed']['causal_trajectories']=STATUS['completed'].get('causal_trajectories',0)+1
            assert cache_hash(prefix)==frozen_hash
            report('attention_and_memory',current_question=pid,precision=precision)
    dump(ROOT/'CAUSAL_MEMORY_INDEX.json',memory_rows)
    STATUS['status']='complete';STATUS['whole_backbone_calls']=rig.calls
    assert rig.calls<=PLAN['total_whole_backbone_calls_max'],(rig.calls,PLAN['total_whole_backbone_calls_max'])
    assert all(not layer._forward_hooks and not layer._forward_pre_hooks for layer in rig.layers)
    report('complete_export_pending')
except BaseException as error:
    STATUS['status']='fault_or_numerical_gate_stop_no_automatic_retry'
    report('fault',error=type(error).__name__+': '+str(error));(ROOT/'FAULT.txt').write_text(traceback.format_exc())
    raise
finally:
    export();TIMER.cancel()
