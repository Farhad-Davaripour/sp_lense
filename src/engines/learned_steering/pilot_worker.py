"""Bounded TRAIN diagnostic. No credentials, model tuning or held-out inputs."""
import argparse,ast,hashlib,json,pathlib,sys,time
import torch
from datetime import datetime,timezone
DEADLINE_FILE=None;PHASE_END=None
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from native_sites import ResidualSites,clone_state,state_digest
from controller_math import matched_maps,ridge_initialize,distillation_objective

def now():
    from datetime import datetime,timezone
    return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def save(p,v):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def file_sha(p):
    with pathlib.Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def kl(teacher,student):
    t=teacher.float();s=student.float();p=torch.softmax(t,-1)
    return float((p*(torch.log_softmax(t,-1)-torch.log_softmax(s,-1))).sum(-1).mean())

class Engine:
    def __init__(self,model,deadline,root):
        self.model,self.deadline,self.root=model,deadline,root
        self.attempts=self.completed=self.tokens=0
    def guard(self):
        end=self.deadline
        if DEADLINE_FILE is not None and DEADLINE_FILE.is_file():
            remaining=(datetime.fromisoformat(read(DEADLINE_FILE)['neural_end_UTC'])-datetime.now(timezone.utc)).total_seconds()
            end=min(PHASE_END,time.monotonic()+remaining)
        if time.monotonic()>=end:raise TimeoutError('Fixed pilot neural or prepaid-credit window exhausted')
    def forward(self,ids,past,total,grad=False,use_cache=True):
        self.guard();self.attempts+=1
        x=torch.tensor([ids],device='cuda');mask=torch.ones((1,total),device='cuda',dtype=torch.long)
        with torch.set_grad_enabled(grad):out=self.model(input_ids=x,attention_mask=mask,past_key_values=past,use_cache=use_cache,logits_to_keep=1)
        self.completed+=1
        save(self.root/'PHYSICAL_COUNTS.json',{'utc':now(),'model_forward_attempts':self.attempts,'model_forwards_completed':self.completed,'emitted_token_IDs':self.tokens,'new_reference_calls_in_GPU_worker':0})
        return out
    def replay(self,panel,hook=None,grad_anchor=False):
        prefix=panel['prompt_ids'];past=None;previous=panel['past_generated_ids']
        for index in range(len(previous)+1):
            if hook:hook.record=index==len(previous);hook.enabled=panel['gate_enabled']
            if grad_anchor and index==len(previous) and past is not None:past=clone_state(past)
            out=self.forward(prefix if index==0 else [previous[index-1]],past,len(prefix)+index,grad_anchor and index==len(previous))
            past=out.past_key_values
            if index<len(previous):del out
        return out
    def proposal(self,panel,hook,cap):
        out=self.replay(panel,hook);ids=[];past=out.past_key_values
        eos=self.model.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos)
        while len(ids)<cap:
            token=int(out.logits[0,-1].argmax());ids.append(token);self.tokens+=1
            if token in eos or len(ids)>=cap:break
            if hook:hook.record=False
            del out
            out=self.forward([token],past,len(panel['prompt_ids'])+len(panel['past_generated_ids'])+len(ids));past=out.past_key_values
        del out,past
        return ids
    def uncached_zero_proposal(self,panel,hook,cap):
        context=panel['prompt_ids']+panel['past_generated_ids'];ids=[]
        for _ in range(cap):
            if hook:hook.enabled=panel['gate_enabled'];hook.record=False
            out=self.forward(context+ids,None,len(context)+len(ids),use_cache=False)
            token=int(out.logits[0,-1].argmax());ids.append(token);self.tokens+=1;del out
        return ids

def mechanics(engine,layers,teacher_layers,panel,root):
    assert layers[30].block_type=='linear_attention' and layers[31].block_type=='full_attention'
    rows=[];raw=engine.proposal(panel,None,8)
    uncached_raw=engine.uncached_zero_proposal(panel,None,8)
    assert uncached_raw==raw,'Native unwrapped cached/uncached greedy IDs differ'
    for variant in ('one_site','two_site'):
        hook=ResidualSites(layers,None,[30,31],variant,'zero')
        try:
            zero=engine.proposal(panel,hook,8)
            uncached_zero=engine.uncached_zero_proposal(panel,hook,8)
        finally:hook.close()
        assert zero==uncached_zero==raw,'Zero wrapper changed native greedy IDs'
        save(root/(variant+'_ZERO_CACHE_CHECK.json'),{'utc':now(),'passed':True,'cached_uncached_zero_and_native_IDs_exact':True,'tokens':8,'original_full_prompt_prefill_retained':True})
        hook=ResidualSites(layers,teacher_layers,[30,31],variant,'reference')
        hook.audit_other_layers=True
        try:
            first=engine.replay(panel,hook);a=first.logits.detach().float().cpu();records=list(hook.records);hook.records=[];del first
            second=engine.replay(panel,hook);b=second.logits.detach().float().cpu();del second
            relative=float((a-b).norm()/b.norm().clamp_min(1e-8));assert relative<=.01
            assert records and all(r['current_state_not_rewritten'] for r in records)
            torch.save(records,root/(variant+'_representative_reference_private.pt'))
            rows.append({'variant':variant,'zero_greedy_IDs_exact':True,'zero_and_unwrapped_both_cache_paths_exact':True,'original_full_prompt_prefill_retained':True,'independently_rebuilt_literal_cached_logits_relative_L2':relative,'actual_blocks':[30,31],'state_not_rewritten':True})
        finally:hook.close()
    save(root/'REPRESENTATIVE_MECHANICS.json',{'utc':now(),'passed':True,'checks':rows,'scope':'two representative blocks and fixed TRAIN prefix only'})

def linearization(engine,layers,teacher_layers,teacher_model,panel,root):
    reports={}
    for index in (30,31):
        for kind in ('attention','output'):
            layer=teacher_layers[index]
            module=layer.mlp if kind=='output' else layer.linear_attn if layer.block_type=='linear_attention' else layer.self_attn
            scaling=[(m,dict(m.scaling)) for m in module.modules() if isinstance(getattr(m,'scaling',None),dict)];assert scaling
            targets={};inputs={}
            try:
                for alpha in (0.,.5,1.):
                    for m,original in scaling:
                        for name,value in original.items():m.scaling[name]=value*alpha
                    # One block at a time prevents lower edited blocks from
                    # changing this probe's incoming residual across strengths.
                    # For the MLP curve, attention stays at full strength.
                    hook=ResidualSites(layers,teacher_layers,[index],'two_site','reference')
                    try:
                        out=engine.replay(panel,hook);del out;row=hook.records[0]
                        targets[alpha]=row[kind+'_target']
                        inputs[alpha]=row['attention_input'] if kind=='attention' else row['attention_input']+row['attention_effective_delta']
                    finally:hook.close()
            finally:
                for m,original in scaling:m.scaling.clear();m.scaling.update(original)
            assert torch.equal(inputs[0.],inputs[.5]) and torch.equal(inputs[.5],inputs[1.]),'Local strength probe input changed'
            value=targets[1.]
            reports[f'{index}/{kind}']={'alpha0_norm':float(targets[0.].norm()),'alpha1_norm':float(value.norm()),
                'first_order_half_to_full_relative_L2':float((value-2*targets[.5]).norm()/value.norm().clamp_min(1e-8)),
                'same_local_input_verified':True,'lower_block_interventions_absent':True}
    save(root/'LOCAL_LINEARIZATION.json',{'utc':now(),'adapter_strengths':[0,.5,1],'actual_blocks':[30,31],'teacher_scaling_restored':True,'no_weight_merge_or_optimizer':True,'reports':reports})

def collect(engine,layers,teacher_layers,panels,variant,root):
    data={f'{i}/{kind}':[] for i in range(64) for kind in (('output',) if variant=='one_site' else ('attention','output'))};logits={}
    for p in panels:
        hook=ResidualSites(layers,teacher_layers,range(64),variant,'reference')
        try:
            out=engine.replay(p,hook);logits[p['pilot_index']]=out.logits.detach().float().cpu();del out
            by_layer={r['layer']:r for r in hook.records};assert len(by_layer)==64
            for i,r in by_layer.items():
                for kind in (('output',) if variant=='one_site' else ('attention','output')):
                    x=r[kind+'_input'];y=r[kind+'_target'] if p['gate_enabled'] else torch.zeros_like(x)
                    data[f'{i}/{kind}'].append({'pilot_index':p['pilot_index'],'case_id':p['case_id'],'gate_on':p['gate_enabled'],'x':x.reshape(1,-1),'y':y.reshape(1,-1)})
            torch.save(hook.records,root/f"anchor_{p['pilot_index']:02}_private.pt")
        finally:hook.close()
        save(root/'PROGRESS.json',{'utc':now(),'captured_prefixes':len(logits),'scheduled_prefixes':len(panels)})
    torch.save(data,root/'LOCAL_TARGETS_PRIVATE.pt')
    return data,logits

def fit_pilot(data,variant,root):
    maps=matched_maps(5120,range(64),16,variant).cuda();metadata={};projectors={}
    for key,rows in data.items():
        x=torch.cat([r['x'] for r in rows]).cuda();y=torch.cat([r['y'] for r in rows]).cuda()
        counts={r['case_id']:sum(v['case_id']==r['case_id'] for v in rows) for r in rows}
        w=torch.tensor([1/counts[r['case_id']] for r in rows],device='cuda')
        metadata[key]=ridge_initialize(maps[key],x,y,w)
        projectors[key]=maps[key].up.detach().clone()
    from safetensors.torch import save_file
    save_file({k:v.detach().cpu().contiguous() for k,v in maps.state_dict().items()},str(root/'PILOT_RIDGE_INITIALIZER.safetensors'))
    save(root/'RIDGE_DIAGNOSTIC.json',{'utc':now(),'learned_parameters':sum(p.numel() for p in maps.parameters()),'rank_one_site':16,'rank_two_site_each':8,'rows_per_function':12,'full_TRAIN_capacity_not_selected':True,'functions':metadata})
    return maps,projectors

def gradient_checks(engine,layers,teacher_logits,panels,maps,data,variant,root):
    opt=torch.optim.AdamW(maps.parameters(),lr=2e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    scales={k:torch.cat([r['y'] for r in rows if r['gate_on']]).square().mean().clamp_min(1e-6).cuda() for k,rows in data.items()}
    history=[]
    for update in range(2):
        opt.zero_grad(set_to_none=True)
        for p in panels:
            hook=ResidualSites(layers,None,range(64),variant,'predicted',maps)
            try:
                out=engine.replay(p,hook,grad_anchor=True)
                rows=hook.records;pred=[];targets=[];normalizers=[]
                # Frozen reference inputs are regularization anchors only. KL
                # carries gradients through the actual current student forward.
                for key,entries in data.items():
                    r=next(r for r in entries if r['pilot_index']==p['pilot_index'])
                    pred.append(maps[key](r['x'].cuda()));targets.append(r['y'].cuda());normalizers.append(scales[key])
                loss,terms=distillation_objective(out.logits,teacher_logits[p['pilot_index']].cuda(),pred,targets,normalizers,p['gate_enabled'])
                (loss/len(panels)).backward();history.append({'update':update,'prefix':p['pilot_index'],'loss':float(loss.detach())});del out
            finally:hook.close()
        assert all(p.grad is None for layer in layers for p in layer.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in maps.parameters())
        norm=torch.nn.utils.clip_grad_norm_(maps.parameters(),1.);assert torch.isfinite(norm)
        opt.step();save(root/'GRADIENT_PROGRESS.json',{'utc':now(),'completed_updates':update+1,'cap':2,'frozen_generator_gradient_absent':True})
    save(root/'GRADIENT_CHECK.json',{'utc':now(),'completed_updates':2,'passed':True,'scratch_mechanics_only':True,'losses':history})

def main(a):
    global DEADLINE_FILE,PHASE_END
    inputs,root=pathlib.Path(a.inputs),pathlib.Path(a.root);root.mkdir(exist_ok=False)
    deadline=time.monotonic()+a.max_seconds;freeze=read(inputs/'EXECUTION_FREEZE.json')
    PHASE_END=deadline;DEADLINE_FILE=inputs/'RUN_DEADLINE.json'
    admission=read(inputs/'NATIVE_ADMISSION.json');assert admission['pilot_admitted'] and (admission['credit_fresh'] or admission.get('user_start_now_partial_credit_guarantee')) and admission['worker_seconds']>=a.max_seconds
    legacy=inputs/'loader/source/qwen38_colab'
    sys.path[:0]=[str(legacy/'frozen_base_control_v1'),str(legacy/'coverage_repair_v3'),str(legacy/'initial_gate_ridge_v1'),str(inputs/'loader/simulator')]
    from campaign_worker import load_model,layers_of,verify_inputs
    from gpu_validation_worker import digest_model
    from initial_gate import decode_reference
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(20261012)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    requests={r['case_id']:r for r in read(inputs/'TRAIN_INITIAL_REQUESTS_PRIVATE.json')}
    responses=read(inputs/'PILOT_reference_RECEIPTS_PRIVATE.json');panels=read(inputs/'PILOT_PREFIXES_PRIVATE.json')
    for p in panels:
        r=responses[p['case_id']];assert r['actual_service_call'] and r['request_sha256']==requests[p['case_id']]['request_sha256']
        p['gate_enabled']=decode_reference(requests[p['case_id']]['request'],r['response']).enabled
    verify_inputs(root,pathlib.Path(a.model),pathlib.Path(a.adapter))
    base,tok=load_model(pathlib.Path(a.model));teacher,teacher_tok=load_model(pathlib.Path(a.model),pathlib.Path(a.adapter))
    before={'base':digest_model(base),'teacher':digest_model(teacher)};assert before==read(inputs/'PROTECTED_DIGESTS.json')
    layers,teacher_layers=layers_of(base),layers_of(teacher);assert len(layers)==len(teacher_layers)==64
    engine=Engine(base,deadline,root);teacher_engine=Engine(teacher,deadline,root/'teacher_counts');tlogits={};blogits={}
    save(root/'MODEL_READY.json',{'utc':now(),'protected_before':before,'generator_parameters_frozen':True,'teacher_is_offline_diagnostic_only':True})
    for p in panels:
        out=engine.replay(p);blogits[p['pilot_index']]=out.logits.detach().float().cpu();del out
        if p['gate_enabled']:
            out=teacher_engine.replay(p);tlogits[p['pilot_index']]=out.logits.detach().float().cpu();del out
        else:tlogits[p['pilot_index']]=blogits[p['pilot_index']].clone()
    first_own=next(p for p in panels if p['group']=='own' and not p['past_generated_ids'])
    mechanics(engine,layers,teacher_layers,first_own,root)
    linearization(engine,layers,teacher_layers,teacher,first_own,root)
    collected={};reference_logits={};maps={};projectors={}
    for variant in ('one_site','two_site'):
        folder=root/variant;folder.mkdir()
        collected[variant],reference_logits[variant]=collect(engine,layers,teacher_layers,panels,variant,folder)
        maps[variant],projectors[variant]=fit_pilot(collected[variant],variant,folder)
    from world import parse
    results=[]
    for p in panels:
        if p['group']!='own' or p['past_generated_ids']:continue
        for mode,actor in (('base_diagnostic',engine),('teacher_diagnostic',teacher_engine if p['gate_enabled'] else engine)):
            ids=actor.proposal(p,None,128);text=tok.decode(ids,skip_special_tokens=True);actions,error=parse(text)
            results.append({'mode':mode,'prefix':p['pilot_index'],'group':p['group'],'proposal_token_IDs':ids,'proposal_text':text,'parse_error':error,'valid_call_count':len(actions) if not error else 0,'workflow_success_not_scored':True})
    for variant in ('one_site','two_site'):
        for mode in ('reference','projected'):
            for p in panels:
                hook=ResidualSites(layers,teacher_layers if mode!='predicted' else None,range(64),variant,'reference' if mode!='predicted' else 'predicted',maps[variant] if mode=='predicted' else None,projectors[variant] if mode=='projected' else None)
                try:
                    out=engine.replay(p,hook);logits=out.logits.detach().float().cpu();del out
                    row={'variant':variant,'mode':mode,'prefix':p['pilot_index'],'group':p['group'],'gate_enabled':p['gate_enabled'],
                        'forward_KL_to_teacher':kl(tlogits[p['pilot_index']],logits),'base_forward_KL':kl(tlogits[p['pilot_index']],blogits[p['pilot_index']])}
                    if p['group']=='own' and not p['past_generated_ids']:
                        ids=engine.proposal(p,hook,128);text=tok.decode(ids,skip_special_tokens=True);actions,error=parse(text)
                        row.update(proposal_token_IDs=ids,proposal_text=text,parse_error=error,valid_call_count=len(actions) if not error else 0,
                                   complete_response_not_established=True,workflow_success_not_scored=True)
                    results.append(row);save(root/'REFERENCE_COMPRESSION_PREDICTION_PRIVATE.json',results)
                finally:hook.close()
    teacher_after=digest_model(teacher);assert teacher_after==before['teacher']
    save(root/'OFFLINE_TEACHER_INTEGRITY.json',{'before':before['teacher'],'after':teacher_after,'unchanged':True})
    del actor,teacher_engine,teacher_layers,teacher,teacher_tok
    import gc
    gc.collect();torch.cuda.empty_cache()
    save(root/'TEACHER_UNLOADED.json',{'utc':now(),'teacher_model_absent_before_predicted_rollouts_and_gradients':True,'teacher_logits_are_offline_identical_prefix_targets':True})
    for variant in ('one_site','two_site'):
        for p in panels:
            hook=ResidualSites(layers,None,range(64),variant,'predicted',maps[variant])
            try:
                out=engine.replay(p,hook);logits=out.logits.detach().float().cpu();del out
                row={'variant':variant,'mode':'predicted','prefix':p['pilot_index'],'group':p['group'],'gate_enabled':p['gate_enabled'],'teacher_model_absent':True,
                    'forward_KL_to_teacher':kl(tlogits[p['pilot_index']],logits),'base_forward_KL':kl(tlogits[p['pilot_index']],blogits[p['pilot_index']])}
                if p['group']=='own' and not p['past_generated_ids']:
                    ids=engine.proposal(p,hook,128);text=tok.decode(ids,skip_special_tokens=True);actions,error=parse(text)
                    row.update(proposal_token_IDs=ids,proposal_text=text,parse_error=error,valid_call_count=len(actions) if not error else 0,workflow_success_not_scored=True)
                results.append(row);save(root/'REFERENCE_COMPRESSION_PREDICTION_PRIVATE.json',results)
            finally:hook.close()
        gradient_checks(engine,layers,tlogits,panels,maps[variant],collected[variant],variant,root/variant)
    after={'base':digest_model(base),'teacher':teacher_after};assert before==after
    save(root/'MODEL_INTEGRITY.json',{'before':before,'after':after,'unchanged':True})
    own=[p for p in panels if p['group']=='own' and not p['past_generated_ids']]
    eligible={}
    for variant in ('one_site','two_site'):
        r=[r for r in results if r.get('variant')==variant and r['mode']=='reference' and r['prefix'] in {p['pilot_index'] for p in own}]
        eligible[variant]=all(row['base_forward_KL']>0 and row['forward_KL_to_teacher']<=.9*row['base_forward_KL'] for row in r) and len(r)==2
    save(root/'COMPLETE.json',{'utc':now(),'completed':True,'pilot_unique_cases':4,'pilot_prefixes':12,'teacher_reference_supports_scaling':eligible,
        'model_weights_unchanged':True,'full_TRAIN_capacity_selected':False,'new_development_or_holdout_episodes':0,
        'predicted_comparisons_teacher_model_absent':True,'full_development_teacher_absence_check_still_required':True})

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('inputs','root','model','adapter'):p.add_argument('--'+name,required=True)
    p.add_argument('--max-seconds',type=int,required=True);a=p.parse_args()
    try:main(a)
    except Exception as e:
        save(pathlib.Path(a.root)/'SANITIZED_FAULT.json',{'utc':now(),'type':type(e).__name__,'message':str(e)[:200],'automatic_retry':False,'partial_results_preserved':True})
        raise
