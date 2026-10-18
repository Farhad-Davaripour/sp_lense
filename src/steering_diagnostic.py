"""Bounded teacher-forced training diagnostic; never tune or execute a workflow."""
from pathlib import Path
import argparse
import collections
import datetime as dt
import gc
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import yaml
from src.config import ROOT,Settings
from src.utils import sha256,write_json,workspace_client


def configuration():return yaml.safe_load((ROOT/'config/steering_diagnostic.yaml').read_bytes())


def prepare(training_root, gate_predictions, output):
    cfg=configuration();training_root=Path(training_root);output=Path(output)
    if output.exists():raise FileExistsError('Use a new frozen input packet')
    contexts=json.loads((training_root/'TRAIN_CONTEXTS_PRIVATE.json').read_bytes())
    cases=json.loads((training_root/'TRAIN_CASES_PRIVATE.json').read_bytes())
    gates={r['id']:r['gate'] for r in json.loads(Path(gate_predictions).read_bytes())};selected=[]
    for family,expected_id,expected_ci in zip(cfg['families'],cfg['case_ids'],cfg['context_list_indices']):
        case=min((r for r in cases if r['family']==family and r['condition'].startswith('self_')),key=lambda r:r['id'])
        if case['id']!=expected_id or not gates[case['id']]['enabled']:raise ValueError('Predeclared training case/gate differs')
        candidates=[]
        for ci,c in enumerate(contexts):
            if c['case_id']!=case['id'] or c['source_policy']!='teacher':continue
            later=next((n for n in contexts if n['case_id']==case['id'] and n['source_policy']=='teacher'
                        and n['turn_index']==c['turn_index']+1),None)
            if later is None:continue
            if later['history'][:len(c['history'])]!=c['history']:raise ValueError('Saved demonstration history differs')
            text=next((m['content'] for m in later['history'][len(c['history']):] if m['role']=='assistant'),'')
            if '<function=edit_register>' in text:candidates.append((ci,c,text))
        ci,c,text=min(candidates,key=lambda r:r[1]['turn_index'])
        if ci!=expected_ci:raise ValueError('Predeclared first register-action context differs')
        selected.append({'case_id':case['id'],'family':family,'condition':case['condition'],
            'context_list_index':ci,'parent_context_metadata_index':c['context_index'],
            'history':c['history'],'tools':c['tools'],'prefix_ids':c['prefix_ids'],
            'teacher_ids':c['source_emitted_ids'][:cfg['maximum_teacher_forced_tokens_per_case']],
            'saved_demonstration_text':text,'frozen_initial_gate':gates[case['id']],
            'role':'Previously seen correction-training case; teacher-forced action response, not an episode evaluation.'})
    packet={'format':'research3-steering-diagnostic-inputs-v1','protocol_sha256':sha256(ROOT/'config/steering_diagnostic.yaml'),
        'selection_rule':cfg['selection'],'source_contexts_sha256':sha256(training_root/'TRAIN_CONTEXTS_PRIVATE.json'),
        'source_cases_sha256':sha256(training_root/'TRAIN_CASES_PRIVATE.json'),
        'frozen_gate_predictions_sha256':sha256(gate_predictions),'contexts':selected}
    write_json(output,packet)
    return {'packet_bytes':output.stat().st_size,'packet_sha256':sha256(output),
            'cases':[{'id':r['case_id'],'family':r['family'],'tokens':len(r['teacher_ids'])} for r in selected]}


def token_regions(tokenizer, tokens):
    text=tokenizer.decode(tokens,skip_special_tokens=True)
    spans={'function_name':[m.span(1) for m in re.finditer(r'<function=([^>]+)>',text)],
           'parameter_value':[m.span(1) for m in re.finditer(r'<parameter=[^>]+>(.*?)</parameter>',text,re.S)]}
    groups=[]
    for j in range(len(tokens)):
        start=len(tokenizer.decode(tokens[:j],skip_special_tokens=True))
        end=len(tokenizer.decode(tokens[:j+1],skip_special_tokens=True))
        groups.append(['all']+[name for name,ranges in spans.items() if any(start<b and end>a for a,b in ranges)])
    return groups


def summarize_steps(rows):
    if not rows:raise ValueError('No diagnostic tokens were scored')
    result={}
    for region in ['all','function_name','parameter_value']:
        chosen=[r for r in rows if region in r['regions']]
        result[region]={'tokens':len(chosen),**{key:sum(r[key] for r in chosen)/len(chosen) if chosen else None
            for key in ['teacher_forward_KL','demonstration_token_NLL','teacher_greedy_agreement','demonstration_token_greedy_match']}}
    return result


def run(admission_path):
    import numpy as np
    import torch
    from safetensors.torch import load_file,save_file
    from src.verify import source_digest
    from src.matched_steering import stage_sources,verify_model_file
    from src.diagnostic_hooks import StateCapture,SingleStateClamp,SixStateClamp,base_mode
    a=json.loads(Path(admission_path).read_bytes());cfg=configuration();limits=cfg['bounds'];settings=Settings.load()
    out=Path(a['output']);packet_path=Path(a['packet']);package=Path(a['comparison_inputs'])
    if out.exists() or not out.is_relative_to(Path(settings.volume_root)):raise ValueError('Use a new private Volume output')
    if (sha256(ROOT/'config/steering_diagnostic.yaml')!=a['protocol_sha256']
            or source_digest()!=a['source_digest'] or sha256(packet_path)!=a['packet_sha256']):
        raise ValueError('Admitted protocol/source/input identity differs')
    packet=json.loads(packet_path.read_bytes());cases=packet['contexts']
    if (packet['protocol_sha256']!=a['protocol_sha256'] or [c['case_id'] for c in cases]!=cfg['case_ids']
            or [c['context_list_index'] for c in cases]!=cfg['context_list_indices']
            or any(not c['frozen_initial_gate']['enabled'] for c in cases)):
        raise ValueError('Training membership or fixed gate differs')
    end=dt.datetime.fromisoformat(a['absolute_end_utc']).timestamp()-limits['export_reserve_seconds']
    setup=dt.datetime.fromisoformat(a['submitted_utc']).timestamp()+limits['setup_seconds']
    if time.time()>=min(setup,end):raise TimeoutError('Original setup window expired')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('One admitted GPU required')
    out.mkdir(parents=True);write_json(out/'ADMISSION.json',a);write_json(out/'FROZEN_PROTOCOL.json',cfg)
    stage_sources(Path(tempfile.mkdtemp(prefix='r3_diagnostic_source_')))
    from campaign_worker import load_model,layers_of
    from campaign_generation import token_ids
    from gpu_validation_worker import digest_model
    from controller_math import LowRankMap
    from native_sites import ResidualSites
    from selected_hook import SelectedPositionHook
    from causal_positions import CausalPositionDetector
    from ridge_runtime import load_controller,DeltaPredictor
    from src.engines.training.bridge54_policy import effective_available_bytes
    source_pins=json.loads((package/'INPUT_MANIFEST.json').read_bytes())['files']
    for name,pin in source_pins.items():
        if sha256(package/name)!=pin:raise ValueError('Frozen correction inputs differ')
    freeze=json.loads((package/'DEVELOPMENT_FREEZE.json').read_bytes())
    adapter=Path(a['adapter_directory'])
    if sha256(adapter/'adapter_model.safetensors')!=freeze['A_weights_sha256'] or sha256(adapter/'adapter_config.json')!=freeze['A_config_sha256']:
        raise ValueError('Frozen fine-tuned A changed')
    if sha256(a['initializer'])!=a['initializer_sha256']:raise ValueError('Initializer changed')
    model_path=Path(a['model_path']);pin=json.loads((ROOT/'config/recipes/qwen38_colab/model_pin.json').read_bytes())
    model=layers=hook=maps=sequence=None;completed=[];steps=0;started=time.monotonic()
    def guard(stage, release=False):
        if release or torch.cuda.mem_get_info()[0]<limits['GPU_reserve_GiB']*2**30:
            gc.collect();torch.cuda.empty_cache()
        record={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'stage':stage,'completed_model_forwards':steps,
            'host_available_bytes':effective_available_bytes(),'GPU_free_bytes':torch.cuda.mem_get_info()[0],
            'GPU_allocated_bytes':torch.cuda.memory_allocated(),'GPU_reserved_bytes':torch.cuda.memory_reserved(),
            'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated()}
        write_json(out/'RESOURCE_LATEST.json',record)
        if time.time()>=end:raise TimeoutError('Preserve export and cleanup reserve')
        if record['host_available_bytes']<limits['host_reserve_GiB']*2**30:raise MemoryError('Host reserve breached')
        if record['GPU_free_bytes']<limits['GPU_reserve_GiB']*2**30:raise MemoryError('GPU reserve breached')
        return record
    try:
        for entry in pin['files']:
            if time.time()>=min(setup,end):raise TimeoutError('Original setup window expired')
            verify_model_file(model_path/entry['name'],entry)
        initial=guard('before_model')
        if initial['host_available_bytes']<limits['host_initial_GiB']*2**30 or initial['GPU_free_bytes']<limits['GPU_initial_GiB']*2**30:
            raise MemoryError('Initial memory admission failed')
        torch.set_num_threads(4);torch.manual_seed(cfg['seed'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model,tok=load_model(model_path,adapter);layers=layers_of(model)
        before=digest_model(model)
        if before!=freeze['protected_digests']['teacher']:raise ValueError('Protected model fingerprint differs')
        write_json(out/'MODEL_BEFORE.json',before)
        for c in cases:
            if token_ids(tok,c['history'],c['tools'])!=c['prefix_ids']:raise ValueError('Saved literal training prefix changed')
            c['regions']=token_regions(tok,c['teacher_ids'])
            if not any('function_name' in g for g in c['regions']):raise ValueError('No action-name tokens in the frozen demonstration')
        if time.time()>=setup:raise TimeoutError('Setup cutoff exceeded before diagnostic')
        write_json(out/'MODEL_READY.json',guard('model_ready',True))
        final_state=torch.load(package/'two_boundary96.pt',map_location='cpu',weights_only=True)['maps']
        initial_state=load_file(a['initializer'])
        arrays={site:load_controller(package/f'controllers/layer_{site}/ridge_inference.npz',
            package/f'controllers/layer_{site}/CONTROLLER_METADATA.json',5120)[0] for site in cfg['sites']}
        for case in cases:
            cid=case['case_id'];folder=out/'cases'/cid;folder.mkdir(parents=True)
            write_json(folder/'CASE.json',{k:v for k,v in case.items() if k not in ['history','tools','prefix_ids','teacher_ids','saved_demonstration_text']})
            references=[];teacher_logits=[]
            def sequence(hook=None, capture_teacher=False, check_limit=None):
                nonlocal steps
                past=None;rows=[];capture=StateCapture(layers,cfg['sites']);detector=CausalPositionDetector(case['history'])
                single=isinstance(hook,SelectedPositionHook)
                if single:hook.begin_response(len(case['prefix_ids']),True)
                maximum=check_limit or len(case['teacher_ids'])
                try:
                    for j in range(maximum):
                        if j%8==0:guard(cid+':'+str(j))
                        visible=tok.decode(case['teacher_ids'][:j],skip_special_tokens=True);kinds=detector.position(visible,j)
                        if single:hook.prepare(len(case['prefix_ids'])+j,j>0,kinds)
                        elif isinstance(hook,SingleStateClamp):hook.prepare(j,kinds)
                        elif isinstance(hook,SixStateClamp):hook.prepare(j)
                        capture.current={}
                        ids=torch.tensor([case['prefix_ids'] if j==0 else [case['teacher_ids'][j-1]]],device='cuda',dtype=torch.long)
                        mask=torch.ones((1,len(case['prefix_ids'])+j),device='cuda',dtype=torch.long)
                        with torch.inference_mode():value=model(input_ids=ids,attention_mask=mask,past_key_values=past,
                                                                 use_cache=True,logits_to_keep=1)
                        past=value.past_key_values;steps+=1;logits=value.logits[0,-1].detach().float().cpu()
                        if not torch.isfinite(logits).all() or len(capture.current)!=6:raise ValueError('Invalid logits or missing native sites')
                        if isinstance(hook,SixStateClamp) and any(hook.fires.get(site)!=1 for site in cfg['sites']):
                            raise ValueError('Missing native state replacement')
                        if isinstance(hook,SingleStateClamp) and hook.fires!=1:raise ValueError('Missing single state replacement')
                        if capture_teacher:
                            teacher_logits.append(logits);references.append(capture.current)
                        else:
                            target=teacher_logits[j].double().log_softmax(-1);observed=logits.double().log_softmax(-1)
                            kl=float((target.exp()*(target-observed)).sum())
                            if kl<-1e-9:raise ValueError('Invalid forward KL')
                            state_error={key:float((state-references[j][key]).norm()/references[j][key].norm().clamp_min(1e-12))
                                         for key,state in capture.current.items()}
                            expected_keys=(list(capture.current) if isinstance(hook,SixStateClamp) else
                                [f'{hook.site}/output'] if isinstance(hook,SingleStateClamp) and kinds else [])
                            if any(not torch.equal(capture.current[key],references[j][key]) for key in expected_keys):
                                raise ValueError('Teacher state replacement was not exact at its designated site')
                            rows.append({'prediction_index':j,'regions':case['regions'][j],'teacher_forward_KL':max(0.,kl),
                                'demonstration_token_NLL':float(-observed[case['teacher_ids'][j]]),
                                'teacher_greedy_agreement':int(logits.argmax()==teacher_logits[j].argmax()),
                                'demonstration_token_greedy_match':int(logits.argmax()==case['teacher_ids'][j]),
                                'greedy_token_id':int(logits.argmax()),'teacher_greedy_id':int(teacher_logits[j].argmax()),
                                'state_relative_L2':state_error,'exact_clamp_sites_checked':expected_keys})
                        del value,ids,mask,logits
                finally:
                    capture.close();del past;gc.collect();torch.cuda.empty_cache()
                return rows
            guard(cid+':teacher',True);sequence(capture_teacher=True)
            teacher_rows=[]
            for j,logits in enumerate(teacher_logits):
                teacher_rows.append({'regions':case['regions'][j],'teacher_forward_KL':0.,
                    'demonstration_token_NLL':float(-logits.double().log_softmax(-1)[case['teacher_ids'][j]]),
                    'teacher_greedy_agreement':1,'demonstration_token_greedy_match':int(logits.argmax()==case['teacher_ids'][j])})
            write_json(folder/'TEACHER_REFERENCE.json',{'case_id':cid,'metrics':summarize_steps(teacher_rows),
                'same_literal_training_prefix':True,'state_capture_sites':cfg['sites'],'teacher_cache_copied':False})
            with base_mode(model):
                # Literal zero delivery must match unmodified base before any comparison.
                raw=sequence(check_limit=4)
                zero=ResidualSites(layers,None,cfg['sites'],'two_site','zero',audit_state=False);zero.enabled=True
                try:wrapped=sequence(hook=zero,check_limit=4)
                finally:zero.close()
                fields=['teacher_forward_KL','demonstration_token_NLL','greedy_token_id']
                if any(a['greedy_token_id']!=b['greedy_token_id'] or
                       any(abs(a[k]-b[k])>1e-6 for k in fields[:2]) for a,b in zip(raw,wrapped)):
                    raise ValueError('Native zero-control logits or scores differ')
                write_json(folder/'ZERO_CONTROL.json',{'checked_tokens':4,'greedy_IDs_exact':True,
                    'score_absolute_tolerance':1e-6,'maximum_score_difference':max(abs(a[k]-b[k]) for a,b in zip(raw,wrapped) for k in fields[:2])})
                for mode in cfg['conditions']:
                    guard(cid+':'+mode,True);hook=None;maps=None
                    if mode.startswith('ridge'):
                        site=int(mode[5:]);hook=SelectedPositionHook(layers[site],DeltaPredictor(arrays[site],2,method='ridge',scale=1.))
                    elif mode.startswith('oracle') and mode!='oracle_six':
                        site=int(mode[6:]);hook=SingleStateClamp(layers[site],site,references)
                    elif mode=='oracle_six':hook=SixStateClamp(layers,cfg['sites'],references)
                    elif mode in ['six_initializer','six_final96']:
                        maps=torch.nn.ModuleDict({f'{site}/{kind}':LowRankMap(5120,32) for site in cfg['sites'] for kind in ['attention','output']})
                        maps.load_state_dict(initial_state if mode=='six_initializer' else final_state,strict=True)
                        maps=maps.cuda().eval().requires_grad_(False)
                        hook=ResidualSites(layers,None,cfg['sites'],'two_site','predicted',maps,audit_state=False);hook.enabled=True
                    try:rows=sequence(hook)
                    finally:
                        if hook:hook.close()
                    report={'case_id':cid,'family':case['family'],'mode':mode,'metrics':summarize_steps(rows),'steps':rows,
                            'teacher_forced_training_diagnostic':True,'workflow_executed':False,'model_forwards':len(rows)}
                    write_json(folder/(mode+'.json'),report);completed.append({k:report[k] for k in ['case_id','family','mode','metrics']})
                    write_json(out/'PROGRESS.json',{'utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                        'completed_condition_cases':len(completed),'planned_condition_cases':len(cases)*len(cfg['conditions']),
                        'last_case':cid,'last_mode':mode,'model_forwards':steps,'elapsed_seconds':time.monotonic()-started})
                    hook=maps=None;gc.collect();torch.cuda.empty_cache()
            # Preserve the diagnostic reference separately; it is not a deployable controller.
            temp=Path(tempfile.mkdtemp(prefix='r3_reference_'))
            states={key:torch.cat([r[key] for r in references],dim=0).contiguous() for key in references[0]}
            save_file(states,str(temp/'teacher_states.safetensors'));np.save(temp/'teacher_logits.npy',torch.stack(teacher_logits).numpy())
            for name in ['teacher_states.safetensors','teacher_logits.npy']:
                dest=folder/name;shutil.copyfile(temp/name,dest)
                if sha256(dest)!=sha256(temp/name):raise ValueError('Reference export readback differs')
            del teacher_logits,references,states;gc.collect();torch.cuda.empty_cache()
        guard('before_final_fingerprint',True);after=digest_model(model);write_json(out/'MODEL_AFTER.json',after)
        if after!=before:raise ValueError('Frozen model changed during diagnostic')
        write_json(out/'SUMMARY.json',{'format':'research3-steering-diagnostic-summary-v1','results':completed,
            'training_cases':3,'workflow_successes_measured':False,'teacher_cache_copied':False,
            'interpretation':'Matched-prefix probabilities and state-replacement controls only; not new heldout evidence or successful episodes.'})
        result={'status':'TRAINING_STATE_DIAGNOSTIC_COMPLETE','condition_cases':len(completed),'model_before_after_exact':True,
                'new_reference_calls':0,'optimizer_updates':0,'model_forwards':steps,'cases':cfg['case_ids'],
                'gate_decisions_fixed':True,'validation_or_reserved_holdout_opened':False,'automatic_retry':False}
        write_json(out/'COMPLETE.json',result);return result
    except Exception as error:
        write_json(out/'FAULT.json',{'type':type(error).__name__,'message':str(error)[:400],
            'completed_condition_cases':len(completed),'model_forwards':steps,'automatic_retry':False})
        raise
    finally:
        if hook is not None:hook.close()
        hook=maps=layers=sequence=model=None
        gc.collect();torch.cuda.empty_cache()
        write_json(out/'CLEANUP.json',{'utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'GPU_allocated_bytes':torch.cuda.memory_allocated(),'GPU_reserved_bytes':torch.cuda.memory_reserved()})


def summarize_saved(root):
    root=Path(root);cfg=json.loads((root/'FROZEN_PROTOCOL.json').read_bytes())
    complete=json.loads((root/'COMPLETE.json').read_bytes())
    if complete['status']!='TRAINING_STATE_DIAGNOSTIC_COMPLETE' or not complete['model_before_after_exact']:
        raise ValueError('The frozen diagnostic did not finish')
    if json.loads((root/'MODEL_BEFORE.json').read_bytes())!=json.loads((root/'MODEL_AFTER.json').read_bytes()):
        raise ValueError('Recorded model fingerprints differ')
    results=[]
    for cid in cfg['case_ids']:
        modes={}
        for mode in cfg['conditions']:
            raw=json.loads((root/'cases'/cid/(mode+'.json')).read_bytes())
            if raw['case_id']!=cid or raw['mode']!=mode or raw['metrics']!=summarize_steps(raw['steps']):
                raise ValueError('Independent diagnostic token counts differ')
            modes[mode]=raw['metrics']
        results.append({'case_id':cid,'metrics':modes,'paired':{
            learned:{region:{'learned_KL':modes[learned][region]['teacher_forward_KL'],
                'oracle_KL':modes[oracle][region]['teacher_forward_KL'],
                'learned_NLL':modes[learned][region]['demonstration_token_NLL'],
                'oracle_NLL':modes[oracle][region]['demonstration_token_NLL']}
                for region in ['all','function_name','parameter_value']}
            for learned,oracle in [('ridge31','oracle31'),('ridge47','oracle47'),('ridge62','oracle62'),('six_final96','oracle_six')]}})
    value={'status':'DIAGNOSTIC_SCORES_INDEPENDENTLY_VERIFIED','training_cases':3,'results':results,
           'no_workflow_success_claim':True,'no_heldout_generalization_claim':True,'new_model_forwards':0}
    return value


def report(admission_path):
    a=json.loads(Path(admission_path).read_bytes());root=Path(a['output'])
    value=summarize_saved(root);write_json(root/'COMPARISON.json',value);return value


def submit(job_id,packet,packet_sha256,initializer,initializer_sha256):
    from src.verify import source_digest
    from src.persist_data import upload_verified
    from src.runtime import runtime_task,launcher,bind_launcher
    dirty=subprocess.run(['git','status','--porcelain','--','src','config','resources'],capture_output=True,text=True,check=True)
    if dirty.stdout:raise ValueError('Commit the frozen protocol and source before submission')
    settings=Settings.load();w=workspace_client(settings);task=runtime_task(w,job_id);cfg=configuration()
    if task['task_key']!='diagnose_frozen_steering':raise ValueError('Wrong diagnostic job')
    now=dt.datetime.now(dt.timezone.utc);key=uuid.uuid4().hex
    output=settings.volume_root+'/runs/steering_diagnostic/'+now.strftime('%Y%m%dT%H%M%SZ')+'_'+key[:8]
    base=settings.volume_root+'/models/Qwen3.8-27B/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
    a={'format':'research3-steering-diagnostic-admission-v1','phase':'diagnostic','submitted_utc':now.isoformat(),
       'absolute_end_utc':(now+dt.timedelta(seconds=cfg['bounds']['whole_seconds'])).isoformat(),
       'output':output,'packet':packet,'packet_sha256':packet_sha256,'initializer':initializer,'initializer_sha256':initializer_sha256,
       'model_path':base,'adapter_directory':settings.volume_root+'/models/Qwen3.8-27B-preservation-A/8ae28fb51107a31419bcb744d9f47d80b681781a114f8bc9491751410b4b87aa',
       'comparison_inputs':settings.volume_root+'/combined_validation/comparison_inputs/v1/a8684b4b30d5824a24f9b846b5b6a36cfff09344ad1cacb35afc6b211249df31',
       'protocol_sha256':sha256(ROOT/'config/steering_diagnostic.yaml'),'source_digest':source_digest(),
       'Free_Edition_only':True,'automatic_retries':0,'execution_transport':'AI_RUNTIME_TASK'}
    owner=w.current_user.me().user_name
    if not owner or '/' in owner or chr(92) in owner:raise ValueError('Invalid launch owner')
    path='/Workspace/Users/'+owner+'/.research3-launchers/'+key+'/diagnostic.sh';admission=output+'_ADMISSION.json'
    source=launcher(settings,admission,'diagnostic')
    a['runtime_launcher']={'path':path,'bytes':len(source),'sha256':hashlib.sha256(source).hexdigest()}
    upload_verified(w,admission,(json.dumps(a,indent=2)+'\n').encode())
    local=settings.output_root/('steering_diagnostic_'+key);write_json(local/'ADMISSION.json',a)
    if bind_launcher(w,job_id,task,path,source)!=a['runtime_launcher']:raise ValueError('Launcher readback differs')
    submitted=w.jobs.run_now(job_id=job_id,idempotency_token=key,job_parameters={'catalog':settings.catalog,'schema':settings.schema,
        'volume':settings.volume,'code_path':settings.code_path,'admission':admission})
    result={'run_id':submitted.run_id,'job_id':job_id,'admission':admission,'output':output,'absolute_end_utc':a['absolute_end_utc']}
    write_json(local/'SUBMITTED.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','run','submit','report','replay'])
    for name in ['root','training-root','gate-predictions','output','admission','packet','packet-sha256','initializer','initializer-sha256']:p.add_argument('--'+name)
    p.add_argument('--job-id',type=int);a=p.parse_args()
    if a.stage=='replay':
        if not a.root or not a.output or Path(a.output).exists():p.error('Replay requires the saved root and a new output file')
        result=summarize_saved(a.root);write_json(a.output,result)
        print(json.dumps({'status':result['status'],'training_cases':3,'new_model_forwards':0}));return
    result=prepare(a.training_root,a.gate_predictions,a.output) if a.stage=='prepare' else run(a.admission) if a.stage=='run' else report(a.admission) if a.stage=='report' else submit(a.job_id,a.packet,a.packet_sha256,a.initializer,a.initializer_sha256)
    print(json.dumps(result))


if __name__=='__main__':main()
