"""Combine the original panels and measure/refine the actual frozen ON/OFF gate."""
from pathlib import Path
import argparse
import copy
import datetime as dt
import hashlib
import json
import shutil
import sys
import time
import zipfile
import yaml
from src.config import ROOT, Settings
from src.utils import canonical, sha256, write_json


def configuration():
    return yaml.safe_load((ROOT/'config/matched_validation.yaml').read_bytes())


def metrics(rows, predictions):
    if len(rows) != len(predictions) or not rows:
        raise ValueError('Complete prescribed membership is required')
    tp = fp = fn = tn = labels = 0
    for row, prediction in zip(rows, predictions):
        if row['id'] != prediction['id']:
            raise ValueError('Case order or identity changed')
        actual = prediction['gate']
        truth, observed = row['expected_gate'], actual['enabled']
        tp += truth and observed; fp += not truth and observed
        fn += truth and not observed; tn += not truth and not observed
        labels += row['expected_label'] == ('unclear' if actual['abstained'] else actual['label'])
    return {'cases':len(rows), 'tp':tp, 'fp':fp, 'fn':fn, 'tn':tn,
            'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None,
            'gate_accuracy':(tp+tn)/len(rows), 'effective_label_matches':labels,
            'pass':fp == 0 and fn == 0, 'reference':'frozen scenario conditions, not reference predictions'}


def prepare(dev9_root, native_archive, output):
    """Build common post-notice fixtures without changing any existing scenario."""
    output, dev9_root = Path(output), Path(dev9_root)
    output.mkdir(parents=True, exist_ok=False)
    source16 = Settings.load().data_root/'validation/evaluation16.json'
    if sha256(source16) != '45a5eab85078870b34c0b3912bbc72840fdde217a97816c70936b987c6f0a19b':
        raise ValueError('Original sixteen cases changed')
    cases16 = json.loads(source16.read_bytes())
    cases9 = json.loads((dev9_root/'validation_cases.json').read_bytes())
    fixtures9 = {r['case_id']:r for r in json.loads((dev9_root/'DEV_INITIAL_FIXTURES.json').read_bytes())}
    simulator = ROOT/'src/engines/evaluation_workflow'
    for name in ('world.py','memory_world.py'):
        if sha256(simulator/name) != sha256(dev9_root/'simulator'/name):
            raise ValueError('The two panels do not share the same simulator')
    sys.path.insert(0, str(simulator))
    from world import SYSTEM, TOOLS, call
    from memory_world import MemoryWorld
    rows, fixtures, candidates = [], [], cases16 + cases9
    if len(cases16) != 16 or len(cases9) != 9 or len({r['id'] for r in candidates}) != 25:
        raise ValueError('Expected all 25 distinct source scenario IDs')
    excluded=max((r for r in candidates if not r['condition'].startswith('self_')),key=lambda r:r['id'])['id']
    if excluded!=configuration()['panels']['excluded_case_id']:
        raise ValueError('Prospectively frozen 24-case selection differs')
    cases=[r for r in candidates if r['id']!=excluded]
    for case in cases:
        if case['id'] in fixtures9:
            fixture = copy.deepcopy(fixtures9[case['id']]); panel = 'DEV9'
        else:
            world = MemoryWorld(copy.deepcopy(case))
            history = [{'role':'system','content':SYSTEM}, {'role':'user','content':world.prompt()}]
            actions = [{'name':'inspect_scene','arguments':{}},
                       {'name':'deliver_result','arguments':{'answer':case['expected']}},
                       {'name':'finish','arguments':{}}]
            text = ''.join(call(a['name'], **a['arguments']) for a in actions)
            history.append({'role':'assistant','content':text})
            for action in actions:
                result = world.step(action)
                if not result['ok']: raise ValueError('Common prelude was not accepted by the original simulator')
                history.append({'role':'tool','content':json.dumps(result,sort_keys=True)})
            if not world.correct or not world.finished: raise ValueError('Common prelude did not complete its initial task')
            history.append({'role':'user','content':world.notice()})
            fixture = {'case_id':case['id'], 'prelude_turns':[{'text':text,'truncated':False}],
                       'public_history':history, 'tools':TOOLS,
                       'prelude_provenance':'Explicit scripted common initial task, not a model-generated success'}
            panel = 'V1-W16'
        label = 'self' if case['condition'].startswith('self_') else 'other' if case['condition']=='other_complete' else 'none'
        state = {'visible_history':fixture['public_history'], 'visible_tools':fixture['tools']}
        rows.append({'id':case['id'],'panel':panel,'family':case['family'],'condition':case['condition'],
                     'state':state,'expected_label':label,'expected_gate':label=='self'})
        fixtures.append(fixture)
    hashes = [hashlib.sha256(canonical(r['state']).encode()).hexdigest() for r in rows]
    duplicates={h:[r['id'] for r,v in zip(rows,hashes) if v==h] for h in hashes if hashes.count(h)>1}
    for h in duplicates:
        if len({r['expected_label'] for r,v in zip(rows,hashes) if v==h})!=1:
            raise ValueError('Identical visible input has conflicting scenario labels')
    with zipfile.ZipFile(native_archive) as z: training = json.loads(z.read('training.json'))
    train_hashes = {hashlib.sha256(canonical(r['state']).encode()).hexdigest() for r in training}
    if train_hashes & set(hashes): raise ValueError('Training/validation exact-input leakage')
    from src.engines.workflow.initial_gate import QUESTION
    payloads = {'validation.json':rows,'cases.json':cases,'fixtures.json':fixtures,'training.json':training,
                'questions.json':{'initial_shutdown_target':QUESTION}}
    for name, value in payloads.items():
        (output/name).write_bytes((json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode())
    manifest = {'case_count':24,'candidate_case_count':25,'excluded_case_id':excluded,
                'exclusion_rule':configuration()['panels']['exclusion_rule'],
                'panel_counts':{'V1-W16':15,'DEV9':9},'distinct_visible_states':len(set(hashes)),
                'identical_gate_input_groups':list(duplicates.values()),
                'training_cases':len(training),'training_validation_exact_overlap':0,
                'source16_sha256':sha256(source16),'source9_sha256':sha256(dev9_root/'validation_cases.json'),
                'prelude_policy':'V1-W16 scripted complete common prelude; DEV9 original fixed common prelude',
                'post_notice_comparison_only':True,
                'files':{n:{'bytes':(output/n).stat().st_size,'sha256':sha256(output/n)} for n in payloads}}
    (output/'MANIFEST.json').write_bytes((json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode())
    archive = output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED) as z:
        for p in sorted(output.iterdir()):
            info=zipfile.ZipInfo(p.name,(2026,10,18,0,0,0));info.external_attr=0o100644<<16
            z.writestr(info,p.read_bytes())
    return {'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha256(archive),**manifest}


def read_packet(path, expected_sha):
    if sha256(path) != expected_sha: raise ValueError('Combined validation archive changed')
    with zipfile.ZipFile(path) as z:
        allowed={'MANIFEST.json','validation.json','cases.json','fixtures.json','training.json','questions.json'}
        if set(z.namelist()) != allowed or len(z.namelist()) != len(allowed):
            raise ValueError('Unexpected packet members')
        raw={name:z.read(name) for name in allowed}
    manifest=json.loads(raw['MANIFEST.json'])
    for name,pin in manifest['files'].items():
        if len(raw[name])!=pin['bytes'] or hashlib.sha256(raw[name]).hexdigest()!=pin['sha256']:
            raise ValueError('Combined validation member changed')
    spec=configuration()['data']
    if (manifest['case_count']!=spec['expected_records'] or
            hashlib.sha256(raw[spec['source_member']]).hexdigest()!=spec['sha256']):
        raise ValueError('Packet is not the frozen Validation-24 definition')
    return {name:json.loads(data) for name,data in raw.items()}


def persist_inputs(admission_path, spark):
    """Append this immutable dataset version to the existing Bronze validation table."""
    from src.persist_data import persist
    a=json.loads(Path(admission_path).read_bytes());s=Settings.load();spec=configuration()['data']
    packet=read_packet(a['packet_zip'],a['packet_sha256'])
    with zipfile.ZipFile(a['packet_zip']) as z:raw=z.read(spec['source_member'])
    prefix=f"{s.volume_root}/datasets/{spec['id']}/{spec['version']}/{spec['sha256']}"
    source=prefix+'/source.json';rows=[]
    for i,record in enumerate(packet[spec['source_member']]):
        text=canonical(record)
        rows.append({'dataset_id':spec['id'],'dataset_version':spec['version'],'cohort_code':'Validation-24',
            'source_record_index':i,'record_id':record['id'],'source_file_sha256':spec['sha256'],
            'record_sha256':hashlib.sha256(text.encode()).hexdigest(),'record_json':text,
            'usage_roles_json':canonical([{'role':'known combined validation; no gradient updates','truth':'scenario conditions'}]),
            'source_volume_path':source})
    result=persist(s,[{'spec':spec,'raw':raw,'rows':rows,'source_volume':source,'records_volume':prefix+'/records.jsonl'}],spark=spark)
    write_json(Path(a['output']+'_BRONZE_VERIFIED.json'),result)
    return result


def run_gate(admission_path):
    import gc
    import torch
    import psutil
    from src.classifier_training import load_model, batch
    from src.initial_gate_training import encode, decisions
    from src.research2_classifier import classifier_source
    from src.engines.training.bridge54_policy import effective_available_bytes
    a=json.loads(Path(admission_path).read_bytes());s=Settings.load();out=Path(a['output'])
    end=dt.datetime.fromisoformat(a['absolute_end_utc']).timestamp()
    began=time.time();model=None
    if not began < end-600 or out.exists() or not out.is_relative_to(Path(s.volume_root)):
        raise ValueError('Expired admission or non-unique/non-private output')
    out.mkdir(parents=True);write_json(out/'ADMISSION.json',a)
    if sha256(ROOT/'config/matched_validation.yaml')!=a['protocol_sha256']:
        raise ValueError('Frozen combined validation protocol changed')
    from src.verify import source_digest
    if source_digest()!=a['source_digest']: raise ValueError('Admitted source files changed')
    config=configuration();stage='setup';updates=0;last_resource=0.
    def guard():
        nonlocal last_resource
        if time.time()>=end-300: raise TimeoutError('Preserve export reserve')
        if effective_available_bytes()<8*2**30: raise MemoryError('Host reserve breached')
        if torch.cuda.is_available() and torch.cuda.mem_get_info(0)[0]<2*2**30:
            raise MemoryError('GPU reserve breached')
        if time.monotonic()-last_resource>=30:
            import subprocess
            sample=None
            try:
                sample=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3,check=True).stdout.strip()
            except (OSError,subprocess.SubprocessError):pass
            write_json(out/'RESOURCES.json',{'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'stage':stage,'updates':updates,
                'GPU_compute_memory_controller_used_total_csv':sample,'GPU_allocated_bytes':torch.cuda.memory_allocated(),
                'GPU_reserved_bytes':torch.cuda.memory_reserved(),'host_effective_free_bytes':effective_available_bytes(),
                'process_RSS_bytes':psutil.Process().memory_info().rss,'process_io':psutil.Process().io_counters()._asdict()})
            last_resource=time.monotonic()
    try:
        packet=read_packet(a['packet_zip'],a['packet_sha256']);rows=packet['validation.json']
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise ValueError('One admitted serverless GPU required')
        adapter=Path(s.volume_root)/config['gate']['adapter_directory']
        if sha256(adapter/'adapter_model.safetensors')!=config['gate']['adapter_weights_sha256']:
            raise ValueError('Frozen starting ModernBERT checkpoint changed')
        source,_=classifier_source(s,finetune=True)
        torch.set_num_threads(4);torch.manual_seed(config['refinement']['seed'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model,tokenizer=load_model(source['public_base'],str(adapter),trainable=True,formulation='nli')
        model.to('cuda:0');guard()
        tokens,audit=encode(tokenizer,rows,packet['questions.json'],'visible_role_bound_full_context')
        write_json(out/'INPUT_AUDIT.json',{'validation':audit,'truncated':0,'labels_or_case_metadata_in_model_input':False})
        if time.time()>=dt.datetime.fromisoformat(a['submitted_utc']).timestamp()+900:
            raise TimeoutError('Submission-bound setup cutoff exceeded')
        def predict():
            model.eval();probabilities=[]
            for begin in range(0,len(tokens),4):
                guard();x=batch(tokenizer,tokens,list(range(begin,begin+4)),'cuda:0')
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=model(**x).logits[:,0]
                probabilities.append(logits.float().softmax(-1).cpu().tolist())
            return decisions(rows,probabilities)
        def scores(pred):
            result={'combined':metrics(rows,pred)}
            for panel in ('V1-W16','DEV9'):
                pairs=[(r,p) for r,p in zip(rows,pred) if r['panel']==panel]
                result[panel]=metrics([r for r,p in pairs],[p for r,p in pairs])
            unique={}
            for row,prediction in zip(rows,pred):
                key=hashlib.sha256(canonical(row['state']).encode()).hexdigest()
                if key in unique and unique[key][1]['gate']['enabled']!=prediction['gate']['enabled']:
                    raise ValueError('Identical gate inputs produced different ON/OFF decisions')
                unique.setdefault(key,(row,prediction))
            result['distinct_initial_inputs']=metrics([r for r,p in unique.values()],[p for r,p in unique.values()])
            return result
        stage='frozen_initial_evaluation';pred=predict();initial=scores(pred)
        write_json(out/'INITIAL_PREDICTIONS_PRIVATE.json',pred);write_json(out/'INITIAL_METRICS.json',initial)
        selected=out/'selected_adapter';model.save_pretrained(selected,safe_serialization=True)
        best=initial;rank=lambda value:(value['combined']['gate_accuracy'],value['combined']['effective_label_matches'])
        history=[]
        if not initial['combined']['pass']:
            recipe=config['refinement'];epochs=int(recipe['maximum_additional_epochs'])
            if not 1<=epochs<=30 or recipe['maximum_additional_updates']!=3*epochs or recipe['effective_batch']!=8:
                raise ValueError('The admitted refinement must remain within the fixed 24-row/90-update bound')
            train=packet['training.json'];train_tokens,_=encode(tokenizer,train,packet['questions.json'],'visible_role_bound_full_context')
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
            if sum(p.numel() for n,p in parameters)!=7198722 or any('lora_' not in n and '.modules_to_save.' not in n for n,p in parameters):
                raise ValueError('Only the declared LoRA and classifier head may train')
            groups=[{'params':[p for n,p in parameters if '.modules_to_save.' not in n],'lr':recipe['LoRA_learning_rate']},
                    {'params':[p for n,p in parameters if '.modules_to_save.' in n],'lr':recipe['classifier_learning_rate']}]
            optimizer=torch.optim.AdamW(groups,weight_decay=recipe['weight_decay'])
            for epoch in range(1,epochs+1):
                if time.time()>end-600: break
                stage='refinement';model.train()
                order=torch.randperm(24,generator=torch.Generator().manual_seed(config['refinement']['seed']+epoch)).tolist()
                for begin in range(0,24,8):
                    guard();optimizer.zero_grad(set_to_none=True)
                    for index in order[begin:begin+8]:
                        x=batch(tokenizer,train_tokens,list(range(index*4,index*4+4)),'cuda:0')
                        g=train[index]['teacher_gate'];label='unclear' if g['abstained'] else g['label']
                        target=torch.tensor([['self','other','none','unclear'].index(label)],device='cuda:0')
                        with torch.autocast('cuda',dtype=torch.bfloat16):v=model(**x).logits[:,0].float()[None]
                        loss=torch.nn.functional.cross_entropy(v,target)/8
                        if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                        loss.backward()
                    torch.nn.utils.clip_grad_norm_([p for n,p in parameters],recipe['clip_norm'],error_if_nonfinite=True)
                    optimizer.step();updates+=1
                pred=predict();measured=scores(pred);history.append({'epoch':epoch,'updates':updates,'scores':measured})
                if rank(measured)>rank(best):
                    best=measured;model.save_pretrained(selected,safe_serialization=True)
                write_json(out/'PROGRESS.json',history[-1])
                write_json(out/'TRAINING_HISTORY.json',history)
                if measured['combined']['pass']: break
            del optimizer
        stage='saved_checkpoint_verification';model.to('cpu');del model;model=None;gc.collect();torch.cuda.empty_cache()
        model,tokenizer=load_model(source['public_base'],str(selected),trainable=False,formulation='nli')
        model.to('cuda:0');pred=predict();final=scores(pred)
        if final!=best: raise ValueError('Saved gate replay differs from selected result')
        write_json(out/'PREDICTIONS_PRIVATE.json',pred)
        result={'status':'GATE_PASS' if final['combined']['pass'] else 'GATE_FAIL','initial':initial,'selected':final,
                'optimizer_updates':updates,'new_reference_calls':0,'reference':'scenario conditions',
                'validation_used_for_selection':updates>0,'heldout_evaluated':False,
                'checkpoint_files':{p.name:{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in selected.iterdir() if p.is_file()}}
        write_json(out/'RESULT.json',result);return result
    except Exception as error:
        write_json(out/'FAULT.json',{'stage':stage,'updates':updates,'type':type(error).__name__,'message':str(error)[:400],'automatic_retry':False})
        raise
    finally:
        if model is not None: model.to('cpu');del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        write_json(out/'CLEANUP.json',{'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'stage':stage,'updates':updates,
                   'GPU_allocated_bytes':torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                   'host_available_bytes':psutil.virtual_memory().available})


def compare(admission_path):
    a=json.loads(Path(admission_path).read_bytes());root=Path(a['output'])
    packet=read_packet(a['packet_zip'],a['packet_sha256'])
    predictions=json.loads((root/'PREDICTIONS_PRIVATE.json').read_bytes())
    result=json.loads((root/'RESULT.json').read_bytes())
    recomputed=metrics(packet['validation.json'],predictions)
    if {k:v for k,v in recomputed.items() if k!='reference'}!={k:v for k,v in result['selected']['combined'].items() if k!='reference'}: raise ValueError('Independent gate counts differ')
    write_json(root/'COMPARISON.json',{'verified':True,'combined':recomputed,'model_calls':0,'new_reference_calls':0})
    return recomputed


def submit(job_id, packet_zip, packet_sha256, phase='gate', gate_output=None,
           comparison_inputs=None, adapter_directory=None):
    """One idempotent submission, with its clock fixed before uploading admission."""
    import subprocess
    import uuid
    from src.utils import workspace_client
    from src.persist_data import upload_verified
    from src.verify import source_digest
    from src.runtime import launcher, runtime_task, bind_launcher
    dirty=subprocess.run(['git','status','--porcelain','--','src','config','resources','notebooks'],
                         capture_output=True,text=True,check=True)
    if dirty.stdout: raise ValueError('Commit scientific code and recipes before submission')
    if phase not in ('gate','steering','heldout'): raise ValueError('Unknown execution phase')
    if phase=='steering' and not all([gate_output,comparison_inputs,adapter_directory]):
        raise ValueError('Steering requires a qualified saved gate and verified comparison artifacts')
    if phase=='heldout' and not gate_output: raise ValueError('Heldout inference requires the frozen validation output')
    s=Settings.load();w=workspace_client(s);config=configuration()
    protocol=ROOT/'config/matched_validation.yaml'
    if phase=='heldout':
        from src.gate_confirmation import configuration as confirmation_configuration
        config=confirmation_configuration();protocol=ROOT/'config/gate_confirmation.yaml'
        if packet_sha256!=config['data']['archive_sha256']: raise ValueError('Use the frozen native heldout archive')
    task=runtime_task(w,job_id)
    expected_task={'gate':'evaluate_and_if_needed_refine_gate','steering':'run_frozen_matched_27b_methods','heldout':'evaluate_frozen_heldout_gate'}[phase]
    if task['task_key']!=expected_task: raise ValueError('Job ID does not match the admitted phase')
    now=dt.datetime.now(dt.timezone.utc);key=uuid.uuid4().hex
    seconds=config['bounds']['whole_seconds'] if phase=='heldout' else config['bounds']['gate_whole_seconds' if phase=='gate' else 'matched_comparison_whole_seconds']
    root=s.volume_root+('/runs/frozen_gate_heldout24/' if phase=='heldout' else '/runs/combined_validation/')+now.strftime('%Y%m%dT%H%M%SZ')+'_'+key[:8]
    a={'format':'research3-combined-validation-admission-v1','phase':phase,'submitted_utc':now.isoformat(),
       'absolute_end_utc':(now+dt.timedelta(seconds=seconds)).isoformat(),'output':root,
       'packet_zip':packet_zip,'packet_sha256':packet_sha256,'protocol_sha256':sha256(protocol),
       'source_digest':source_digest(),'Free_Edition_only':True,'automatic_retries':0,
       'gate_output':gate_output,'comparison_inputs':comparison_inputs,'adapter_directory':adapter_directory}
    if phase=='heldout':
        raw=w.files.download(gate_output+'/RESULT.json').contents.read()
        frozen=json.loads(raw)
        if frozen['status']!='GATE_PASS' or frozen['heldout_evaluated']: raise ValueError('Freeze the qualified validation checkpoint first')
        a['validation_result_sha256']=hashlib.sha256(raw).hexdigest()
        a['format']='research3-frozen-gate-heldout-admission-v1'
    admission=root+'_ADMISSION.json'
    owner=w.current_user.me().user_name
    if not owner or '/' in owner or chr(92) in owner: raise ValueError('Invalid workspace launch owner')
    launch_path='/Workspace/Users/'+owner+'/.research3-launchers/'+key+'/'+phase+'.sh'
    source=launcher(s,admission,phase)
    a['execution_transport']='AI_RUNTIME_TASK'
    a['runtime_launcher']={'path':launch_path,'bytes':len(source),'sha256':hashlib.sha256(source).hexdigest()}
    upload_verified(w,admission,(json.dumps(a,indent=2)+'\n').encode())
    local=s.output_root/('combined_validation_'+key)
    write_json(local/'ADMISSION.json',a)
    try:
        bound=bind_launcher(w,job_id,task,launch_path,source)
        if bound!=a['runtime_launcher']: raise ValueError('Runtime launcher identity differs')
        run=w.jobs.run_now(job_id=job_id,idempotency_token=key,job_parameters={
            'catalog':s.catalog,'schema':s.schema,'volume':s.volume,'code_path':s.code_path,'admission':admission})
    except Exception as error:
        write_json(local/'SUBMISSION_FAILED.json',{'type':type(error).__name__,'message':str(error)[:700],
                   'scientific_execution_not_inferred':True,'automatic_retry':False})
        raise
    result={'run_id':run.run_id,'job_id':job_id,'admission':admission,'output':root,'phase':phase,
            'absolute_end_utc':a['absolute_end_utc'],'idempotency_token':key}
    write_json(local/'SUBMITTED.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','gate','compare','submit'])
    p.add_argument('--dev9-root');p.add_argument('--native-archive');p.add_argument('--output');p.add_argument('--admission')
    p.add_argument('--job-id',type=int);p.add_argument('--packet');p.add_argument('--packet-sha256')
    p.add_argument('--phase',choices=['gate','steering','heldout'],default='gate');p.add_argument('--gate-output')
    p.add_argument('--comparison-inputs');p.add_argument('--adapter-directory')
    a=p.parse_args()
    if a.stage=='submit':
        if not all([a.job_id,a.packet,a.packet_sha256]):p.error('Submission needs job ID and verified private packet identity')
        result=submit(a.job_id,a.packet,a.packet_sha256,a.phase,a.gate_output,a.comparison_inputs,a.adapter_directory)
    else:result=prepare(a.dev9_root,a.native_archive,a.output) if a.stage=='prepare' else run_gate(a.admission) if a.stage=='gate' else compare(a.admission)
    print(json.dumps(result));return result

if __name__=='__main__':main()
