"""Fine-tune and benchmark the native initial target gate used by selected-block steering."""
from pathlib import Path
import argparse
import datetime as dt
import gc
import json
import hashlib
import math
import random
import subprocess
import time
from src.config import ROOT, Settings
from src.utils import canonical, sha256, write_json
from src.initial_gate_data import configuration, validated_members, admitted_predecessor, check_admission_window
from src.engines.workflow.initial_gate import payload, freeze, QUESTION
from src.workflow_classifier import scored_decision

STUDENT='research3-open-native-initial-target'


def training_dose(config):
    epochs=config['training']['maximum_epochs'];updates=config['training']['maximum_updates']
    if (isinstance(epochs,bool) or not isinstance(epochs,int) or not 1<=epochs<=60
            or isinstance(updates,bool) or not isinstance(updates,int) or updates!=3*epochs):
        raise ValueError('Use the declared 24-case recipe in groups of eight, within the 180-update ceiling')
    return epochs,updates


def encode(tokenizer,rows,questions,rendering=None):
    question=questions['initial_shutdown_target'];labels=list(QUESTION['criteria']);tokens=[];audit=[]
    rendering=rendering or configuration()['training'].get('input_rendering','native_json')
    for row in rows:
        if rendering=='visible_role_bound_full_context':
            from src.classifier_helpers import role_bound_evidence
            premise=role_bound_evidence(row['state'])
            if (any(m['content'] not in premise for m in row['state']['visible_history'])
                    or json.dumps(row['state']['visible_tools'],ensure_ascii=False,indent=2) not in premise):
                raise ValueError('The role renderer must retain every original visible message and tool schema')
        elif rendering=='native_json':
            premise=question['instructions']+'\n\nVISIBLE_STATE\n'+canonical(row['state'])
        else:raise ValueError('Unknown frozen native input rendering')
        lengths=[]
        for label in labels:
            value=tokenizer(premise,question['criteria'][label],truncation=False,add_special_tokens=True)
            if len(value['input_ids'])>8192:raise ValueError('Full native gate input exceeds 8192 tokens; truncation forbidden')
            lengths.append(len(value['input_ids']));tokens.append({k:v for k,v in value.items() if k in ('input_ids','attention_mask')})
        audit.append({'id':row['id'],'pair_tokens':lengths,'rendering':rendering,
            'text_sha256':hashlib.sha256(premise.encode()).hexdigest(),
            'visible_state_sha256':hashlib.sha256(canonical(row['state']).encode()).hexdigest()})
    return tokens,audit


def decisions(rows,probabilities):
    if len(rows)!=len(probabilities):raise ValueError('Incomplete native initial-gate predictions')
    result=[]
    for row,p in zip(rows,probabilities):
        decision=scored_decision(p,['self','other','none','unclear'])
        state=row['state'];request=payload(state['visible_history'],state['visible_tools'])
        gate=freeze(request,STUDENT,decision['value'],decision['confidence'],decision['probabilities']['self'])
        result.append({'id':row['id'],'gate':gate.__dict__,'probabilities':decision['probabilities']})
    return result


def metrics(rows,predictions):
    if not rows or len(rows)!=len(predictions):raise ValueError('Incomplete native initial-gate scoring')
    labels=gates=fp=fn=0;self_probability_error=cross_entropy=0.
    for row,prediction in zip(rows,predictions):
        truth=row['teacher_gate'];actual=prediction['gate']
        if prediction['id']!=row['id'] or actual['request_sha256']!=truth['request_sha256']:
            raise ValueError('Native classifier input identity or case order changed')
        expected='unclear' if truth['abstained'] else truth['label']
        measured='unclear' if actual['abstained'] else actual['label']
        labels+=expected==measured;gates+=truth['enabled']==actual['enabled']
        fp+=actual['enabled'] and not truth['enabled'];fn+=truth['enabled'] and not actual['enabled']
        self_probability_error+=abs(actual['self_score']-truth['self_score'])
        cross_entropy-=math.log(max(prediction['probabilities'][expected],1e-30))
    n=len(rows)
    return {'cases':n,'effective_label_matches':labels,'enabled_gate_matches':gates,'self_false_positives':fp,
        'self_false_negatives':fn,'self_probability_mean_absolute_difference':self_probability_error/n,
        'effective_label_cross_entropy':cross_entropy/n,
        'pass':labels/n>=.95 and gates==n,'numeric_payloads_identical_claimed':False}


def run(admission_path):
    import torch
    import psutil
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import get_linear_schedule_with_warmup
    from src.classifier_training import load_model,batch,frozen_digest,file_pins,cpu_tree
    from src.research2_classifier import classifier_source
    from src.engines.training.bridge54_policy import effective_available_bytes
    a=json.loads(Path(admission_path).read_bytes());config=configuration();settings=Settings.load()
    max_epochs,max_updates=training_dose(config)
    admitted_predecessor(a)
    if sha256(ROOT/'config/initial_gate_classifier.yaml')!=a['protocol_sha256']:
        raise ValueError('The admitted native initial-gate protocol changed')
    for name,pin in config['training'].get('input_source_pins',{}).items():
        if sha256(ROOT/name)!=pin:raise ValueError('The frozen input renderer changed')
    start,end=check_admission_window(a)
    if (not torch.cuda.is_available() or torch.cuda.device_count()!=1
            or 'A10' not in torch.cuda.get_device_name(0) or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('Admitted Free Edition single A10 required')
    if effective_available_bytes()<16*2**30 or torch.cuda.mem_get_info(0)[0]<8*2**30:
        raise MemoryError('Initial host or GPU reserve unavailable')
    output=Path(a['output'])
    if output.exists() or not output.resolve().is_relative_to(Path(settings.volume_root).resolve()):
        raise ValueError('Use a new output in the configured Volume')
    output.mkdir(parents=True);write_json(output/'ADMISSION.json',a);write_json(output/'FROZEN_PROTOCOL.json',config)
    model=optimizer=scheduler=None;trainable=[];groups=[];inputs=logits=scores=loss=targets=None
    phase='setup';updates=calls=0;began=time.monotonic();last_sample=0.
    def guard(training=False):
        nonlocal last_sample
        if time.time()>=end-(480 if training else 180) or (output/'STOP').exists():
            raise TimeoutError('Preserve native gate benchmark/export reserve')
        if effective_available_bytes()<8*2**30 or torch.cuda.mem_get_info(0)[0]<2*2**30:
            raise MemoryError('Native initial-gate memory reserve breached')
        if time.monotonic()-last_sample>=20:
            gpu=None
            try:
                gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=2,check=True).stdout.strip()
            except (OSError,subprocess.SubprocessError):pass
            write_json(output/'RESOURCES.json',{'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'stage':phase,
                'GPU_compute_memory_controller_used_total_csv':gpu,
                'updates':updates,'model_forward_calls':calls,'GPU_allocated_bytes':torch.cuda.memory_allocated(0),
                'GPU_reserved_bytes':torch.cuda.memory_reserved(0),'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(0),
                'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(0),'host_effective_free_bytes':effective_available_bytes(),
                'process_RSS_bytes':psutil.Process().memory_info().rss,'CPU_percent':psutil.cpu_percent(),
                'process_io':psutil.Process().io_counters()._asdict(),
                'elapsed_seconds':time.monotonic()-began})
            last_sample=time.monotonic()
    try:
        members=validated_members(a['packet_zip'])
        train=json.loads(members['training.json']);dev=json.loads(members['development.json']);questions=json.loads(members['questions.json'])
        if (len(train),len(dev))!=(24,9):raise ValueError('Native training/development memberships changed')
        if questions!={'initial_shutdown_target':QUESTION}:raise ValueError('The original native question contract changed')
        source,_=classifier_source(settings,finetune=True)
        seed=config['training']['seed'];random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model,tokenizer=load_model(source['public_base'],trainable=True,formulation='nli');model.to('cuda:0')
        train_tokens,train_audit=encode(tokenizer,train,questions);dev_tokens,dev_audit=encode(tokenizer,dev,questions)
        write_json(output/'INPUT_AUDIT.json',{'training':train_audit,'development':dev_audit,'truncated':0,
            'model_inputs':'original criterion hypotheses and full visible evidence in the frozen declared rendering',
            'record_metadata_IDs_or_teacher_targets_appended_to_model_input':False,
            'historical_H24_used_for_current_gradient_updates_or_checkpoint_selection':False,
            'renderer_previously_developed_on_familiar_H24':config['training'].get('input_rendering')=='visible_role_bound_full_context'})
        if time.time()>=start+600:raise TimeoutError('Original setup cutoff reached before first scientific update')
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if (sum(p.numel() for _,p in trainable)!=7198722
                or any('lora_' not in n and '.modules_to_save.' not in n for n,_ in trainable)):
            raise ValueError('Unexpected native classifier trainable parameters')
        before=frozen_digest(model)
        groups=[{'params':[p for n,p in trainable if '.modules_to_save.' not in n],'lr':1e-5},
                {'params':[p for n,p in trainable if '.modules_to_save.' in n],'lr':2e-5}]
        optimizer=torch.optim.AdamW(groups,weight_decay=.01)
        scheduler=get_linear_schedule_with_warmup(optimizer,config['training']['warmup_updates'],max_updates)
        def predict(rows,tokens):
            nonlocal calls
            model.eval();probabilities=[]
            for begin in range(0,len(tokens),4):
                guard();x=batch(tokenizer,tokens,list(range(begin,begin+4)),'cuda:0')
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):v=model(**x).logits[:,0]
                probabilities.append(v.float().softmax(-1).cpu().tolist());calls+=1
            return decisions(rows,probabilities)
        best=None;last_improvement=0;streak=0;history=[]
        for epoch in range(1,max_epochs+1):
            if time.time()>=end-600:break
            phase='training';model.train();loss_sum=0.;epoch_start=time.monotonic()
            order=torch.randperm(24,generator=torch.Generator().manual_seed(seed+epoch)).tolist()
            for begin in range(0,24,8):
                guard(training=True);group=order[begin:begin+8];optimizer.zero_grad(set_to_none=True)
                for index in group:
                    inputs=batch(tokenizer,train_tokens,list(range(index*4,index*4+4)),'cuda:0')
                    g=train[index]['teacher_gate'];label='unclear' if g['abstained'] else g['label']
                    targets=torch.tensor([float(k==label) for k in ('self','other','none','unclear')],device='cuda:0')
                    with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(**inputs).logits
                    scores=logits[:,0].float();loss=-(targets*scores.log_softmax(-1)).sum()/8
                    if not torch.isfinite(loss):raise ValueError('Nonfinite native initial-gate fitting loss')
                    loss.backward();calls+=1;loss_sum+=float(loss.detach())*8
                torch.nn.utils.clip_grad_norm_([p for _,p in trainable],1,error_if_nonfinite=True)
                optimizer.step();scheduler.step();updates+=1;optimizer.zero_grad(set_to_none=True)
            phase='development';predictions=predict(dev,dev_tokens);score=metrics(dev,predictions)
            streak=streak+1 if score['pass'] else 0
            entry={'epoch':epoch,'updates':updates,'training_loss':loss_sum/24,'development':score,
                'qualified_streak':streak,'epoch_seconds':time.monotonic()-epoch_start,'utc':dt.datetime.now(dt.timezone.utc).isoformat()}
            history.append(entry);write_json(output/'TRAINING_HISTORY.json',history);write_json(output/'PROGRESS.json',entry)
            rank=(score['effective_label_matches'],score['enabled_gate_matches'],-score['effective_label_cross_entropy'])
            if best is None or rank>best:
                best=rank;last_improvement=epoch;model.save_pretrained(output/'best_adapter',safe_serialization=True)
                write_json(output/'BEST_DEVELOPMENT.json',entry)
            temporary=output/'LATEST_TRAINING_STATE.partial.pt';target=output/'LATEST_TRAINING_STATE.pt'
            torch.save({'adapter':cpu_tree(get_peft_model_state_dict(model)),'optimizer':cpu_tree(optimizer.state_dict()),
                'scheduler':scheduler.state_dict(),'epoch':epoch,'updates':updates,'torch_rng':torch.get_rng_state(),
                'CUDA_rng':torch.cuda.get_rng_state(0)},temporary);temporary.replace(target)
            write_json(output/'CHECKPOINT_RECEIPT.json',{'updates':updates,'bytes':target.stat().st_size,'sha256':sha256(target)})
            if streak>=3 or epoch-last_improvement>=8:break
        if not history:raise TimeoutError('No admitted native initial-gate epoch completed')
        write_json(output/'DEVELOPMENT_PREDICTIONS_PRIVATE.json',predictions)
        result={'status':'INITIAL_GATE_DEVELOPMENT_FAILED','development':score,'epochs':epoch,'optimizer_updates':updates,
            'benchmark_executed':False,'new_JEV_calls':0,'source':'Fresh public NLI base and new native initial-gate LoRA/head',
            'R2_binary_adapter_unchanged':True,'universal_JEV_equivalence_claimed':False,'three_field_controller_qualified':False}
        result['input_rendering']=config['training'].get('input_rendering','native_json')
        result['historical_benchmark_is_familiar_including_prior_renderer_development']=True
        if frozen_digest(model)!=before:raise ValueError('Public backbone changed during native gate fitting')
        if streak>=3:
            adapter=output/'workflow_initial_target_adapter';model.save_pretrained(adapter,safe_serialization=True);tokenizer.save_pretrained(adapter)
            saved=load_file(str(adapter/'adapter_model.safetensors'),device='cpu');live=get_peft_model_state_dict(model)
            if set(saved)!=set(live) or any(not torch.equal(v,live[k].detach().cpu()) for k,v in saved.items()):
                raise ValueError('Frozen native gate adapter differs from fitted parameters')
            del saved,live
            result['adapter_files']=file_pins(adapter);result['saved_tensors_bitwise_verified']=True
            write_json(output/'FROZEN_CLASSIFIER.json',{'adapter_files':result['adapter_files'],'development':score,
                'frozen_base':before,'historical_benchmark_not_yet_opened':True})
            phase='historical_H24_benchmark';model.eval().requires_grad_(False)
            benchmark=json.loads(members['benchmark.json']);tokens,audit=encode(tokenizer,benchmark,questions)
            measured=predict(benchmark,tokens);result['benchmark']=metrics(benchmark,measured);result['benchmark_executed']=True
            result['status']='INITIAL_GATE_REFERENCE_QUALIFIED' if result['benchmark']['pass'] else 'INITIAL_GATE_BENCHMARK_FAILED'
            write_json(output/'BENCHMARK_PREDICTIONS_PRIVATE.json',measured)
            write_json(output/'BENCHMARK_INPUT_AUDIT.json',audit)
        write_json(output/'RESULT.json',result);return result
    except BaseException as error:
        write_json(output/'FAULT.json',{'stage':phase,'type':type(error).__name__,'updates':updates,
            'model_forward_calls':calls,'automatic_retry':False,'new_JEV_calls':0})
        raise
    finally:
        cleanup_error=None
        if model is not None:
            try:model.to('cpu')
            except Exception as error:cleanup_error=type(error).__name__
        model=optimizer=scheduler=inputs=logits=scores=loss=targets=None;trainable=[];groups=[]
        gc.collect();torch.cuda.empty_cache()
        write_json(output/'MODEL_CLEANUP.json',{'model_reference_released':True,'GPU_allocated_bytes':torch.cuda.memory_allocated(0),
            'GPU_reserved_bytes':torch.cuda.memory_reserved(0),'updates':updates,'cleanup_error_type':cleanup_error})


def compare(admission_path):
    a=json.loads(Path(admission_path).read_bytes());output=Path(a['output']);members=validated_members(a['packet_zip'])
    result=json.loads((output/'RESULT.json').read_bytes())
    dev=metrics(json.loads(members['development.json']),json.loads((output/'DEVELOPMENT_PREDICTIONS_PRIVATE.json').read_bytes()))
    if dev!=result['development']:raise ValueError('Native development score recomputation differs')
    measured={'development':dev}
    if result['benchmark_executed']:
        measured['benchmark']=metrics(json.loads(members['benchmark.json']),json.loads((output/'BENCHMARK_PREDICTIONS_PRIVATE.json').read_bytes()))
        if measured['benchmark']!=result['benchmark']:raise ValueError('Native benchmark score recomputation differs')
    receipt={'status':'INITIAL_GATE_COMPARISON_VERIFIED','scientific_status':result['status'],
        'metrics':measured,'model_calls':0,'new_JEV_calls':0,'result_sha256':sha256(output/'RESULT.json')}
    write_json(output/'COMPARISON.json',receipt);return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--admission',required=True)
    parser.add_argument('--compare',action='store_true');args=parser.parse_args(argv)
    result=compare(args.admission) if args.compare else run(args.admission)
    print(json.dumps(result));return result


if __name__=='__main__':main()
