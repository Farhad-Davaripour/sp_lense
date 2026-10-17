"""One bounded joint refinement on cached workflow and original binary training labels."""
from pathlib import Path
import argparse
import datetime as dt
import gc
import hashlib
import json
import math
import random
import subprocess
import tempfile
import time
import zipfile
import yaml
from src.config import ROOT, Settings
from src.utils import canonical, sha256, write_json
from src.workflow_classifier import encode_workflow, scored_decision, summarize, summarize_initial
from src.research2_classifier import classifier_source, pairs, agreement, qualified
from src.engines.workflow.schema import QUESTIONS


def training_rows(rows, validation, initial_validation):
    """Exclude exact visible validation states, including the separate initial-gate panel."""
    blocked = {canonical(r['state']) for r in validation + initial_validation}
    return [r for r in rows if canonical(r['state']) not in blocked]


def run(admission_path):
    import torch
    import psutil
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import get_linear_schedule_with_warmup
    from src.classifier_training import load_model, batch, frozen_digest, cpu_tree, file_pins
    from src.research2_data import read_dataset
    from src.engines.training.bridge54_policy import effective_available_bytes

    settings = Settings.load()
    a = json.loads(Path(admission_path).read_bytes())
    protocol = yaml.safe_load((ROOT/'config/workflow_classifier_joint.yaml').read_bytes())
    packet_protocol = yaml.safe_load((ROOT/'config/workflow_classifier_transfer.yaml').read_bytes())
    start = dt.datetime.fromisoformat(a['submission_utc']).timestamp()
    end = min(start+protocol['bounds']['maximum_seconds'], dt.datetime.fromisoformat(a['absolute_end_utc']).timestamp())
    if not start <= time.time() < start+900 or end-time.time() < 1500:
        raise TimeoutError('Original joint-refinement setup/window expired')
    output = Path(a['output'])
    if not output.resolve().is_relative_to(Path(settings.volume_root).resolve()) or output.exists():
        raise ValueError('Use a new output directory in the configured Volume')
    packet = Path(a['packet_zip'])
    if sha256(packet) != packet_protocol['dataset']['archive_sha256']:
        raise ValueError('Frozen workflow reference packet changed')
    source, adapter = classifier_source(settings,a['classifier_output'])
    if sha256(adapter/'adapter_model.safetensors') != protocol['starting_adapter_sha256']:
        raise ValueError('The prospective joint refinement has one specified starting checkpoint')
    if (not torch.cuda.is_available() or torch.cuda.device_count()!=1
            or 'A10' not in torch.cuda.get_device_name(0) or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('The admitted Free Edition single A10 is required')
    if effective_available_bytes()<16*2**30:
        raise MemoryError('Initial host memory reserve unavailable')
    output.mkdir(parents=True)
    write_json(output/'ADMISSION.json',a)
    write_json(output/'FROZEN_PROTOCOL.json',protocol)
    phase='setup';updates=calls=0;model=optimizer=scheduler=None
    trainable=[];groups=[];inputs=logits=scores=loss=targets=None
    last_sample=0.;began=time.monotonic()
    def guard(training=False):
        nonlocal last_sample
        reserve=1200 if training else 300
        if time.time()>=end-reserve or (output/'STOP').exists():
            raise TimeoutError('Preserve the original joint-refinement evaluation/export reserve')
        if effective_available_bytes()<8*2**30 or torch.cuda.mem_get_info(0)[0]<2*2**30:
            raise MemoryError('Joint-refinement memory reserve breached')
        if time.monotonic()-last_sample>=20:
            gpu=None
            try:
                gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=2,check=True).stdout.strip()
            except (OSError,subprocess.SubprocessError):pass
            write_json(output/'RESOURCES.json',{'utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                'stage':phase,'updates':updates,'model_forward_calls':calls,'GPU_utilization_memory_used_total_csv':gpu,
                'GPU_allocated_bytes':torch.cuda.memory_allocated(0),'GPU_reserved_bytes':torch.cuda.memory_reserved(0),
                'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(0),'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(0),
                'host_effective_free_bytes':effective_available_bytes(),'process_RSS_bytes':psutil.Process().memory_info().rss,
                'CPU_percent':psutil.cpu_percent(),'process_io':psutil.Process().io_counters()._asdict(),
                'elapsed_seconds':time.monotonic()-began})
            last_sample=time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix='workflow-joint-') as directory:
            root=Path(directory)
            with zipfile.ZipFile(packet) as z:
                manifest=json.loads(z.read('MANIFEST.json'))
                if hashlib.sha256(z.read('MANIFEST.json')).hexdigest()!=packet_protocol['dataset']['manifest_sha256']:
                    raise ValueError('Workflow reference manifest changed')
                expected={'train.json','development.json','screen.json','confirmation.json','questions.json',
                          'initial_development.json','initial_familiar_H24.json'}
                if set(z.namelist())!=expected|{'MANIFEST.json'} or set(manifest['files'])!=expected:
                    raise ValueError('Unexpected workflow packet members')
                for name,pin in manifest['files'].items():
                    payload=z.read(name)
                    if len(payload)!=pin['bytes'] or hashlib.sha256(payload).hexdigest()!=pin['sha256']:
                        raise ValueError('Workflow reference member changed')
                    (root/name).write_bytes(payload)
            read=lambda name:json.loads((root/(name+'.json')).read_bytes())
            if read('questions')!=QUESTIONS:raise ValueError('Original workflow questions changed')
            dev=read('development');initial_dev=read('initial_development')
            train=training_rows(read('train'),dev,initial_dev)
            if (len(train),len(dev),len(initial_dev))!=(92,49,9):
                raise ValueError('Prospective joint workflow memberships changed')
            binary_config=yaml.safe_load((ROOT/'config/research2_classifier.yaml').read_bytes())
            reference_root=Path(a['reference_root'])
            if sha256(reference_root.parent/'REFERENCE_MANIFEST.json')!=binary_config['reference_manifest_sha256']:
                raise ValueError('Original binary reference manifest changed')
            binary_manifest=json.loads((reference_root.parent/'REFERENCE_MANIFEST.json').read_bytes())
            for name,pin in binary_manifest['files'].items():
                path=(reference_root/name).resolve()
                if not path.is_relative_to(reference_root.resolve()) or sha256(path)!=pin['sha256']:
                    raise ValueError('Original binary input changed')
            question=json.loads((reference_root/'study/02_jev_gate/plan.json').read_bytes())['question']
            binary_train=read_dataset('R2-JEV-TRAIN240');binary_dev=read_dataset('R2-JEV-DEV80')
            random.seed(protocol['training']['seed']);torch.manual_seed(protocol['training']['seed'])
            torch.cuda.manual_seed_all(protocol['training']['seed'])
            torch.set_num_threads(4);torch.set_num_interop_threads(1)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            model,tokenizer=load_model(source['public_base'],adapter,trainable=True,formulation='nli')
            model.to('cuda:0')
            wf_train=encode_workflow(tokenizer,train);wf_dev=encode_workflow(tokenizer,dev)
            initial_encoded=encode_workflow(tokenizer,initial_dev)
            encoded_binary_train=pairs(tokenizer,binary_train,question)
            encoded_binary_dev=pairs(tokenizer,binary_dev,question)
            examples=[]
            for i,(field,labels,tokens) in enumerate(wf_train):
                label=train[i//3]['teacher_effective'][field]
                examples.append((tokens,[float(x==label) for x in labels],'workflow:'+field))
            for i,row in enumerate(binary_train):
                p=row['shutdown_probability']
                examples.append((encoded_binary_train[2*i:2*i+2],[p,1-p],'binary'))
            if len(examples)!=516:raise ValueError('Joint effective example count changed')
            write_json(output/'INPUT_AUDIT.json',{'workflow_training_states':len(train),'workflow_training_questions':len(wf_train),
                'binary_training_states':len(binary_train),'task_instances_per_epoch':len(examples),
                'workflow_development_states':len(dev),'binary_development_states':len(binary_dev),'initial_DEV9':len(initial_dev),
                'excluded_original_training_states':5,'exact_training_validation_state_overlap':0,
                'training_ids':[row['id'] for row in train],'teacher_labels_or_ids_in_model_inputs':False,
                'max_training_pair_tokens':max(len(t['input_ids']) for tokens,_,_ in examples for t in tokens),'truncation':False,
                'historical_H24_benchmark_used_for_fitting_or_selection':False,'binary_confirmation_used_for_fitting_or_selection':False})
            if time.time()>=start+900:raise TimeoutError('Original setup window ended before training admission')
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
            if (sum(p.numel() for _,p in trainable)!=7198722
                    or any('lora_' not in n and '.modules_to_save.' not in n for n,_ in trainable)):
                raise ValueError('Unexpected trainable public-base parameter')
            before=frozen_digest(model)
            groups=[{'params':[p for n,p in trainable if '.modules_to_save.' not in n],'lr':1e-5},
                    {'params':[p for n,p in trainable if '.modules_to_save.' in n],'lr':2e-5}]
            optimizer=torch.optim.AdamW(groups,weight_decay=.01)
            scheduler=get_linear_schedule_with_warmup(optimizer,26,65*12)
            def workflow_predictions(records,queries):
                nonlocal calls
                model.eval();result=[{'model':'research3-open-nli','concepts':{}} for _ in records]
                for i,(field,labels,tokens) in enumerate(queries):
                    guard();values=[]
                    for begin in range(0,len(tokens),2):
                        x=batch(tokenizer,tokens,list(range(begin,min(begin+2,len(tokens)))),'cuda:0')
                        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):v=model(**x).logits[:,0]
                        values.extend(v.float().cpu().tolist());calls+=1
                    result[i//3]['concepts'][field]=scored_decision(torch.tensor(values).softmax(-1).tolist(),labels)
                return result
            def binary_predictions(rows,encoded):
                nonlocal calls
                model.eval();result=[]
                for begin in range(0,len(encoded),8):
                    guard();x=batch(tokenizer,encoded,list(range(begin,min(begin+8,len(encoded)))),'cuda:0')
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):v=model(**x).logits
                    result.extend(v[:,0].float().reshape(-1,2).softmax(-1)[:,0].cpu().tolist());calls+=1
                return result
            best=None;last_improvement=0;streak=0;history=[]
            for epoch in range(1,13):
                if time.time()>=end-1500:break
                phase='training';model.train();epoch_start=time.monotonic();loss_sum=0.
                order=torch.randperm(len(examples),generator=torch.Generator().manual_seed(protocol['training']['seed']+epoch)).tolist()
                for begin in range(0,len(order),8):
                    guard(training=True);members=order[begin:begin+8];optimizer.zero_grad(set_to_none=True)
                    for index in members:
                        tokens,target,_=examples[index];inputs=batch(tokenizer,tokens,list(range(len(tokens))),'cuda:0')
                        targets=torch.tensor(target,device='cuda:0',dtype=torch.float32)
                        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(**inputs).logits
                        scores=logits[:,0].float();loss=-(targets*scores.log_softmax(-1)).sum()/len(members)
                        if not torch.isfinite(loss):raise ValueError('Nonfinite joint distillation loss')
                        loss.backward();loss_sum+=float(loss.detach())*len(members);calls+=1
                    torch.nn.utils.clip_grad_norm_([p for _,p in trainable],1,error_if_nonfinite=True)
                    optimizer.step();scheduler.step();updates+=1;optimizer.zero_grad(set_to_none=True)
                    write_json(output/'PROGRESS.json',{'stage':phase,'epoch':epoch,'updates':updates,'maximum_updates':780,
                        'model_forward_calls':calls,'utc':dt.datetime.now(dt.timezone.utc).isoformat()})
                phase='development'
                wf=workflow_predictions(dev,wf_dev);bi=binary_predictions(binary_dev,encoded_binary_dev)
                initial=workflow_predictions(initial_dev,initial_encoded)
                wf_metrics=summarize(dev,wf);bi_metrics=agreement(binary_dev,bi);initial_metrics=summarize_initial(initial_dev,initial)
                passed=wf_metrics['pass'] and qualified(bi_metrics) and initial_metrics['pass']
                rates=list(wf_metrics['effective_label_agreement'].values())+[wf_metrics['directive_matches']/len(dev),
                    bi_metrics['decision_agreement'],bi_metrics['shutdown_recall_vs_JEV'],initial_metrics['enabled_gate_matches']/9,
                    initial_metrics['effective_label_matches']/9]
                score=(int(passed),min(rates),sum(rates),-wf_metrics['self_false_positives']-wf_metrics['self_false_negatives']-bi_metrics['fp'])
                streak=streak+1 if passed else 0
                entry={'epoch':epoch,'updates':updates,'training_loss':loss_sum/len(examples),
                    'epoch_seconds':time.monotonic()-epoch_start,'workflow_development':wf_metrics,'binary_development':bi_metrics,
                    'initial_DEV9':initial_metrics,'qualified_streak':streak,'utc':dt.datetime.now(dt.timezone.utc).isoformat()}
                history.append(entry);write_json(output/'TRAINING_HISTORY.json',history);write_json(output/'TRAINING_PROGRESS.json',entry)
                checkpoint=output/'checkpoints'/('epoch_'+str(epoch).zfill(2))
                model.save_pretrained(checkpoint,safe_serialization=True)
                write_json(checkpoint/'METRICS.json',entry)
                if best is None or score>best:
                    best=score;last_improvement=epoch;write_json(output/'BEST_DEVELOPMENT.json',entry)
                temporary=output/'LATEST_TRAINING_STATE.partial.pt';target=output/'LATEST_TRAINING_STATE.pt'
                torch.save({'adapter':cpu_tree(get_peft_model_state_dict(model)),'optimizer':cpu_tree(optimizer.state_dict()),
                    'scheduler':scheduler.state_dict(),'epoch':epoch,'updates':updates,'torch_rng':torch.get_rng_state(),
                    'CUDA_rng':torch.cuda.get_rng_state(0)},temporary);temporary.replace(target)
                write_json(output/'CHECKPOINT_RECEIPT.json',{'epoch':epoch,'updates':updates,'bytes':target.stat().st_size,'sha256':sha256(target)})
                if streak>=3 or epoch-last_improvement>=4:break
            if not history:raise TimeoutError('No admitted joint epoch completed')
            write_json(output/'DEVELOPMENT_PREDICTIONS_PRIVATE.json',{'workflow':wf,'binary_probabilities':bi,'initial_DEV9':initial})
            result={'status':'JOINT_CLASSIFIER_DEVELOPMENT_FAILED','workflow_development':wf_metrics,'binary_development':bi_metrics,
                'initial_DEV9':initial_metrics,'epochs':epoch,'optimizer_updates':updates,'new_JEV_calls':0,
                'workflow_benchmark_executed':False,'initial_H24_executed':False,'binary_confirmation_executed':False,
                'selection_uses_only_development':True,'benchmark_is_previously_inspected':True}
            if frozen_digest(model)!=before:raise ValueError('Public NLI base changed during joint fitting')
            if streak>=3:
                frozen=output/'joint_adapter';model.save_pretrained(frozen,safe_serialization=True);tokenizer.save_pretrained(frozen)
                saved=load_file(str(frozen/'adapter_model.safetensors'),device='cpu');live=get_peft_model_state_dict(model)
                if set(saved)!=set(live) or any(not torch.equal(v,live[k].detach().cpu()) for k,v in saved.items()):
                    raise ValueError('Saved joint adapter differs from the frozen fitted model')
                del saved,live
                result['adapter_files']=file_pins(frozen);result['saved_adapter_tensors_bitwise_verified']=True
                write_json(output/'FROZEN_CLASSIFIER.json',{'adapter_files':result['adapter_files'],'development':history[-1],
                    'frozen_base':before,'confirmation_not_yet_opened':True,'protocol_sha256':sha256(ROOT/'config/workflow_classifier_joint.yaml')})
                model.eval().requires_grad_(False);phase='historical_benchmark'
                binary_test=read_dataset('R2-JEV-CONFIRM128')
                bpred=binary_predictions(binary_test,pairs(tokenizer,binary_test,question))
                wf_test=read('confirmation');wpred=workflow_predictions(wf_test,encode_workflow(tokenizer,wf_test))
                initial_test=read('initial_familiar_H24');ipred=workflow_predictions(initial_test,encode_workflow(tokenizer,initial_test))
                result.update(binary_confirmation=agreement(binary_test,bpred),workflow_benchmark=summarize(wf_test,wpred),
                    initial_H24=summarize_initial(initial_test,ipred),workflow_benchmark_executed=True,
                    initial_H24_executed=True,binary_confirmation_executed=True)
                result['status']='JOINT_CLASSIFIER_QUALIFIED' if (qualified(result['binary_confirmation']) and
                    result['workflow_benchmark']['pass'] and result['initial_H24']['pass']) else 'JOINT_CLASSIFIER_BENCHMARK_FAILED'
                write_json(output/'BENCHMARK_PREDICTIONS_PRIVATE.json',{'workflow':wpred,'binary_probabilities':bpred,'initial_H24':ipred})
            write_json(output/'RESULT.json',result)
            return result
    except BaseException as error:
        write_json(output/'FAULT.json',{'stage':phase,'type':type(error).__name__,'updates':updates,
            'model_forward_calls':calls,'new_JEV_calls':0,'automatic_retry':False})
        raise
    finally:
        model=optimizer=scheduler=inputs=logits=scores=loss=targets=None;trainable=[];groups=[]
        gc.collect();torch.cuda.empty_cache()
        write_json(output/'MODEL_CLEANUP.json',{'model_reference_released':True,'updates':updates,
            'GPU_allocated_bytes':torch.cuda.memory_allocated(0),'GPU_reserved_bytes':torch.cuda.memory_reserved(0)})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--admission',required=True)
    result=run(parser.parse_args(argv).admission);print(json.dumps(result));return result


if __name__=='__main__':main()
