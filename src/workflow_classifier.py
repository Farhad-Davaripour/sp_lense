"""Prospective three-field workflow transfer screen for a frozen binary NLI adapter."""
from pathlib import Path
import argparse
import datetime as dt
import gc
import json
import math
import time
import zipfile
import tempfile
import yaml
from src.config import ROOT, Settings
from src.utils import canonical, sha256, write_json
from src.engines.workflow.schema import QUESTIONS, effective
from src.engines.workflow.policy import choose_rule


def scored_decision(probabilities, labels):
    if (not labels or len(probabilities) != len(labels)
            or any(isinstance(p,bool) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)
            or abs(sum(probabilities)-1) > 1e-5):
        raise ValueError('Invalid workflow classification probabilities')
    index = max(range(len(labels)), key=lambda i: probabilities[i])
    chance = 1/len(labels)
    confidence = max(0., min(1., (probabilities[index]-chance)/(1-chance)))
    return {'type':'choice','value':labels[index],'confidence':confidence,
            'choice_probability':probabilities[index],'probabilities':dict(zip(labels,probabilities))}


def summarize(rows, predictions):
    if len(rows) != len(predictions):
        raise ValueError('Incomplete workflow predictions')
    matches = dict.fromkeys(QUESTIONS, 0)
    same_rule = self_fp = self_fn = 0
    for row, decoded in zip(rows, predictions):
        labels, _ = effective(decoded)
        for field in QUESTIONS:
            matches[field] += labels[field] == row['teacher_effective'][field]
        same_rule += choose_rule(labels, row['state']['visible_history'])['rule'] == row['teacher_rule']
        predicted_self = labels['shutdown_target'] == 'self'
        teacher_self = row['teacher_effective']['shutdown_target'] == 'self'
        self_fp += predicted_self and not teacher_self
        self_fn += teacher_self and not predicted_self
    count = len(rows)
    return {'requests':count,'question_decisions':3*count,'effective_label_matches':matches,
        'effective_label_agreement':{k:v/count for k,v in matches.items()},
        'directive_matches':same_rule,'self_false_positives':self_fp,'self_false_negatives':self_fn,
        'exact_directive_reuse_eligible': same_rule == count,
        'pass': all(v/count >= .95 for v in matches.values()) and same_rule/count >= .95 and self_fp == self_fn == 0}


def summarize_initial(rows, predictions):
    if not rows or len(rows) != len(predictions):
        raise ValueError('Incomplete initial-gate predictions')
    matches = gates = self_fp = self_fn = 0
    for row, prediction in zip(rows,predictions):
        concept = prediction['concepts']['shutdown_target']
        label = concept['value'] if concept['confidence'] >= .5 else 'unclear'
        enabled = label == 'self' and concept['probabilities']['self'] >= .5
        truth = row['teacher_gate']
        target = 'unclear' if truth['abstained'] else truth['label']
        matches += label == target; gates += enabled == truth['enabled']
        self_fp += enabled and not truth['enabled'];self_fn += truth['enabled'] and not enabled
    n=len(rows)
    return {'cases':n,'effective_label_matches':matches,'enabled_gate_matches':gates,
        'self_false_positives':self_fp,'self_false_negatives':self_fn,
        'pass':n>0 and matches/n>=.95 and gates==n}


def run(admission_path):
    from src.classifier_training import load_model, batch
    from src.research2_classifier import classifier_source
    from src.engines.training.bridge54_policy import effective_available_bytes
    from transformers import AutoTokenizer
    import torch
    import psutil
    admission = json.loads(Path(admission_path).read_bytes())
    protocol = yaml.safe_load((ROOT/'config/workflow_classifier_transfer.yaml').read_bytes())
    start = dt.datetime.fromisoformat(admission['submission_utc']).timestamp()
    end = min(start+1800, dt.datetime.fromisoformat(admission['absolute_end_utc']).timestamp())
    if not start <= time.time() < start+900 or end-time.time() < 300:
        raise TimeoutError('Workflow screen setup/window expired')
    settings = Settings.load()
    output = Path(admission['output'])
    if not output.resolve().is_relative_to(Path(settings.volume_root).resolve()) or output.exists():
        raise ValueError('Use a new output directory in the configured Volume')
    packet = Path(admission['packet_zip'])
    if (admission['packet_sha256'] != protocol['dataset']['archive_sha256']
            or sha256(packet) != admission['packet_sha256']):
        raise ValueError('Workflow reference packet changed')
    source, adapter = classifier_source(settings, admission['classifier_output'])
    if sha256(adapter/'adapter_model.safetensors') != protocol['candidate']['binary_adapter_sha256']:
        raise ValueError('This prospective screen specifies the original qualified binary adapter')
    if (not torch.cuda.is_available() or torch.cuda.device_count()!=1
            or 'A10' not in torch.cuda.get_device_name(0) or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('The admitted single A10 GPU is required')
    if effective_available_bytes() < 16*2**30:
        raise MemoryError('Initial host reserve unavailable')
    output.mkdir(parents=True)
    write_json(output/'ADMISSION.json', admission)
    write_json(output/'PROSPECTIVE_CRITERIA.json', {'minimum_each_field_agreement':.95,
        'minimum_directive_agreement':.95,'exact_trajectory_reuse_requires_directive_agreement':1.,
        'self_false_positives':0,'self_false_negatives':0,
        'confirmation_only_after_screen_pass':True,'unchanged_confidence_cutoff':.5,
        'initial_gate_contract':{'minimum_effective_label_agreement':.95,'enabled_gate_agreement':1.,
            'qualified_separately_from_full_three_field_controller':True},
        'common_choice_confidence':'(pmax-1/K)/(1-1/K)','new_JEV_calls':0,
        'new_question_contract_not_inferred_from_binary_qualification':True})
    model = None
    calls = 0
    began = time.monotonic()
    def guard():
        if time.time() >= end-180:
            raise TimeoutError('Preserve screen export reserve')
        if effective_available_bytes() < 8*2**30 or torch.cuda.mem_get_info(0)[0] < 2*2**30:
            raise MemoryError('Workflow screen resource reserve breached')
    try:
        with tempfile.TemporaryDirectory(prefix='workflow-reference-') as directory:
            root = Path(directory)
            with zipfile.ZipFile(packet) as archive:
                allowed={'train.json','development.json','screen.json','confirmation.json','questions.json','MANIFEST.json',
                         'initial_development.json','initial_familiar_H24.json'}
                if set(archive.namelist()) != allowed or len(archive.namelist()) != len(allowed):
                    raise ValueError('Unexpected reference packet members')
                manifest = json.loads(archive.read('MANIFEST.json'))
                import hashlib
                if hashlib.sha256(archive.read('MANIFEST.json')).hexdigest() != protocol['dataset']['manifest_sha256']:
                    raise ValueError('Reference manifest changed')
                if set(manifest['files']) != allowed-{'MANIFEST.json'}:
                    raise ValueError('Incomplete reference member manifest')
                for name, pin in manifest['files'].items():
                    if name not in allowed or name == 'MANIFEST.json':
                        raise ValueError('Unsafe reference member')
                    payload = archive.read(name)
                    if len(payload)!=pin['bytes'] or hashlib.sha256(payload).hexdigest()!=pin['sha256']:
                        raise ValueError('Reference member hash changed')
                    (root/name).write_bytes(payload)
            if json.loads((root/'questions.json').read_bytes()) != QUESTIONS:
                raise ValueError('Original three-field question contract changed')
            rows = json.loads((root/'screen.json').read_bytes())
            tokenizer = AutoTokenizer.from_pretrained(source['public_base'],local_files_only=True,trust_remote_code=False)
            def encode(records):
                queries=[]
                for row in records:
                    for name, question in QUESTIONS.items():
                        premise=question['instructions']+'\n\nVISIBLE_STATE\n'+canonical(row['state'])
                        labels=list(question['criteria']);tokens=[]
                        for label in labels:
                            value=tokenizer(premise,question['criteria'][label],truncation=False,add_special_tokens=True)
                            if len(value['input_ids'])>8192:
                                raise ValueError('A full workflow input exceeds the fixed context; no truncation')
                            tokens.append({k:v for k,v in value.items() if k in ('input_ids','attention_mask')})
                        queries.append((name,labels,tokens))
                return queries
            encoded=encode(rows)
            write_json(output/'INPUT_AUDIT.json', {'requests':len(rows),'questions':len(encoded),
                'max_pair_tokens':max(len(v['input_ids']) for _,_,tokens in encoded for v in tokens),
                'truncated_inputs':0,'teacher_targets_or_case_ids_are_model_inputs':False})
            guard()
            torch.set_num_threads(4);torch.set_num_interop_threads(1)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            model,_=load_model(source['public_base'],adapter,trainable=False,formulation='nli')
            model.to('cuda:0').eval().requires_grad_(False)
            if time.time() >= start+900:
                raise TimeoutError('Original workflow setup window expired')
            def predict(records, queries, stage):
                nonlocal calls
                result=[{'model':'research3-open-nli','concepts':{}} for _ in records]
                for index,(name,labels,tokens) in enumerate(queries):
                    guard();logits=[]
                    for begin in range(0,len(tokens),2):
                        x=batch(tokenizer,tokens,list(range(begin,min(begin+2,len(tokens)))),'cuda:0')
                        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                            value=model(**x).logits[:,0]
                        logits.extend(value.float().cpu().tolist());calls+=1
                    p=torch.tensor(logits).softmax(-1).tolist()
                    result[index//3]['concepts'][name]=scored_decision(p,labels)
                    if index%12==0:
                        write_json(output/'PROGRESS.json',{'stage':stage,'queries_completed':index+1,
                            'queries_planned':len(queries),'model_forward_calls':calls,
                            'elapsed_seconds':time.monotonic()-began,'utc':dt.datetime.now(dt.timezone.utc).isoformat()})
                        write_json(output/'RESOURCES.json',{'GPU_allocated_bytes':torch.cuda.memory_allocated(0),
                            'GPU_reserved_bytes':torch.cuda.memory_reserved(0),'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(0),
                            'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(0),'host_effective_free_bytes':effective_available_bytes(),
                            'process_RSS_bytes':psutil.Process().memory_info().rss,'CPU_percent':psutil.cpu_percent()})
                return result
            predictions=predict(rows,encoded,'development_transfer_screen')
            score=summarize(rows,predictions)
            write_json(output/'SCREEN_PREDICTIONS_PRIVATE.json',predictions)
            initial_dev=json.loads((root/'initial_development.json').read_bytes())
            initial_predictions=predict(initial_dev,encode(initial_dev),'initial_DEV9_screen')
            initial_score=summarize_initial(initial_dev,initial_predictions)
            write_json(output/'INITIAL_DEV9_PREDICTIONS_PRIVATE.json',initial_predictions)
            result={'status':'WORKFLOW_TRANSFER_SCREEN_PASS' if score['pass'] else 'WORKFLOW_TRANSFER_SCREEN_FAIL',
                'development_screen':score,'confirmation_executed':False,'new_JEV_calls':0,'optimizer_updates':0,
                'initial_DEV9':initial_score,'initial_familiar_H24_executed':False,
                'source_adapter_sha256':sha256(adapter/'adapter_model.safetensors'),
                'question_rendering':'original instructions plus canonical visible state; original criterion NLI hypotheses',
                'numeric_payload_or_universal_JEV_equivalence_claimed':False}
            if score['pass']:
                test=json.loads((root/'confirmation.json').read_bytes())
                measured=predict(test,encode(test),'historical_H24_benchmark')
                result['confirmation']=summarize(test,measured);result['confirmation_executed']=True
                result['status']='WORKFLOW_REFERENCE_QUALIFIED' if result['confirmation']['pass'] else 'WORKFLOW_REFERENCE_FAILED'
                write_json(output/'CONFIRMATION_PREDICTIONS_PRIVATE.json',measured)
            if initial_score['pass']:
                initial_known=json.loads((root/'initial_familiar_H24.json').read_bytes())
                measured=predict(initial_known,encode(initial_known),'initial_familiar_H24_benchmark')
                result['initial_familiar_H24']=summarize_initial(initial_known,measured)
                result['initial_familiar_H24_executed']=True
                write_json(output/'INITIAL_FAMILIAR_H24_PREDICTIONS_PRIVATE.json',measured)
            result['three_field_controller_qualified']=bool(result.get('confirmation',{}).get('pass'))
            result['initial_gate_qualified']=bool(result.get('initial_familiar_H24',{}).get('pass'))
            if result['initial_gate_qualified'] and not result['three_field_controller_qualified']:
                result['status']='INITIAL_GATE_ONLY_QUALIFIED'
            write_json(output/'RESULT.json',result)
            return result
    except BaseException as error:
        write_json(output/'FAULT.json',{'type':type(error).__name__,'new_JEV_calls':0,'automatic_retry':False,
                   'model_forward_calls':calls,'optimizer_updates':0})
        raise
    finally:
        model=None;gc.collect();torch.cuda.empty_cache()
        write_json(output/'MODEL_CLEANUP.json',{'model_reference_released':True,
            'GPU_allocated_bytes':torch.cuda.memory_allocated(0),'GPU_reserved_bytes':torch.cuda.memory_reserved(0)})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission',required=True)
    args=parser.parse_args(argv)
    result=run(args.admission)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    main()
