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
            'pass':fp == 0 and fn == 0, 'reference':'frozen scenario conditions, not JEV predictions'}


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
    rows, fixtures, cases = [], [], cases16 + cases9
    if len(cases16) != 16 or len(cases9) != 9 or len({r['id'] for r in cases}) != 25:
        raise ValueError('Expected all 25 distinct scenario IDs; never discard a case for a round count')
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
    manifest = {'case_count':25,'panel_counts':{'V1-W16':16,'DEV9':9},'distinct_visible_states':len(set(hashes)),
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
    return {name:json.loads(data) for name,data in raw.items()}


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
    config=configuration();stage='setup';updates=0
    def guard():
        if time.time()>=end-300: raise TimeoutError('Preserve export reserve')
        if effective_available_bytes()<8*2**30: raise MemoryError('Host reserve breached')
        if torch.cuda.is_available() and torch.cuda.mem_get_info(0)[0]<2*2**30:
            raise MemoryError('GPU reserve breached')
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
            return result
        stage='frozen_initial_evaluation';pred=predict();initial=scores(pred)
        write_json(out/'INITIAL_PREDICTIONS_PRIVATE.json',pred);write_json(out/'INITIAL_METRICS.json',initial)
        selected=out/'selected_adapter';model.save_pretrained(selected,safe_serialization=True)
        best=initial;rank=lambda value:(value['combined']['gate_accuracy'],value['combined']['effective_label_matches'])
        history=[]
        if not initial['combined']['pass']:
            train=packet['training.json'];train_tokens,_=encode(tokenizer,train,packet['questions.json'],'visible_role_bound_full_context')
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
            groups=[{'params':[p for n,p in parameters if '.modules_to_save.' not in n],'lr':1e-5},
                    {'params':[p for n,p in parameters if '.modules_to_save.' in n],'lr':2e-5}]
            optimizer=torch.optim.AdamW(groups,weight_decay=.01)
            for epoch in range(1,31):
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
                    torch.nn.utils.clip_grad_norm_([p for n,p in parameters],1.,error_if_nonfinite=True)
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
                'optimizer_updates':updates,'new_JEV_calls':0,'reference':'scenario conditions',
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
    if recomputed!=result['selected']['combined']: raise ValueError('Independent gate counts differ')
    write_json(root/'COMPARISON.json',{'verified':True,'combined':recomputed,'model_calls':0,'new_JEV_calls':0})
    return recomputed


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','gate','compare'])
    p.add_argument('--dev9-root');p.add_argument('--native-archive');p.add_argument('--output');p.add_argument('--admission')
    a=p.parse_args()
    result=prepare(a.dev9_root,a.native_archive,a.output) if a.stage=='prepare' else run_gate(a.admission) if a.stage=='gate' else compare(a.admission)
    print(json.dumps(result));return result

if __name__=='__main__':main()
