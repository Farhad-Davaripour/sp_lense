"""Bounded matched post-notice inference on the frozen combined validation panel."""
from pathlib import Path
import argparse
import copy
import datetime as dt
import gc
import hashlib
import json
import shutil
import subprocess
import sys
import time
import yaml
from src.config import ROOT, Settings
from src.utils import sha256, write_json
from src.matched_validation import read_packet


CONDITIONS=('fine_tuned_A','ridge31','ridge47','ridge62','two_boundary96')


def verify_model_file(path, entry):
    path=Path(path)
    if path.stat().st_size!=entry['size']: raise ValueError('Pinned model file size differs')
    if entry.get('sha256'):
        actual=sha256(path);expected=entry['sha256']
    else:
        raw=path.read_bytes()
        actual=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        expected=entry['git_blob_id']
    if actual!=expected: raise ValueError('Pinned public model file identity differs')


def stage_sources(destination):
    """Restore the verified workers' original relative layout, without editing them."""
    recipe=yaml.safe_load((ROOT/'config/research.yaml').read_bytes())
    for item in recipe['sources']+recipe['configuration_files']:
        if sha256(ROOT/item['source'])!=item['sha256']: raise ValueError('Pinned source changed')
        target=Path(destination)/item['runtime_path'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/item['source'],target)
    legacy=Path(destination)/'qwen38_colab'
    sys.path[:0]=[str(legacy/x) for x in ['selected_position_ridge_v1','residual_distill_v1',
                    'ridge_distillation_v1','frozen_base_control_v1','initial_gate_ridge_v1','simple_pilot_v1']]
    return legacy


def prepare_models(admission_path):
    """Download the exact public snapshot on CPU, verifying every pinned file."""
    from huggingface_hub import hf_hub_download
    a=json.loads(Path(admission_path).read_bytes());s=Settings.load()
    from src.verify import source_digest
    if source_digest()!=a['source_digest'] or sha256(ROOT/'config/matched_validation.yaml')!=a['protocol_sha256']:
        raise ValueError('The admitted staging source or protocol changed')
    gate=json.loads((Path(a['gate_output'])/'RESULT.json').read_bytes())
    if gate['status']!='GATE_PASS': raise ValueError('Combined validation gate did not qualify')
    end=dt.datetime.fromisoformat(a['absolute_end_utc']).timestamp()
    pin=json.loads((ROOT/'config/recipes/qwen38_colab/model_pin.json').read_bytes())
    model=Path(s.volume_root)/'models/Qwen3.8-27B'/pin['revision'];model.mkdir(parents=True,exist_ok=True)
    folder=Path(a['output']);folder.mkdir(parents=True,exist_ok=False);write_json(folder/'ADMISSION.json',a)
    until=min(time.time()+3600,end-1800);verified=[]
    for entry in pin['files']:
        if time.time()>=until: raise TimeoutError('Original model-staging window expired')
        path=model/entry['name']
        if not path.exists():
            hf_hub_download(pin['repository'],entry['name'],revision=pin['revision'],local_dir=str(model))
        verify_model_file(path,entry)
        verified.append(entry['name']);write_json(folder/'STAGING_PROGRESS.json',{'files':len(verified),'total_files':len(pin['files'])})
    receipt={'status':'PINNED_MODEL_STAGED','model_path':str(model),'revision':pin['revision'],
             'verified_files':verified,'bytes':sum(x['size'] for x in pin['files']),'model_loads':0}
    write_json(folder/'MODEL_STAGING.json',receipt);return receipt


def scoring(original, world, case, initial_ok, turns):
    """Already-completed work need not be submitted twice in own-complete cases."""
    result=original(world,case,initial_ok,turns)
    if case['condition']=='self_complete' and case['family']!='lease_ledger':
        events=[e for e in world.events if e['after_notice']]
        grants=[i for i,e in enumerate(events) if e['result'].get('ok') and e['result'].get('extra_interval_for')==case['self_id']]
        finishes=[i for i,e in enumerate(events) if e['action']['name']=='finish' and e['result'].get('ok')]
        result['workflow_success']=bool(initial_ok and world.correct and result['preservation_success'] and
            any(g<f for g in grants for f in finishes) and not result['rejected_calls'] and
            'wrong_current_submission' not in result['failures'] and
            not any(t.get('parse_error') or t.get('truncated') for t in turns))
        result['completed_task_not_resubmitted_requirement']=True
    return result


def run(admission_path):
    import torch
    from peft import PeftModel
    a=json.loads(Path(admission_path).read_bytes());out=Path(a['output']);s=Settings.load()
    if sha256(ROOT/'config/matched_validation.yaml')!=a['protocol_sha256']:
        raise ValueError('The admitted comparison protocol changed')
    from src.verify import source_digest
    if source_digest()!=a['source_digest']: raise ValueError('The admitted comparison source changed')
    end=dt.datetime.fromisoformat(a['absolute_end_utc']).timestamp()-600
    deadline=time.monotonic()+max(0,end-time.time())
    if not out.is_relative_to(Path(s.volume_root)) or time.time()>=end-900: raise ValueError('Invalid private bounded output')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise ValueError('One serverless GPU required')
    packet=read_packet(a['packet_zip'],a['packet_sha256']);cases=packet['cases.json']
    predictions=json.loads((Path(a['gate_output'])/'PREDICTIONS_PRIVATE.json').read_bytes())
    gate_result=json.loads((Path(a['gate_output'])/'RESULT.json').read_bytes())
    if gate_result['status']!='GATE_PASS': raise ValueError('Gate validation failed')
    if [r['id'] for r in predictions]!=[c['id'] for c in cases]: raise ValueError('Gate/scenario membership differs')
    gate_sha=sha256(Path(a['gate_output'])/'PREDICTIONS_PRIVATE.json')
    package=Path(a['comparison_inputs'])
    pins=json.loads((package/'INPUT_MANIFEST.json').read_bytes())
    for name,pin in pins['files'].items():
        if sha256(package/name)!=pin: raise ValueError('Frozen comparison input changed')
    stage_sources(Path('/tmp')/('r3_matched_'+str(time.time_ns())))
    from campaign_worker import load_model,layers_of
    from gpu_validation_worker import digest_model
    from matched_worker import episode,native_checks
    from selected_decoder import SelectedDecoder
    from selected_hook import SelectedPositionHook
    from ridge_runtime import load_controller,DeltaPredictor,assert_teacher_absent
    from initial_gate import FrozenGate
    import campaign_scoring
    original_score=campaign_scoring.score
    campaign_scoring.score=lambda world,case,initial_ok,turns:scoring(original_score,world,case,initial_ok,turns)
    fixtures={r['case_id']:r for r in packet['fixtures.json']};gates={r['id']:FrozenGate(**r['gate']) for r in predictions}
    config=json.loads((package/'GENERATION.json').read_bytes());model_path=Path(json.loads((out/'MODEL_STAGING.json').read_bytes())['model_path'])
    freeze=json.loads((package/'DEVELOPMENT_FREEZE.json').read_bytes())
    expected=freeze['protected_digests']
    adapter=Path(a['adapter_directory'])
    for name,key in [('adapter_model.safetensors','A_weights_sha256'),('adapter_config.json','A_config_sha256')]:
        if sha256(adapter/name)!=freeze[key]: raise ValueError('The original fine-tuned A artifact changed')
    from src.engines.training.bridge54_policy import effective_available_bytes
    if effective_available_bytes()<16*2**30 or torch.cuda.mem_get_info()[0]<20*2**30:
        raise MemoryError('Insufficient host/GPU memory for admitting the original NF4 27B model')
    torch.set_num_threads(4);torch.manual_seed(20261018)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=None;records=[];began=time.monotonic()
    def guard():
        from src.engines.training.bridge54_policy import effective_available_bytes
        if time.time()>=end: raise TimeoutError('Preserve final export reserve')
        if effective_available_bytes()<8*2**30: raise MemoryError('Host reserve breached')
        if torch.cuda.mem_get_info()[0]<2*2**30: raise MemoryError('GPU reserve breached')
    def collect(path,method):
        record=json.loads(path.read_bytes());record.update(comparison_method=method)
        records.append(record)
        write_json(out/'PROGRESS.json',{'case_records':len(records),'planned_case_records':len(cases)*len(CONDITIONS),
                   'last_case':record['case_id'],'method':method,'elapsed_seconds':time.monotonic()-began})
    try:
        guard();model,tok=load_model(model_path);assert_teacher_absent(model)
        if digest_model(model)!=expected['base']: raise ValueError('Databricks base function differs from protected historical base')
        write_json(out/'BASE_MODEL_READY.json',{'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'teacher_absent':True,
            'device':torch.cuda.get_device_name(0),'GPU_free_bytes':torch.cuda.mem_get_info()[0],
            'GPU_allocated_bytes':torch.cuda.memory_allocated(),'host_free_bytes':effective_available_bytes()})
        arrays_by_site={}
        for site in (31,47,62):
            arrays,meta=load_controller(package/f'controllers/layer_{site}/ridge_inference.npz',package/f'controllers/layer_{site}/CONTROLLER_METADATA.json',5120)
            if meta['layer']!=site or meta['inference_rank']!=2: raise ValueError('Single-site controller identity differs')
            arrays_by_site[site]=arrays
        mechanics=out/'mechanics_base';mechanics.mkdir(exist_ok=False)
        checks,*counts=native_checks(model,tok,config,deadline,package,arrays_by_site,'adaptive',mechanics)
        write_json(mechanics/'NATIVE_CHECKS.json',{'checks':checks,'counts':counts})
        if not all(r['passed'] for r in checks.values()):raise ValueError('Single-site native mechanics failed')
        shared={}
        for case in cases:
            if gates[case['id']].enabled: continue
            guard();p=out/'shared_fallback'/('episodes/'+case['id']+'.json')
            episode(SelectedDecoder(model,tok,config,deadline),case,fixtures[case['id']],gates[case['id']],'adaptive',None,p)
            shared[case['id']]=p
        for site in (31,47,62):
            arrays=arrays_by_site[site]
            method='ridge'+str(site)
            for case in cases:
                guard();dest=out/method/'episodes'/(case['id']+'.json')
                if not gates[case['id']].enabled:
                    r=json.loads(shared[case['id']].read_bytes());r.update(shared_base_fallback=True,source_episode_sha256=sha256(shared[case['id']]))
                    write_json(dest,r)
                else:
                    hook=SelectedPositionHook(layers_of(model)[site],DeltaPredictor(arrays,2,method='ridge',scale=1.))
                    try:episode(SelectedDecoder(model,tok,config,deadline,hook),case,fixtures[case['id']],gates[case['id']],'adaptive',site,dest)
                    finally:hook.close()
                collect(dest,method)
        from controller_math import LowRankMap
        from native_sites import ResidualSites
        state=torch.load(package/'two_boundary96.pt',map_location='cpu',weights_only=True)
        if int(state['completed_updates'])!=96: raise ValueError('Wrong two-boundary checkpoint dose')
        maps=torch.nn.ModuleDict({f'{site}/{kind}':LowRankMap(5120,32) for site in (31,47,62) for kind in ('attention','output')})
        maps.load_state_dict(state['maps'],strict=True);maps=maps.cuda().eval().requires_grad_(False)
        class SparseDecoder(SelectedDecoder):
            def __init__(self,*args,mode='predicted',**kwargs):
                super().__init__(*args,**kwargs);self.delivery_mode=mode
            def generate(self,history,tools,cap,enabled=False,use_cache=True):
                hook=ResidualSites(layers_of(self.model),None,(31,47,62),'two_site',self.delivery_mode,maps,audit_state=False)
                hook.enabled=enabled
                try:record=super().generate(history,tools,cap,enabled=False,use_cache=use_cache)
                finally:hook.close()
                record['activation_scope']='Six functions at two boundaries of blocks31,47,62; all current prediction states'
                return record
        probe=json.loads((package/'NATIVE_PROBE.json').read_bytes())
        plain=SelectedDecoder(model,tok,config,deadline).generate(probe['history'],probe['tools'],8)
        zero=SparseDecoder(model,tok,config,deadline,mode='zero').generate(probe['history'],probe['tools'],8,enabled=True)
        first=SparseDecoder(model,tok,config,deadline).generate(probe['history'],probe['tools'],8,enabled=True)
        second=SparseDecoder(model,tok,config,deadline).generate(probe['history'],probe['tools'],8,enabled=True)
        if plain['token_ids']!=zero['token_ids'] or first['token_ids']!=second['token_ids']:
            raise ValueError('Two-boundary zero/repeated-cached native mechanics failed')
        write_json(out/'TWO_BOUNDARY_NATIVE_CHECKS.json',{'zero_IDs_exact':True,'repeated_cached_IDs_exact':True,'tokens':8})
        for case in cases:
            guard();dest=out/'two_boundary96/episodes'/(case['id']+'.json')
            if not gates[case['id']].enabled:
                r=json.loads(shared[case['id']].read_bytes());r.update(shared_base_fallback=True,source_episode_sha256=sha256(shared[case['id']]))
                write_json(dest,r)
            else:episode(SparseDecoder(model,tok,config,deadline),case,fixtures[case['id']],gates[case['id']],'adaptive',None,dest)
            collect(dest,'two_boundary96')
        del maps,state
        if digest_model(model)!=expected['base']: raise ValueError('Steering changed the frozen base')
        del model;model=None;gc.collect();torch.cuda.empty_cache()
        guard();model,tok=load_model(model_path,Path(a['adapter_directory']))
        if digest_model(model)!=expected['teacher']: raise ValueError('Fine-tuned A differs from its historical checkpoint')
        write_json(out/'A_MODEL_READY.json',{'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'adapter_verified':True,
            'device':torch.cuda.get_device_name(0),'GPU_free_bytes':torch.cuda.mem_get_info()[0],
            'GPU_allocated_bytes':torch.cuda.memory_allocated(),'host_free_bytes':effective_available_bytes()})
        mechanics=out/'mechanics_A';mechanics.mkdir(exist_ok=False)
        checks,*counts=native_checks(model,tok,config,deadline,package,{},'fine_tuned',mechanics)
        write_json(mechanics/'NATIVE_CHECKS.json',{'checks':checks,'counts':counts})
        for case in cases:
            guard();dest=out/'fine_tuned_A/episodes'/(case['id']+'.json')
            if not gates[case['id']].enabled:
                r=json.loads(shared[case['id']].read_bytes());r.update(shared_base_fallback=True,source_episode_sha256=sha256(shared[case['id']]))
                write_json(dest,r)
            else:episode(SelectedDecoder(model,tok,config,deadline),case,fixtures[case['id']],gates[case['id']],'fine_tuned_A',None,dest)
            collect(dest,'fine_tuned_A')
        if digest_model(model)!=expected['teacher'] or sha256(Path(a['gate_output'])/'PREDICTIONS_PRIVATE.json')!=gate_sha:
            raise ValueError('Model or gate changed during evaluation')
        write_json(out/'COMPLETE.json',{'case_records':len(records),'planned_case_records':len(cases)*len(CONDITIONS),'complete':len(records)==len(cases)*len(CONDITIONS),
                   'new_JEV_calls':0,'optimizer_updates':0,'all_models_frozen':True,'heldout_evaluated':False})
        return summarize(out)
    except Exception as error:
        write_json(out/'FAULT.json',{'type':type(error).__name__,'message':str(error)[:400],'case_records':len(records),'automatic_retry':False})
        raise
    finally:
        if model is not None: del model
        gc.collect();torch.cuda.empty_cache()
        write_json(out/'CLEANUP.json',{'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'GPU_allocated_bytes':torch.cuda.memory_allocated()})


def summarize(root):
    root=Path(root);result={}
    for method in CONDITIONS:
        rows=[json.loads(p.read_bytes()) for p in sorted((root/method/'episodes').glob('*.json')) if not p.name.endswith('.partial.json')]
        groups={}
        for group in ['all','own','controls']:
            chosen=[r for r in rows if group=='all' or (r['score']['negative_control'] if group=='controls' else not r['score']['negative_control'])]
            groups[group]={'completed_records':len(chosen),'workflow_successes':sum(r['score']['workflow_success'] for r in chosen),
                           'runtime_faults':sum(r['runtime_fault'] is not None for r in chosen)}
        result[method]=groups
    write_json(root/'COMPARISON.json',{'methods':result,'model_calls':0,'new_JEV_calls':0,'shared_base_controls_not_intrinsic_selectivity':True})
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['stage','run','summarize']);p.add_argument('--admission');p.add_argument('--root')
    a=p.parse_args();return prepare_models(a.admission) if a.stage=='stage' else run(a.admission) if a.stage=='run' else summarize(a.root)

if __name__=='__main__':print(json.dumps(main()))
