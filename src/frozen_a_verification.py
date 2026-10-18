"""One bounded replay of the frozen A arm; retain the stopped original unchanged."""
from pathlib import Path
import argparse
import datetime as dt
import gc
import hashlib
import json
import subprocess
import time
import uuid
import yaml
from src.config import ROOT, Settings
from src.utils import sha256, write_json, workspace_client
from src.matched_validation import read_packet


def configuration():
    return yaml.safe_load((ROOT/'config/frozen_a_verification.yaml').read_bytes())


def validate_episode(row, case, gate):
    if (row['case_id'] != case['id'] or row['frozen_initial_gate'] != gate
            or not row.get('completed') or row.get('runtime_fault') is not None
            or row['score']['negative_control'] == gate['enabled']):
        raise ValueError('Incomplete, mismatched or faulted source episode')


def compare_episode(original, fresh):
    """Timing and allocator details are not scientific output identities."""
    return {'token_IDs_exact': [t['token_ids'] for t in original['turns']] ==
                              [t['token_ids'] for t in fresh['turns']],
            'events_exact': original['events'] == fresh['events'],
            'scores_exact': original['score'] == fresh['score'],
            'original_success': original['score']['workflow_success'],
            'fresh_success': fresh['score']['workflow_success']}


def run(admission_path):
    import torch
    from src.verify import source_digest
    from src.matched_steering import stage_sources, scoring, verify_model_file
    a=json.loads(Path(admission_path).read_bytes());cfg=configuration();s=Settings.load()
    out=Path(a['output']);original=Path(a['original_output'])
    if (source_digest()!=a['source_digest'] or sha256(ROOT/'config/frozen_a_verification.yaml')!=a['protocol_sha256']
            or sha256(ROOT/cfg['frozen_protocol'])!=a['comparison_protocol_sha256']):
        raise ValueError('Admitted scientific source changed')
    end=dt.datetime.fromisoformat(a['absolute_end_utc']).timestamp()-cfg['export_reserve_seconds']
    setup_end=dt.datetime.fromisoformat(a['submitted_utc']).timestamp()+cfg['setup_seconds']
    if (not out.is_relative_to(Path(s.volume_root)) or out.exists() or out==original
            or time.time()>=min(setup_end,end)):
        raise ValueError('Invalid or expired one-shot output')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError('Exactly one admitted GPU required')
    packet=read_packet(a['packet_zip'],a['packet_sha256']);cases=packet['cases.json']
    predictions=json.loads((Path(a['gate_output'])/'PREDICTIONS_PRIVATE.json').read_bytes())
    if ([r['id'] for r in predictions]!=[c['id'] for c in cases]
            or len(cases)!=cfg['cases']
            or sum(r['gate']['enabled'] for r in predictions)!=cfg['fresh_own_episodes']):
        raise ValueError('Frozen 24-case membership differs')
    gates={r['id']:r['gate'] for r in predictions}
    if any(gates[c['id']]['enabled']!=c['condition'].startswith('self_') for c in cases):
        raise ValueError('Frozen initial gates differ from the qualified panel')
    gate_file=Path(a['gate_output'])/'PREDICTIONS_PRIVATE.json';gate_sha=sha256(gate_file)
    package=Path(a['comparison_inputs']);manifest=package/'INPUT_MANIFEST.json';manifest_sha=sha256(manifest)
    pins=json.loads(manifest.read_bytes())['files']
    for name,pin in pins.items():
        if sha256(package/name)!=pin: raise ValueError('Frozen comparison input changed')
    freeze=json.loads((package/'DEVELOPMENT_FREEZE.json').read_bytes())
    adapter=Path(a['adapter_directory'])
    for name,key in [('adapter_model.safetensors','A_weights_sha256'),('adapter_config.json','A_config_sha256')]:
        if sha256(adapter/name)!=freeze[key]: raise ValueError('Frozen A adapter changed')
    staged=json.loads((original/'MODEL_STAGING.json').read_bytes());model_path=Path(staged['model_path'])
    pin=json.loads((ROOT/'config/recipes/qwen38_colab/model_pin.json').read_bytes())
    out.mkdir(parents=True,exist_ok=False);write_json(out/'ADMISSION.json',a)
    stage_sources(Path('/tmp')/('r3_frozen_A_'+uuid.uuid4().hex))
    from campaign_worker import load_model
    from gpu_validation_worker import digest_model
    from matched_worker import episode,native_checks
    from selected_decoder import SelectedDecoder
    from initial_gate import FrozenGate
    from src.engines.training.bridge54_policy import effective_available_bytes
    import campaign_scoring
    original_score=campaign_scoring.score
    campaign_scoring.score=lambda world,case,initial_ok,turns:scoring(original_score,world,case,initial_ok,turns)
    generation=json.loads((package/'GENERATION.json').read_bytes())
    fixtures={r['case_id']:r for r in packet['fixtures.json']}
    deadline=time.monotonic()+max(0,end-time.time());model=None;comparisons=[];count=0
    def resources(stage, release=False):
        before={'free_bytes':torch.cuda.mem_get_info()[0], 'allocated_bytes':torch.cuda.memory_allocated(),
                'reserved_bytes':torch.cuda.memory_reserved()}
        if release:
            gc.collect();torch.cuda.empty_cache()
        r={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'stage':stage,'before':before,
           'GPU_free_bytes':torch.cuda.mem_get_info()[0], 'GPU_allocated_bytes':torch.cuda.memory_allocated(),
           'GPU_reserved_bytes':torch.cuda.memory_reserved(),'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
           'host_free_bytes':effective_available_bytes()}
        # Unity Catalog Volumes do not support append/seek. Keep each sample immutable.
        write_json(out/'resources'/(str(time.time_ns())+'.json'),r)
        if time.time()>=end: raise TimeoutError('Preserve export and cleanup reserve')
        if r['host_free_bytes']<cfg['host_reserve_GiB']*2**30: raise MemoryError('Host reserve breached')
        if r['GPU_free_bytes']<cfg['GPU_reserve_GiB']*2**30: raise MemoryError('GPU reserve breached')
        return r
    try:
        for entry in pin['files']:
            if time.time()>=min(setup_end,end): raise TimeoutError('Submission-bound setup expired')
            verify_model_file(model_path/entry['name'],entry)
        initial=resources('before_model')
        if initial['host_free_bytes']<cfg['host_initial_GiB']*2**30 or initial['GPU_free_bytes']<cfg['GPU_initial_GiB']*2**30:
            raise MemoryError('Initial host/GPU admission failed')
        torch.set_num_threads(4);torch.manual_seed(cfg['seed'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model,tok=load_model(model_path,adapter)
        before=digest_model(model)
        if before!=freeze['protected_digests']['teacher']: raise ValueError('Historical A model fingerprint differs')
        write_json(out/'MODEL_BEFORE.json',before)
        mechanics=out/'mechanics_A';mechanics.mkdir()
        checks,*counts=native_checks(model,tok,generation,deadline,package,{},'fine_tuned',mechanics)
        write_json(mechanics/'NATIVE_CHECKS.json',{'checks':checks,'counts':counts})
        if not all(r['IDs_exact'] for r in checks.values()): raise ValueError('Frozen A routing mechanics differ')
        if time.time()>=setup_end: raise TimeoutError('Submission-bound setup expired before episodes')
        write_json(out/'MODEL_READY.json',resources('ready',True))
        for case in cases:
            cid=case['id'];gate=gates[cid];dest=out/'fine_tuned_A/episodes'/(cid+'.json')
            resources('before_'+cid,True)
            old=original/('fine_tuned_A/episodes/' if gate['enabled'] else 'shared_fallback/episodes/')/(cid+'.json')
            row=json.loads(old.read_bytes());validate_episode(row,case,gate)
            if gate['enabled']:
                episode(SelectedDecoder(model,tok,generation,deadline),case,fixtures[cid],FrozenGate(**gate),'fine_tuned_A',None,dest)
                fresh=json.loads(dest.read_bytes());validate_episode(fresh,case,gate)
                comparison=compare_episode(row,fresh)
            else:
                fresh=dict(row,shared_base_fallback=True,source_episode_sha256=sha256(old))
                write_json(dest,fresh)
                comparison={'shared_control_reused':True,'source_episode_sha256':sha256(old)}
            comparisons.append(dict(case_id=cid,**comparison));count+=1
            write_json(out/'PROGRESS.json',{'utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                'case_records':count,'planned_case_records':24,'last_case':cid,'fresh_own_episodes':sum('fresh_success' in r for r in comparisons)})
        resources('before_final_fingerprint',True)
        after=digest_model(model);write_json(out/'MODEL_AFTER.json',after)
        if after!=before or sha256(gate_file)!=gate_sha or sha256(manifest)!=manifest_sha:
            raise ValueError('Final model or frozen input fingerprint differs')
        result={'status':'FROZEN_A_VERIFIED','case_records':count,'fresh_own_episodes':cfg['fresh_own_episodes'],
                'shared_control_aliases':cfg['shared_control_aliases'],'model_before_after_exact':True,
                'original_run_modified':False,'optimizer_updates':0,'new_reference_calls':0,
                'gate_predictions_sha256':gate_sha,'case_comparisons':comparisons,
                'all_fresh_tokens_events_scores_exact':all(all(r[k] for k in ('token_IDs_exact','events_exact','scores_exact'))
                      for r in comparisons if 'fresh_success' in r)}
        write_json(out/'COMPLETE.json',result);return result
    except Exception as error:
        write_json(out/'FAULT.json',{'type':type(error).__name__,'message':str(error)[:400],
                                    'case_records':count,'automatic_retry':False})
        raise
    finally:
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()
        write_json(out/'CLEANUP.json',{'utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'GPU_allocated_bytes':torch.cuda.memory_allocated(),'GPU_reserved_bytes':torch.cuda.memory_reserved()})


def submit(job_id, original_output, technical_predecessor=None, predecessor_run_id=None):
    from src.persist_data import upload_verified
    from src.verify import source_digest
    from src.runtime import launcher,runtime_task,bind_launcher
    dirty=subprocess.run(['git','status','--porcelain','--','src','config','resources'],capture_output=True,text=True,check=True)
    if dirty.stdout:raise ValueError('Commit code and recipes before submission')
    s=Settings.load();w=workspace_client(s);cfg=configuration();task=runtime_task(w,job_id)
    if task['task_key']!='verify_frozen_27b_A':raise ValueError('Wrong verification job')
    original=json.loads(w.files.download(original_output+'/ADMISSION.json').contents.read())
    now=dt.datetime.now(dt.timezone.utc);key=uuid.uuid4().hex
    root=s.volume_root+'/runs/frozen_A_verification/'+now.strftime('%Y%m%dT%H%M%SZ')+'_'+key[:8]
    a={k:original[k] for k in ['packet_zip','packet_sha256','gate_output','comparison_inputs','adapter_directory']}
    a.update(format='research3-frozen-a-verification-admission-v1',phase='frozen_A',original_output=original_output,
        submitted_utc=now.isoformat(),absolute_end_utc=(now+dt.timedelta(seconds=cfg['whole_seconds'])).isoformat(),
        output=root,protocol_sha256=sha256(ROOT/'config/frozen_a_verification.yaml'),
        comparison_protocol_sha256=sha256(ROOT/cfg['frozen_protocol']),source_digest=source_digest(),
        Free_Edition_only=True,automatic_retries=0,execution_transport='AI_RUNTIME_TASK')
    if technical_predecessor:
        if predecessor_run_id is None:raise ValueError('A terminal technical predecessor run is required')
        previous=json.loads(w.files.download(technical_predecessor+'/ADMISSION.json').contents.read())
        fault=json.loads(w.files.download(technical_predecessor+'/FAULT.json').contents.read())
        cleanup=json.loads(w.files.download(technical_predecessor+'/CLEANUP.json').contents.read())
        terminal=w.jobs.get_run(predecessor_run_id)
        if (fault.get('type')!='OSError' or fault.get('message')!='[Errno 29] Illegal seek'
                or fault.get('case_records')!=0 or terminal.state.life_cycle_state.value!='TERMINATED'
                or terminal.state.result_state.value!='FAILED' or not cleanup.get('utc')
                or previous['original_output']!=original_output):
            raise ValueError('Only the closed pre-episode Volume append fault can be repaired')
        for name in ['packet_zip','packet_sha256','gate_output','comparison_inputs','adapter_directory']:
            if a[name]!=previous[name]:raise ValueError('Technical repair cannot change scientific inputs')
        a.update(submitted_utc=previous['submitted_utc'],absolute_end_utc=previous['absolute_end_utc'],
            technical_predecessor=technical_predecessor,predecessor_run_id=predecessor_run_id,
            original_clock_preserved=True,scientific_retry=False)
        if dt.datetime.fromisoformat(a['submitted_utc'])+dt.timedelta(seconds=cfg['setup_seconds']-600)<=now:
            raise TimeoutError('Insufficient original setup window for technical repair')
    admission=root+'_ADMISSION.json';owner=w.current_user.me().user_name
    if not owner or '/' in owner or chr(92) in owner:raise ValueError('Invalid workspace owner')
    path='/Workspace/Users/'+owner+'/.research3-launchers/'+key+'/frozen_A.sh'
    source=launcher(s,admission,'frozen_A')
    a['runtime_launcher']={'path':path,'bytes':len(source),'sha256':hashlib.sha256(source).hexdigest()}
    upload_verified(w,admission,(json.dumps(a,indent=2)+'\n').encode())
    local=s.output_root/('frozen_A_verification_'+key);write_json(local/'ADMISSION.json',a)
    if bind_launcher(w,job_id,task,path,source)!=a['runtime_launcher']:raise ValueError('Launcher readback differs')
    run=w.jobs.run_now(job_id=job_id,idempotency_token=key,job_parameters={'catalog':s.catalog,'schema':s.schema,
        'volume':s.volume,'code_path':s.code_path,'admission':admission})
    result={'run_id':run.run_id,'job_id':job_id,'output':root,'admission':admission,'absolute_end_utc':a['absolute_end_utc']}
    write_json(local/'SUBMITTED.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['run','submit'])
    p.add_argument('--admission');p.add_argument('--job-id',type=int);p.add_argument('--original-output')
    p.add_argument('--technical-predecessor');p.add_argument('--predecessor-run-id',type=int)
    a=p.parse_args();print(json.dumps(run(a.admission) if a.stage=='run' else submit(a.job_id,a.original_output,a.technical_predecessor,a.predecessor_run_id)))


if __name__=='__main__':main()
