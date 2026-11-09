"""One-use v2 owned process supervision; never stop a resident service here."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import psutil
import game_all_layers_plan as plans
import game_all_layers_layout as layout
import owned_validation_process as owned

LOCK=plans.PROJECT/'.local/validation/qwen38_shared_v1/inference.lock'
STATUS=plans.PROJECT/'.local/monitor/sp_lens_all_layers_v2_status.json'


def once(path,value):
    with path.open('x',encoding='utf-8')as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())


def live(value):
    STATUS.parent.mkdir(parents=True,exist_ok=True);temporary=STATUS.with_name(STATUS.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x')as stream:json.dump({**value,'updated_epoch':time.time()},stream,allow_nan=False)
    os.replace(temporary,STATUS)


def idle_check():
    if sys.platform!='darwin'or os.uname().machine!='arm64':raise RuntimeError('Requires the local Apple-silicon Mac')
    if (plans.PROJECT/'.local/monitor/local_game_service/service.lock').exists():raise RuntimeError('Resident service must be independently stopped before a model run')
    names=('game_training_worker.py','game_all_layers_worker.py','exact_prefix_training.py','validation_mlx_worker.py','heldout_mlx_worker.py','local_game_service.py')
    for process in psutil.process_iter(['pid','cmdline']):
        try:
            if process.pid!=os.getpid()and any(str(arg).endswith(names)for arg in (process.info['cmdline']or[])):
                raise RuntimeError('Another model process is active; no simultaneous models')
        except psutil.NoSuchProcess:continue
        except psutil.AccessDenied as exc:raise RuntimeError('Cannot establish model ownership')from exc
    if psutil.virtual_memory().available<8*1024**3:raise RuntimeError('Need at least 8 GiB available before loading')


def campaign_for(plan):
    if (plans.ROOT/'stopped_after_failure.json').exists():raise RuntimeError('Campaign stopped after a stage failure; root evidence review required')
    path=plans.ROOT/'campaign.json'
    if not path.exists():
        if plan['stage']!='pilot_rank4':raise RuntimeError('Campaign must begin with the first rank-4 pilot')
        once(path,{'schema':'research4.sp-lens-game-all-layers-campaign.v2','started_epoch':time.time(),
            'started_monotonic':time.monotonic(),'boot_time':psutil.boot_time(),'global_wall_seconds':22*3600,
            'dataset_manifest_sha256':plans.DATA_SHA,'protocol':plan['protocol'],'config':plan['config'],
            'source_sha256':plan['source_sha256'],'cohort':plan['cohort'],'baseline_completion':plan['baseline_completion'],'retries':0})
    value=json.loads(path.read_text())
    if (value.get('boot_time'),value.get('global_wall_seconds'),value.get('dataset_manifest_sha256'),value.get('protocol'),value.get('config'))!= (
        psutil.boot_time(),22*3600,plans.DATA_SHA,plan['protocol'],plan['config']):raise RuntimeError('Campaign clock/identity differs; no implicit restart')
    if any(value.get(key)!=plan[key]for key in('source_sha256','cohort','baseline_completion')):
        raise RuntimeError('Campaign source, cohort or baseline changed; no silent cross-recipe sweep')
    if time.monotonic()-value['started_monotonic']>=22*3600 or time.time()-value['started_epoch']>=22*3600:
        raise TimeoutError('Campaign exhausted before launch; no model process started')
    return value


def verify_finished(run,plan,plan_path,plan_sha):
    path=run/('trial_receipt.json'if plan['mode']=='trial'else'worker_receipt.json')
    receipt=json.loads(path.read_text())
    if (receipt.get('status'),receipt.get('stage'),receipt.get('completed_steps'),receipt.get('expected_steps'),
        receipt.get('rank'),receipt.get('num_layers'),receipt.get('initialization'),receipt.get('plan_sha256'),
        receipt.get('gradient_b_layer_coverage'),receipt.get('gradient_all_finite'))!= (
        'complete',plan['stage'],plan['iterations'],plan['iterations'],plan['rank'],64,'fresh_pinned_base',plan_sha,list(range(64)),True):
        raise ValueError('Completed stage receipt differs')
    base=json.loads((run/'full_base_file_verify.json').read_text())
    if (base.get('status'),base.get('model'),base.get('revision'),base.get('manifest_sha256'),base.get('full_inventory_verified_before_and_after'))!= (
        'passed',plans.MODEL_ID,plans.REVISION,plans.MODEL_SHA,True):raise ValueError('Before/after full-base verification differs')
    verified=layout.verify_saved_adapter(run/'adapters/adapters.safetensors',run/'adapters/adapter_config.json',
        plans.PROJECT/'.local/models/qwen3.8-27b-mlx-4bit',plan['rank'],64,expected_iterations=plan['iterations'],
        expected_learning_rate=plan['learning_rate'],require_all_layer_updates=True)
    if verified!=receipt['verification']:raise ValueError('Actual saved tensor verification changed')
    if plan['mode']!='final':return receipt,path
    def artifact(path):return {'path':plans.rel(path),'sha256':plans.sha(path)}
    selection_path=plans.path_for(plan['selection']['path']);selection=json.loads(selection_path.read_text())
    completion={'schema':'research4.sp-lens-game-all-layers-completion.v2','status':'passed','label':'sp-lense-adaptors-all-layers',
        'model':{'repo_id':plans.MODEL_ID,'revision':plans.REVISION,'manifest_sha256':plans.MODEL_SHA,'config_sha256':base['config_sha256']},
        'adapter_geometry':{'rank':plan['rank'],'num_layers':64,'scale':2.0,'dropout':0.0,'learning_rate':plan['learning_rate']},
        'artifacts':{'adapter':artifact(run/'adapters/adapters.safetensors'),'adapter_config':artifact(run/'adapters/adapter_config.json'),
            'run_config':artifact(run/'run_config.json'),'plan':artifact(plan_path),
            'dataset_manifest':artifact(plans.path_for(json.loads(plans.path_for(plan['cohort']['path']).read_text())['dataset_manifest_path'])),
            'full_base_file_verify':artifact(run/'full_base_file_verify.json')},
        'catalog_files':[{'path':key,'sha256':value}for key,value in sorted(plan['catalog_sha256'].items())],
        'training_sources':[artifact(plans.PROJECT/'mac_studio'/name)for name in plans.FILES],
        'completed_steps':plan['iterations'],'expected_steps':plan['iterations'],'all_finite':True,'nonzero_lora_b':True,
        'all_requested_layers_have_nonzero_b':True,'base_files_unchanged':True,'heldout_used_for_training':False,
        'checkpoint_origin':{'initialization':'fresh_pinned_base','source_path':plans.rel(run/'adapters/adapters.safetensors'),
            'source_sha256':verified['adapter_sha256'],'step':plan['iterations']},
        'tuning_selection':{**plan['selection'],'winner_trial_id':selection['winner']['trial_id'],
            'criterion':'pooled_dev_target_token_nll','training_initialization':'fresh_pinned_base'},
        'loss_version':plan['loss_version'],'scope':'Training/file completion only; independent reload/serving and final game comparison remain required.'}
    once(run/'completion_audit.json',completion)
    return completion,run/'completion_audit.json'


def supervise(plan_path,plan_sha,flow_run_id=None,display=None):
    plan,_,_=plans.read_plan(plan_path,plan_sha);idle_check()
    LOCK.parent.mkdir(parents=True,exist_ok=True);LOCK.mkdir(exist_ok=False)
    owner=uuid.uuid4().hex;slot=owned.LaunchSlot();cleaned=False;cleanup_attempted=False;run=None
    job={'status':'starting','model_label':'sp-lense-adaptors-all-layers','stage':plan['stage'],'stage_kind':plan['mode'],
        'rank':plan['rank'],'num_layers':64,'completed_steps':0,'total_steps':plan['iterations'],'development_completed':0,
        'development_total':0 if plan['mode']=='pilot'else 144,'stage_complete':False,'final_model_complete':False,
        'tokens_per_second':None,'sequence_tokens_per_second':None,'loss':None,'prefect_flow_run_id':flow_run_id}
    started=time.monotonic();minimum=psutil.virtual_memory().available
    try:
        once(LOCK/'owner.json',{'owner':owner,'pid':os.getpid(),'kind':'sp_lens_all_layers_v2','created_epoch':time.time()})
        idle_check();plans.ROOT.mkdir(parents=True,exist_ok=True);campaign=campaign_for(plan)
        authorities=plans.ROOT/'authorities';authorities.mkdir(exist_ok=True)
        runs=plans.ROOT/'runs';runs.mkdir(exist_ok=True)
        stage_marker=authorities/('stage_'+plan['stage']+'.json')
        if stage_marker.exists():raise RuntimeError('Stage already consumed; no automatic retry/new authority')
        run=runs/(plan['stage']+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+owner[:8]);run.mkdir(exist_ok=False)
        permit={'owner':owner,'plan_sha256':plan_sha,'run_dir':str(run),'stage':plan['stage'],'created_epoch':time.time(),'retries':0}
        once(stage_marker,permit);once(authorities/(plan_sha+'.json'),permit)
        once(run/'request.json',{'plan_path':str(plan_path.resolve()),**permit})
        command=[str(plans.PROJECT/'.local/qwen38-venv/bin/python'),'-u',str(plans.PROJECT/'mac_studio/game_all_layers_worker.py'),
            '--plan',str(plan_path.resolve()),'--plan-sha256',plan_sha,'--run-dir',str(run),'--owner',owner]
        job.update({'run_dir':str(run),'dataset_manifest_path':'.local/datasets/sp_lens_game_v1/authorized_management_v1_r4/manifest.json',
            'plan_sha256':plan_sha,'campaign_deadline_epoch':campaign['started_epoch']+22*3600,'note':'Compatibility only'if plan['mode']=='pilot'else'Full-depth development trial'if plan['mode']=='trial'else'Fresh-base final all-layer training'})
        live(job)
        with (run/'worker.log').open('xb')as log:
            pending=owned.launch(slot,command,stdin=subprocess.DEVNULL,stdout=log,cwd=plans.PROJECT)
            if pending:raise InterruptedError('Signal during owned worker spawn')
            job.update({'status':'running','pid':slot.process.pid,'process_created_at':psutil.Process(slot.process.pid).create_time()});live(job)
            while True:
                exited=owned.peek_exited(slot.process.pid)
                worker_progress=run/'worker_progress.json'
                if worker_progress.exists():
                    value=json.loads(worker_progress.read_text())
                    if (value.get('stage'),value.get('rank'),value.get('num_layers'))!=(plan['stage'],plan['rank'],64):raise ValueError('Live worker identity differs')
                    job.update(value)
                elapsed=time.monotonic()-started;available=psutil.virtual_memory().available;minimum=min(minimum,available)
                job.update({'elapsed_seconds':elapsed,'available_bytes':available,'minimum_available_bytes':minimum,'swap_used_bytes':psutil.swap_memory().used});live(job)
                if elapsed>=plan['max_wall_seconds']:raise TimeoutError('Stage operator wall deadline')
                if time.monotonic()-campaign['started_monotonic']>=22*3600 or time.time()-campaign['started_epoch']>=22*3600:
                    raise TimeoutError('Global 22-hour all-layer experiment deadline')
                if exited:break
                if available<plan['memory_floor_bytes']:raise RuntimeError('Available memory below 4 GiB')
                if display is not None:display(job)
                time.sleep(.5)
            cleanup_attempted=True;code,descendants,pending=owned.cleanup(slot.process);cleaned=True
        if code!=0 or descendants or pending:raise RuntimeError('Owned stage failed/descendant observed; no retry')
        plans.read_plan(plan_path,plan_sha)
        job.update({'status':'finalizing'});live(job)
        receipt,receipt_path=verify_finished(run,plan,plan_path,plan_sha)
        raw_receipt=run/('trial_receipt.json'if plan['mode']=='trial'else'worker_receipt.json')
        closeout={'schema':'research4.sp-lens-game-all-layers-stage-closeout.v2','status':'complete',
            'stage':plan['stage'],'mode':plan['mode'],'rank':plan['rank'],'plan_sha256':plan_sha,'cleanup_confirmed':True,
            'worker_receipt':{'path':plans.rel(raw_receipt),'sha256':plans.sha(raw_receipt)},
            'result_receipt':{'path':plans.rel(receipt_path),'sha256':plans.sha(receipt_path)},
            'supervisor_elapsed_seconds':time.monotonic()-started,'minimum_available_bytes':minimum,'closed_epoch':time.time()}
        once(run/'stage_closeout.json',closeout)
        job.update({'status':'completed','stage_complete':True,'final_model_complete':plan['mode']=='final',
            'completed_steps':plan['iterations'],'receipt_path':str(run/'stage_closeout.json'),'receipt_sha256':plans.sha(run/'stage_closeout.json'),
            'result_receipt_path':str(receipt_path),
            'note':'Compatibility completed; not final model'if plan['mode']=='pilot'else'Development trial completed; selection pending'if plan['mode']=='trial'else'All-layer training complete; independent serving/game audit pending'})
        live(job)
        if display is not None:display(job)
        return receipt
    except BaseException as exc:
        cleanup_error=None
        if slot.process is not None and not cleanup_attempted:
            cleanup_attempted=True
            try:owned.cleanup(slot.process);cleaned=True
            except BaseException as failure:cleanup_error=f'{type(failure).__name__}: {failure}'
        job.update({'status':'failed','note':f'{type(exc).__name__}: {exc}','cleanup_confirmed':cleaned,'cleanup_error':cleanup_error,
            'cleanup_uncertain_lock_retained':slot.process is not None and not cleaned,'final_model_complete':False});live(job)
        if run is not None:once(run/'supervisor_failure.json',job)
        if run is not None and not (plans.ROOT/'stopped_after_failure.json').exists():
            once(plans.ROOT/'stopped_after_failure.json',{'stage':plan['stage'],'plan_sha256':plan_sha,
                'run_dir':str(run),'error':f'{type(exc).__name__}: {exc}','cleanup_confirmed':cleaned,
                'scope':'No automatic continuation, retry or new rank after a failed owned stage.'})
        raise
    finally:
        if slot.process is None or cleaned:
            try:
                path=LOCK/'owner.json'
                if json.loads(path.read_text()).get('owner')==owner:path.unlink();LOCK.rmdir()
            except (OSError,ValueError):pass


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--execute',action='store_true');parser.add_argument('--prefect',action='store_true')
    args=parser.parse_args()
    if not args.execute:raise RuntimeError('Explicit reviewed exact-hash --execute required')
    def interrupted(signum,frame):raise InterruptedError(f'Signal {signum}')
    for sig in(signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupted)
    if args.prefect:
        from prefect import flow
        from prefect.context import get_run_context
        from prefect.artifacts import create_progress_artifact,update_progress_artifact
        @flow(name='research4-all-layers-v2-stage',retries=0,log_prints=True)
        def flow_stage():
            try:artifact=create_progress_artifact(key='research4-all-layers-v2',progress=0,description='Separate compatibility/trial/final stage')
            except Exception:artifact=None
            last=[-1]
            def display(job):
                if artifact is None or job['completed_steps']==last[0]:return
                try:
                    update_progress_artifact(artifact,progress=100*job['completed_steps']/job['total_steps'],
                        description=f"{job['stage_kind']} {job['stage']}; rank{job['rank']};64layers; final_model_complete={job['final_model_complete']}")
                    last[0]=job['completed_steps']
                except Exception:pass
            return supervise(args.plan,args.plan_sha256,str(get_run_context().flow_run.id),display)
        print(json.dumps(flow_stage(),indent=2))
    else:print(json.dumps(supervise(args.plan,args.plan_sha256),indent=2))
