"""One bounded v3 equivalence worker; preserve v2 stop and original clock."""
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import uuid

import psutil
import game_selected_head_v3_plan as plans
from game_selected_head_v3_compare import compare_gradients,compare_logits,loss_equivalence,parameter_state_sha256
import game_all_layers_layout as layout
import owned_validation_process as owned

LOCK=plans.PROJECT/'.local/validation/qwen38_shared_v1/inference.lock'
STATUS=plans.PROJECT/'.local/monitor/sp_lens_selected_head_v3_status.json'


def once(path,value):
    with path.open('x')as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())


def append(path,value):
    with path.open('a')as stream:stream.write(json.dumps(value,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def live(value):
    STATUS.parent.mkdir(parents=True,exist_ok=True);temporary=STATUS.with_name(STATUS.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x')as stream:json.dump({**value,'updated_epoch':time.time()},stream,allow_nan=False)
    os.replace(temporary,STATUS)


def clock_check(plan):
    clock=plan['original_campaign_clock']
    if psutil.boot_time()!=clock['boot_time']:raise RuntimeError('Original host boot changed; no clock restart')
    if time.time()>=plans.ORIGINAL_DEADLINE or time.monotonic()-clock['started_monotonic']>=clock['global_wall_seconds']:
        raise TimeoutError('Original 22-hour deadline reached; v3 cannot extend it')


def idle_check():
    if sys.platform!='darwin'or os.uname().machine!='arm64':raise RuntimeError('Only the configured local Mac is permitted')
    if (plans.PROJECT/'.local/monitor/local_game_service/service.lock').exists():raise RuntimeError('Resident service is still active')
    names=('local_game_service.py','game_training_worker.py','game_all_layers_worker.py','game_selected_head_v3_worker.py',
        'exact_prefix_training.py','validation_mlx_worker.py','heldout_mlx_worker.py')
    for process in psutil.process_iter(['pid','cmdline']):
        try:
            if process.pid!=os.getpid()and any(str(arg).endswith(names)for arg in(process.info['cmdline']or[])):
                raise RuntimeError('Another model worker is active')
        except psutil.NoSuchProcess:continue
        except psutil.AccessDenied as exc:raise RuntimeError('Cannot establish model ownership')from exc
    if psutil.virtual_memory().available<8*1024**3:raise RuntimeError('Need 8 GiB available before a model load')


def replay_proof(run,plan,plan_sha):
    receipt=json.loads((run/'equivalence_receipt.json').read_text())
    if receipt.get('schema')!='research4.selected-head-equivalence-receipt.v3'or receipt.get('plan_sha256')!=plan_sha:
        raise ValueError('Diagnostic receipt schema/packet binding differs')
    if (receipt.get('status'),receipt.get('rank'),receipt.get('num_layers'),receipt.get('gradient_tensors'),
        receipt.get('diagnostic_optimizer_updates'),receipt.get('eligible_for_tuning_or_serving'),receipt.get('original_deadline_epoch'))!= (
        'passed',4,64,992,1,False,plans.ORIGINAL_DEADLINE):raise ValueError('Diagnostic completion scope differs')
    expected=layout.expected_adapter_layout(plans.MODEL,4,64);replayed=[];snapshots=[]
    vocabulary=json.loads((plans.MODEL/'config.json').read_text())['text_config']['vocab_size']
    for state in(0,1):
        full=json.loads((run/f'state{state}_full_capture.json').read_text());selected=json.loads((run/f'state{state}_selected_capture.json').read_text())
        if full['weight_state_sha256']!=selected['weight_state_sha256']:raise ValueError('Captured weight states differ')
        if parameter_state_sha256(run/f'diagnostic_state{state}_weights.safetensors',expected)!=full['weight_state_sha256']:
            raise ValueError('Saved actual parameter snapshot differs from the compared live state')
        if state==0:
            initial=compare_gradients(run/'diagnostic_state0_weights.safetensors',run/'diagnostic_state0_weights.safetensors',
                expected,plan['gradient_tolerances'])
            if not initial['passed']or any(row['reference_l2']!=0 for row in initial['tensors']if row['name'].endswith('.lora_b')):
                raise ValueError('Saved initial parameter state was not finite with exact zero B')
        expected_count=plan['inputs']['short']['total_tokens']-plan['inputs']['short']['prefix_tokens']
        if (full['target_tokens'],selected['target_tokens'],full['target_logits']['shape'],selected['target_logits']['shape'])!= (
            expected_count,expected_count,[expected_count,vocabulary],[expected_count,vocabulary]):raise ValueError('Exact short target coverage differs')
        for item in(full,selected):
            for key in('gradients','target_logits'):
                if plans.sha(plans.path_for(item[key]['path']))!=item[key]['sha256']:raise ValueError('Saved raw numerical bytes changed')
        gradients=compare_gradients(plans.path_for(full['gradients']['path']),plans.path_for(selected['gradients']['path']),expected,
            plan['gradient_tolerances'],require_zero_a=state==0)
        logits=compare_logits(plans.path_for(full['target_logits']['path']),plans.path_for(selected['target_logits']['path']),full['target_logits']['shape'])
        loss=loss_equivalence(full['mean_nll'],selected['mean_nll'],full['target_tokens'],selected['target_tokens'],plan['gradient_tolerances'])
        if not(gradients['passed']and logits['passed']and loss['passed']):raise ValueError('Raw numerical equivalence replay failed')
        for field in('reference_l2','selected_l2'):
            layers={int(row['name'].split('.layers.')[1].split('.')[0])for row in gradients['tensors']
                if row['name'].endswith('.lora_b')and row[field]>0}
            if layers!=set(range(64)):raise ValueError('Raw saved B gradients do not cover all 64 layers')
        if state and not any(row['name'].endswith('.lora_a')and row['reference_l2']>0 and row['selected_l2']>0 for row in gradients['tensors']):
            raise ValueError('After-update raw gradients did not exercise the A branch')
        snapshots.append(full['weight_state_sha256'])
        replayed.append({'state_index':state,'all_992_gradients_passed':True,'target_logits_passed':True,'loss_passed':True})
    full=json.loads((run/'long_full_forward_capture.json').read_text());selected=json.loads((run/'long_selected_forward_capture.json').read_text())
    update=json.loads((run/'single_update_evidence.json').read_text())
    if snapshots[0]==snapshots[1]or update.get('weight_state_sha256')!=snapshots[1]or update.get('optimizer_step')!=1:
        raise ValueError('The after-update numerical snapshot was unchanged or unbound')
    saved_weights=layout.verify_adapter_tensor_file(run/'diagnostic_state1_weights.safetensors',expected)
    if saved_weights.get('nonzero_lora_b_layer_indices')!=list(range(64)):raise ValueError('Saved updated B weights do not cover all layers')
    if full['weight_state_sha256']!=selected['weight_state_sha256']or full['weight_state_sha256']!=snapshots[1]:
        raise ValueError('Long-forward weight state differs from the single updated snapshot')
    expected_count=plan['inputs']['long']['total_tokens']-plan['inputs']['long']['prefix_tokens']
    if (full['target_tokens'],selected['target_tokens'],full['target_logits']['shape'],selected['target_logits']['shape'])!= (
        expected_count,expected_count,[expected_count,vocabulary],[expected_count,vocabulary]):raise ValueError('Exact long target coverage differs')
    for item in(full,selected):
        if plans.sha(plans.path_for(item['target_logits']['path']))!=item['target_logits']['sha256']:raise ValueError('Long logits bytes changed')
    logits=compare_logits(plans.path_for(full['target_logits']['path']),plans.path_for(selected['target_logits']['path']),full['target_logits']['shape'])
    loss=loss_equivalence(full['mean_nll'],selected['mean_nll'],full['target_tokens'],selected['target_tokens'],plan['gradient_tolerances'])
    if not(logits['passed']and loss['passed']):raise ValueError('Long unchanged-context forward replay failed')
    return {'states':replayed,'long_training_forward_passed':True,'scope':'CPU replay of saved numerical measurements, not a new model run.'}


def supervise(plan_path,plan_sha):
    plan,_,_=plans.read_plan(plan_path,plan_sha);clock_check(plan);idle_check()
    if (plans.ROOT/'stopped_after_failure.json').exists():raise RuntimeError('V3 stopped after failure; no automatic retry')
    LOCK.parent.mkdir(parents=True,exist_ok=True);LOCK.mkdir(exist_ok=False)
    owner=uuid.uuid4().hex;slot=owned.LaunchSlot();cleaned=False;cleanup_attempted=False;run=None
    job={'status':'starting','runtime_version':3,'model_label':'sp-lense-adaptors-all-layers','stage':'equivalence_rank4',
        'dataset_manifest_path':'.local/datasets/sp_lens_game_v1/authorized_management_v1_r4/manifest.json',
        'dataset_manifest_sha256':'b33b8e6ac719cbb902ba3c0dd02bad1bd302610ebba333342dec7dc8abb926d2',
        'stage_kind':'diagnostic','phase':'owned_preflight','rank':4,'num_layers':64,
        'diagnostic_checks_completed':0,'diagnostic_checks_total':3,'diagnostic_optimizer_updates':0,
        'head_window':32,'final_model_complete':False,'original_deadline_epoch':plans.ORIGINAL_DEADLINE}
    started=time.monotonic();minimum=psutil.virtual_memory().available;last_guard=-1;previous_phase=None
    try:
        once(LOCK/'owner.json',{'owner':owner,'pid':os.getpid(),'kind':'selected_head_v3_equivalence','created_epoch':time.time()})
        idle_check();clock_check(plan);plans.ROOT.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(plans.ROOT).free<2*1024**3:raise RuntimeError('Need 2 GiB private disk space for complete raw numerical evidence')
        authorities=plans.ROOT/'authorities';authorities.mkdir(exist_ok=True)
        authority=authorities/'equivalence_rank4.json'
        if authority.exists():raise RuntimeError('Equivalence stage already consumed; no duplicate worker')
        runs=plans.ROOT/'runs';runs.mkdir(exist_ok=True)
        run=runs/('equivalence_rank4_'+str(int(time.time()*1000))+'_'+owner[:8]);run.mkdir(exist_ok=False)
        once(authority,{'owner':owner,'plan_sha256':plan_sha,'run_dir':str(run),'retries':0,'original_deadline_epoch':plans.ORIGINAL_DEADLINE})
        once(run/'request.json',{'plan_path':str(plan_path.resolve()),'plan_sha256':plan_sha,'owner':owner,'plan':plan})
        job['run_dir']=str(run);live(job)
        command=[str(plans.PROJECT/'.local/qwen38-venv/bin/python'),'-u',str(plans.PROJECT/'mac_studio/game_selected_head_v3_worker.py'),
            '--plan',str(plan_path.resolve()),'--plan-sha256',plan_sha,'--run-dir',str(run),'--owner',owner]
        with(run/'worker.log').open('xb')as log:
            pending=owned.launch(slot,command,stdin=subprocess.DEVNULL,stdout=log,cwd=plans.PROJECT)
            if pending:raise InterruptedError('Signal during owned diagnostic spawn')
            job.update({'status':'running','pid':slot.process.pid,'process_created_at':psutil.Process(slot.process.pid).create_time()})
            while True:
                exited=owned.peek_exited(slot.process.pid);observed={}
                if(run/'phase.json').exists():observed=json.loads((run/'phase.json').read_text())
                elapsed=time.monotonic()-started;available=psutil.virtual_memory().available;minimum=min(minimum,available)
                try:rss=psutil.Process(slot.process.pid).memory_info().rss
                except psutil.NoSuchProcess:rss=None
                sample={'observed_epoch':time.time(),'elapsed_seconds':elapsed,'phase':observed.get('phase','worker_start'),
                    'diagnostic_checks_completed':observed.get('diagnostic_checks_completed',0),'diagnostic_checks_total':3,
                    'diagnostic_optimizer_updates':observed.get('diagnostic_optimizer_updates',0),
                    'phase_observed_epoch':observed.get('observed_epoch'),'available_bytes':available,
                    'swap_used_bytes':psutil.swap_memory().used,'process_rss_bytes':rss,
                    'last_reported_mlx_peak_bytes':observed.get('mlx_peak_bytes'),'mlx_measurement_epoch':observed.get('observed_epoch')}
                if elapsed-last_guard>=1 or sample['phase']!=previous_phase or exited:
                    append(run/'guard_memory.jsonl',sample);last_guard=elapsed;previous_phase=sample['phase']
                job.update(sample);job['minimum_available_bytes']=minimum;live(job)
                if elapsed>=plan['max_wall_seconds']:raise TimeoutError('20-minute diagnostic operator deadline')
                clock_check(plan)
                if exited:break
                if available<plan['memory_floor_bytes']:
                    append(run/'guard_memory.jsonl',{**sample,'operator_stop':'available_memory_below_4_GiB'})
                    raise RuntimeError('Available memory below 4 GiB; no tolerance/floor relaxation')
                time.sleep(.25)
            cleanup_attempted=True;code,descendants,pending=owned.cleanup(slot.process);cleaned=True
        if code!=0 or descendants or pending:raise RuntimeError('Diagnostic worker failed/owned descendant observed')
        plans.read_plan(plan_path,plan_sha);job.update({'status':'finalizing','phase':'cpu_saved_evidence_replay'});live(job)
        replay=replay_proof(run,plan,plan_sha)
        closeout={'schema':'research4.supervised-selected-head-equivalence.v3','status':'passed','cleanup_confirmed':True,
            'diagnostic_checks_completed':3,'diagnostic_checks_total':3,'diagnostic_optimizer_updates':1,
            'plan_sha256':plan_sha,'receipt':{'path':plans.rel(run/'equivalence_receipt.json'),'sha256':plans.sha(run/'equivalence_receipt.json')},
            'replay':replay,'minimum_available_bytes':minimum,'original_deadline_epoch':plans.ORIGINAL_DEADLINE,
            'original_campaign':plan['original_campaign'],'original_failure_latch':plan['original_failure_latch'],
            'eligible_for_tuning_or_serving':False,'optimized_long_pilot_verified':False}
        once(run/'equivalence_closeout.json',closeout)
        job.update({'status':'completed','phase':'equivalence_passed_long_backward_still_unverified','stage_complete':True,
            'diagnostic_checks_completed':3,'diagnostic_checks_total':3,'diagnostic_optimizer_updates':1,
            'closeout_path':str(run/'equivalence_closeout.json'),'final_model_complete':False});live(job)
        return closeout
    except BaseException as exc:
        cleanup_error=None
        if slot.process is not None and not cleanup_attempted:
            cleanup_attempted=True
            try:owned.cleanup(slot.process);cleaned=True
            except BaseException as failure:cleanup_error=f'{type(failure).__name__}: {failure}'
        job.update({'status':'failed','error':f'{type(exc).__name__}: {exc}','cleanup_confirmed':cleaned,
            'cleanup_uncertain_lock_retained':slot.process is not None and not cleaned,'cleanup_error':cleanup_error,'final_model_complete':False});live(job)
        if run is not None:
            once(run/'supervisor_failure.json',job)
            if not(plans.ROOT/'stopped_after_failure.json').exists():once(plans.ROOT/'stopped_after_failure.json',job)
        raise
    finally:
        if slot.process is None or cleaned:
            try:
                owner_path=LOCK/'owner.json'
                if json.loads(owner_path.read_text()).get('owner')==owner:owner_path.unlink();LOCK.rmdir()
            except(OSError,ValueError):pass


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:raise RuntimeError('Exact reviewed v3 packet and explicit --execute required')
    def interrupted(signum,frame):raise InterruptedError(f'Signal {signum}')
    for sig in(signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupted)
    print(json.dumps(supervise(args.plan,args.plan_sha256),indent=2))
