"""One owned V4 diagnostic, inheriting every original clock/failure bound."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

import psutil
import game_chain_rule_v4_plan as plans
import game_all_layers_layout as layout
from game_chain_rule_v4_compare import verify_native_audit_copy,compare_covectors,exact_native_values
from game_selected_head_v3_compare import compare_gradients,compare_logits,loss_equivalence,parameter_state_sha256
import owned_validation_process as owned

LOCK=plans.PROJECT/'.local/validation/qwen38_shared_v1/inference.lock'
STATUS=plans.PROJECT/'.local/monitor/sp_lens_chain_rule_v4_status.json'


def once(path,value):
    with path.open('x')as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())


def append(path,value):
    with path.open('a')as stream:stream.write(json.dumps(value)+'\n');stream.flush();os.fsync(stream.fileno())


def live(value):
    STATUS.parent.mkdir(parents=True,exist_ok=True);temp=STATUS.with_name(STATUS.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('x')as stream:json.dump({**value,'updated_epoch':time.time()},stream,allow_nan=False)
    os.replace(temp,STATUS)


def clock_check(plan):
    clock=plan['original_campaign_clock']
    if psutil.boot_time()!=clock['boot_time']:raise RuntimeError('Original host boot changed; no implicit clock restart')
    if time.time()>=plans.DEADLINE or time.monotonic()-clock['started_monotonic']>=clock['global_wall_seconds']:
        raise TimeoutError('The original 22-hour deadline expired')


def idle_check():
    if sys.platform!='darwin'or os.uname().machine!='arm64':raise RuntimeError('Only the configured local Mac is permitted')
    if(plans.PROJECT/'.local/monitor/local_game_service/service.lock').exists():raise RuntimeError('Resident model service is active')
    names=('local_game_service.py','game_training_worker.py','game_all_layers_worker.py','game_selected_head_v3_worker.py',
        'game_chain_rule_v4_worker.py','exact_prefix_training.py','validation_mlx_worker.py','heldout_mlx_worker.py')
    for process in psutil.process_iter(['pid','cmdline']):
        try:
            if process.pid!=os.getpid()and any(str(arg).endswith(names)for arg in(process.info['cmdline']or[])):raise RuntimeError('Another model worker is active')
        except psutil.NoSuchProcess:continue
        except psutil.AccessDenied as exc:raise RuntimeError('Cannot establish model ownership')from exc
    if psutil.virtual_memory().available<8*1024**3:raise RuntimeError('Need 8 GiB available before loading')


def replay_proof(run,plan,plan_sha):
    receipt=json.loads((run/'equivalence_receipt.json').read_text())
    if (receipt.get('schema'),receipt.get('status'),receipt.get('plan_sha256'),receipt.get('rank'),receipt.get('num_layers'),
        receipt.get('gradient_tensors'),receipt.get('diagnostic_optimizer_updates'),receipt.get('eligible_for_tuning_or_serving'),
        receipt.get('outer_mx_compile'),receipt.get('execution_policy'))!= (
        'research4.decomposed-chain-rule-equivalence-receipt.v4','passed',plan_sha,4,64,992,1,False,False,plan['execution_policy']):raise ValueError('V4 completion scope/binding differs')
    if (receipt.get('full_context_tokens'),receipt.get('prefix_tokens'),receipt.get('full_hidden_positions_checked'),
        receipt.get('native_hidden_and_covector_dtype_preserved'),receipt.get('vjp_scalar_reported_as_nll'),
        receipt.get('gdn_math_changed'),receipt.get('original_deadline_epoch'))!=(1022,1001,1023,True,False,False,plans.DEADLINE):
        raise ValueError('Full-context/native-dtype/reference scope differs')
    expected=layout.expected_adapter_layout(plans.MODEL,4,64);config=json.loads((plans.MODEL/'config.json').read_text())['text_config']
    shape=(1,1023,config['hidden_size']);logit_shape=[21,config['vocab_size']];snapshots=[]
    import numpy as np
    token_input=json.loads((run/'token_input.json').read_text())
    if (token_input.get('training_index'),token_input.get('prefix_tokens'),token_input.get('padded_width'),len(token_input.get('token_ids',[])))!=(16,1001,1024,1022):
        raise ValueError('Saved full TRAIN token geometry differs')
    input_sha=hashlib.sha256(np.asarray([token_input['token_ids']+[0,0]],dtype='<i4').tobytes()).hexdigest()
    def checked(entry):
        path=plans.path_for(entry['path'])
        if plans.sha(path)!=entry['sha256']:raise ValueError('Saved raw evidence changed')
        return path
    def native_copy(entry,key,audit_key):
        if entry['shape']!=list(shape):raise ValueError('Full native hidden context shape differs')
        return verify_native_audit_copy(checked(entry['native']),key,checked(entry['audit']),audit_key,shape)
    for state in(0,1):
        full=json.loads((run/f'state{state}_head_full_capture.json').read_text());selected=json.loads((run/f'state{state}_head_selected_capture.json').read_text())
        direct=json.loads((run/f'state{state}_direct_selected_capture.json').read_text());reference=json.loads((run/f'state{state}_reference_body_vjp_capture.json').read_text())
        digests={item['weight_state_sha256']for item in(full,selected,direct,reference)}
        if len(digests)!=1:raise ValueError('Within-state parameter values differ')
        if (any(item.get('full_input_tensor_sha256')!=input_sha for item in(full,selected,direct,reference))
            or any(item.get('model_training')is not True or item.get('gdn_training_count')!=48 for item in(full,selected,direct,reference))):
            raise ValueError('Full input or native training mode differs across paths')
        state_digest=next(iter(digests));snapshots.append(state_digest)
        if parameter_state_sha256(run/f'diagnostic_state{state}_weights.safetensors',expected)!=state_digest:raise ValueError('Saved parameter snapshot differs')
        if state==0:
            initial=compare_gradients(run/'diagnostic_state0_weights.safetensors',run/'diagnostic_state0_weights.safetensors',expected,plan['gradient_tolerances'])
            if not initial['passed']or any(row['reference_l2']!=0 for row in initial['tensors']if row['name'].endswith('.lora_b')):raise ValueError('Initial B state is not exact zero')
        hidden_native=full['head_input_native_file'];h_path=checked(hidden_native)
        from game_chain_rule_v4_compare import native_descriptor
        h_descriptor=native_descriptor(h_path,'hidden',shape)
        native_copy(full['detached_hidden'],'hidden','hidden_values')
        if selected['head_input_native_file']!=hidden_native:raise ValueError('Head paths did not use the same saved H')
        for head in(full,selected):
            check=native_copy(head['hidden_covector'],'hidden_cotangent','hidden_gradient')
            if check['native_dtype']!=h_descriptor['dtype']:raise ValueError('Head cotangent native dtype changed')
            if head['target_tokens']!=21 or head['target_logits']['shape']!=logit_shape:raise ValueError('Full target coverage differs')
        covectors=compare_covectors(checked(full['hidden_covector']['audit']),checked(selected['hidden_covector']['audit']),shape,1001,1022,plan['gradient_tolerances'])
        logits=compare_logits(checked(full['target_logits']),checked(selected['target_logits']),logit_shape)
        loss=loss_equivalence(full['mean_nll'],selected['mean_nll'],21,21,plan['gradient_tolerances'])
        if not(covectors['passed']and logits['passed']and loss['passed']):raise ValueError('Raw head/hidden-covector gates failed')
        if reference.get('constant_cotangent_native_file')!=full['hidden_covector']['native']or reference.get('second_target_division')is not False:
            raise ValueError('Body VJP did not bind the full mean-loss stopped cotangent')
        if 'mean_nll'in reference or 'true_mean_nll'in reference or not math.isfinite(reference['vjp_contraction_value_not_nll']):
            raise ValueError('Linear-functional scalar was confused with NLL')
        for item in(direct,reference):
            hidden_check=native_copy(item['recomputed_hidden'],'hidden','hidden_values')
            if hidden_check['native_dtype']!=h_descriptor['dtype']:raise ValueError('Body recompute changed native dtype')
            if not exact_native_values(h_path,checked(item['recomputed_hidden']['native']),'hidden',shape)['passed']:
                raise ValueError('The compared body derivative used different hidden values')
        gradients=compare_gradients(checked(reference['gradients']),checked(direct['gradients']),expected,plan['gradient_tolerances'],state==0)
        direct_loss=loss_equivalence(full['mean_nll'],direct['true_mean_nll'],21,direct['target_tokens'],plan['gradient_tolerances'])
        direct_logits=compare_logits(checked(full['target_logits']),checked(direct['target_logits']),logit_shape)
        if not(gradients['passed']and direct_loss['passed']and direct_logits['passed']):raise ValueError('All-992 chain-rule/direct gates failed')
        for field in('reference_l2','selected_l2'):
            layers={int(row['name'].split('.layers.')[1].split('.')[0])for row in gradients['tensors']if row['name'].endswith('.lora_b')and row[field]>0}
            if layers!=set(range(64)):raise ValueError('Saved B gradients do not cover all layers')
        if state and not any(row['name'].endswith('.lora_a')and row['reference_l2']>0 and row['selected_l2']>0 for row in gradients['tensors']):raise ValueError('After-update A branch was not exercised')
    if snapshots[0]==snapshots[1]:raise ValueError('The two numerical states are not distinct')
    weights=layout.verify_adapter_tensor_file(run/'diagnostic_state1_weights.safetensors',expected)
    if weights['nonzero_lora_b_layer_indices']!=list(range(64)):raise ValueError('Updated B weights do not cover all layers')
    outer=json.loads((run/'eval_outer_full_capture.json').read_text());selected=json.loads((run/'eval_selected_capture.json').read_text())
    if any(item.get('weight_state_sha256')!=snapshots[1]or item.get('model_training')is not False or item.get('gdn_training_count')!=0 or item.get('target_tokens')!=21 for item in(outer,selected)):
        raise ValueError('Final eval pair must use the same updated state and native eval mode')
    eval_loss=loss_equivalence(outer['mean_nll'],selected['mean_nll'],21,21,plan['gradient_tolerances'])
    eval_logits=compare_logits(checked(outer['target_logits']),checked(selected['target_logits']),logit_shape)
    if not(eval_loss['passed']and eval_logits['passed']):raise ValueError('Actual outer/factorized eval forward replay failed')
    return{'state_count':2,'all_992_theta_gradients_replayed':True,'full_hidden_covectors_replayed':True,
        'native_hidden_and_cotangent_dtypes_verified':True,'true_nll_separate_from_vjp_scalar':True,'actual_outer_eval_pair_verified':True,
        'scope':'CPU replay of saved actual measurements; no new model execution.'}


def supervise(plan_path,plan_sha):
    plan,_,_=plans.read_plan(plan_path,plan_sha);clock_check(plan);idle_check()
    if(plans.ROOT/'stopped_after_failure.json').exists():raise RuntimeError('V4 stopped after failure; no automatic retry')
    LOCK.parent.mkdir(parents=True,exist_ok=True);LOCK.mkdir(exist_ok=False)
    owner=uuid.uuid4().hex;slot=owned.LaunchSlot();cleaned=False;cleanup_attempted=False;run=None;started=time.monotonic()
    job={'status':'starting','runtime_version':4,'model_label':'sp-lense-adaptors-all-layers','stage':'chain_rule_equivalence_rank4',
        'stage_kind':'diagnostic','phase':'owned_preflight','rank':4,'num_layers':64,'full_context_tokens':1022,
        'diagnostic_checks_completed':0,'diagnostic_checks_total':3,'diagnostic_optimizer_updates':0,
        'dataset_manifest_path':'.local/datasets/sp_lens_game_v1/authorized_management_v1_r4/manifest.json',
        'dataset_manifest_sha256':'b33b8e6ac719cbb902ba3c0dd02bad1bd302610ebba333342dec7dc8abb926d2',
        'original_deadline_epoch':plans.DEADLINE,'final_model_complete':False,'outer_mx_compile':False}
    minimum=psutil.virtual_memory().available;last_guard=-1;previous_phase=None
    try:
        once(LOCK/'owner.json',{'owner':owner,'pid':os.getpid(),'kind':'chain_rule_v4_equivalence','created_epoch':time.time()})
        idle_check();clock_check(plan);plans.ROOT.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(plans.ROOT).free<4*1024**3:raise RuntimeError('Need 4 GiB private disk for raw native/gradient evidence')
        authorities=plans.ROOT/'authorities';authorities.mkdir(exist_ok=True);authority=authorities/'chain_rule_equivalence_rank4.json'
        if authority.exists():raise RuntimeError('V4 diagnostic authority already consumed')
        runs=plans.ROOT/'runs';runs.mkdir(exist_ok=True);run=runs/('chain_rule_rank4_'+str(int(time.time()*1000))+'_'+owner[:8]);run.mkdir(exist_ok=False)
        once(authority,{'owner':owner,'plan_sha256':plan_sha,'run_dir':str(run),'retries':0,'original_deadline_epoch':plans.DEADLINE})
        once(run/'request.json',{'plan_path':str(plan_path.resolve()),'plan_sha256':plan_sha,'owner':owner,'plan':plan})
        job['run_dir']=str(run);live(job)
        command=[str(plans.PROJECT/'.local/qwen38-venv/bin/python'),'-u',str(plans.PROJECT/'mac_studio/game_chain_rule_v4_worker.py'),
            '--plan',str(plan_path.resolve()),'--plan-sha256',plan_sha,'--run-dir',str(run),'--owner',owner]
        with(run/'worker.log').open('xb')as log:
            pending=owned.launch(slot,command,stdin=subprocess.DEVNULL,stdout=log,cwd=plans.PROJECT)
            if pending:raise InterruptedError('Signal during V4 owned spawn')
            job.update({'status':'running','pid':slot.process.pid,'process_created_at':psutil.Process(slot.process.pid).create_time()})
            while True:
                exited=owned.peek_exited(slot.process.pid);observed=json.loads((run/'phase.json').read_text())if(run/'phase.json').exists()else{}
                elapsed=time.monotonic()-started;available=psutil.virtual_memory().available;minimum=min(minimum,available)
                try:rss=psutil.Process(slot.process.pid).memory_info().rss
                except psutil.NoSuchProcess:rss=None
                sample={'observed_epoch':time.time(),'elapsed_seconds':elapsed,'phase':observed.get('phase','worker_start'),
                    'diagnostic_checks_completed':observed.get('diagnostic_checks_completed',0),'diagnostic_checks_total':3,
                    'diagnostic_optimizer_updates':observed.get('diagnostic_optimizer_updates',0),
                    'phase_observed_epoch':observed.get('observed_epoch'),'available_bytes':available,'swap_used_bytes':psutil.swap_memory().used,
                    'process_rss_bytes':rss,'last_reported_mlx_peak_bytes':observed.get('mlx_peak_bytes')}
                if elapsed-last_guard>=1 or sample['phase']!=previous_phase or exited:
                    append(run/'guard_memory.jsonl',sample)
                    last_guard=elapsed;previous_phase=sample['phase']
                job.update(sample);job['minimum_available_bytes']=minimum;live(job)
                if elapsed>=plan['max_wall_seconds']:raise TimeoutError('30-minute V4 operator deadline')
                clock_check(plan)
                if exited:break
                if available<plan['memory_floor_bytes']:
                    append(run/'guard_memory.jsonl',{**sample,'operator_stop':'available_memory_below_4_GiB'})
                    raise RuntimeError('Available memory below unchanged 4-GiB floor')
                time.sleep(.25)
            cleanup_attempted=True;code,descendants,pending=owned.cleanup(slot.process);cleaned=True
        if code!=0 or descendants or pending:raise RuntimeError('V4 worker failed/owned descendant observed')
        plans.read_plan(plan_path,plan_sha);job.update({'status':'finalizing','phase':'cpu_saved_evidence_replay'});live(job)
        replay=replay_proof(run,plan,plan_sha)
        closeout={'schema':'research4.supervised-chain-rule-equivalence.v4','status':'passed','cleanup_confirmed':True,
            'plan_sha256':plan_sha,'receipt':{'path':plans.rel(run/'equivalence_receipt.json'),'sha256':plans.sha(run/'equivalence_receipt.json')},
            'replay':replay,'original_deadline_epoch':plans.DEADLINE,'prior_failures':plan['prior_failures'],
            'diagnostic_checks_completed':3,'diagnostic_checks_total':3,'diagnostic_optimizer_updates':1,
            'eligible_for_tuning_or_serving':False,'optimized_long_pilot_verified':False}
        once(run/'equivalence_closeout.json',closeout);job.update({'status':'completed','phase':'chain_rule_proof_passed_pilots_still_required',
            'diagnostic_checks_completed':3,'diagnostic_optimizer_updates':1,'stage_complete':True,'closeout_path':str(run/'equivalence_closeout.json')});live(job)
        return closeout
    except BaseException as exc:
        cleanup_error=None
        if slot.process is not None and not cleanup_attempted:
            cleanup_attempted=True
            try:owned.cleanup(slot.process);cleaned=True
            except BaseException as failure:cleanup_error=f'{type(failure).__name__}: {failure}'
        job.update({'status':'failed','error':f'{type(exc).__name__}: {exc}','cleanup_confirmed':cleaned,
            'cleanup_uncertain_lock_retained':slot.process is not None and not cleaned,'cleanup_error':cleanup_error});live(job)
        if run is not None:
            once(run/'supervisor_failure.json',job)
            if not(plans.ROOT/'stopped_after_failure.json').exists():once(plans.ROOT/'stopped_after_failure.json',job)
        raise
    finally:
        if slot.process is None or cleaned:
            try:
                path=LOCK/'owner.json'
                if json.loads(path.read_text()).get('owner')==owner:path.unlink();LOCK.rmdir()
            except(OSError,ValueError):pass


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:raise RuntimeError('Reviewed V4 packet and explicit --execute required')
    def interrupted(signum,frame):raise InterruptedError(f'Signal {signum}')
    for sig in(signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupted)
    print(json.dumps(supervise(args.plan,args.plan_sha256),indent=2))
