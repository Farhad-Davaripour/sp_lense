"""One declared changed-path fake batch; never a real model or gate observation."""
import copy,json,os,time
from pathlib import Path
from core import HERE,ROOT,Budget,Counter,read,require,sha,check_freeze

def synthetic_worker(output):
    from unittest.mock import patch
    import mixed_boundary,run
    from inputs import build_plan
    from synthetic_backend import Toy,prepare_backend,boundary
    plan=build_plan();plan['execution_mode']='SYNTHETIC_ONLY';plan['fixture_scope']='synthetic_complete_two_display_four_request_f04_wiring'
    for p in plan['prompts']:
        p['execution_mode']='SYNTHETIC_ONLY';p['category']='misleading_category';p['expected_route_audit_only']='MISLEADING'
    factory=prepare_backend(plan,'normal')
    Budget(output).write('plan.json',plan)
    deadline=read(output/'RUN_STARTED.json')['deadline_monotonic']
    with patch.object(mixed_boundary,'resolve_choice_boundary',side_effect=boundary):
        result=run.execute(plan,factory,Toy,output,deadline,synthetic=True)
    require(result['real_model_loads']==0 and result['execution_status']=='complete','one complete synthetic traversal')

def pure_checks(plan):
    from admission import admit_production,exact_cohort
    from owned_capture import validate_claim,execution_envelope,OwnedProcess
    from owned import OwnedPair
    records=[];loader_calls=0
    for name,fn in (
        ('disabled_real_authorization',admit_production),
        ('missing_handshake',lambda:validate_claim(b'', 'n')),
        ('invalid_nonce',lambda:validate_claim(b'{"kind":"claim","nonce":"wrong","pid":99}', 'n')),
        ('fake_cannot_enable_real_lane',lambda:execution_envelope({'may_load':True,'nonce':'n','scope':'FAKE_WORK_ONLY'},'worker',HERE/'real_attempt',None)),
        ('real_pin_cannot_enable_fake_lane',lambda:execution_envelope({'may_load':True,'nonce':'n','scope':'FAKE_WORK_ONLY'},'synthetic_worker',HERE/'synthetic','wrong')),
    ):
        error=None
        try:fn();loader_calls+=1
        except (ValueError,KeyError,TypeError) as exc:error=type(exc).__name__+': '+str(exc)
        require(error is not None and loader_calls==0,'admission fails before any loader: '+name)
        records.append({'fixture':name,'status':'PASS','error':error,'loader_calls':loader_calls})
    require(exact_cohort(plan)['forwards']==42,'exact entire f04 cohort positive admission')
    changed=copy.deepcopy(plan);changed['prompts']=changed['prompts'][:1]
    try:exact_cohort(changed)
    except ValueError:records.append({'fixture':'compact_cohort_rejected','status':'PASS','loader_calls':0})
    else:raise ValueError('compact cohort admitted')
    out=HERE/'synthetic/counter';out.mkdir(parents=True,exist_ok=False);counter=Counter(Budget(out),plan['cells'],time.monotonic()+10)
    for cell in plan['cells']:counter.call(cell,lambda:None)
    try:counter.call(plan['cells'][-1],lambda:None)
    except ValueError:records.append({'fixture':'42_cells_no_43rd','status':'PASS','fake_callbacks':counter.attempts})
    else:raise ValueError('43rd forward admitted')
    # Record failure cannot prevent actual-worker-first cleanup (pure retained handles).
    class FakeBudget:
        fault_code=None
        def event(self,*args):raise ValueError('injected recording cap')
        def write(self,*args):raise ValueError('injected recording cap')
        def fault(self,code):self.fault_code=code
    class Handle:
        def __init__(self,tag):self.tag=tag;self.dead=False;self.count=0
        def exited(self):return self.dead
        def terminate(self):self.dead=True;self.count+=1
        def wait(self,t):return self.dead
    facade=OwnedProcess.__new__(OwnedProcess);facade.budget=FakeBudget();facade.events=[];facade.event_threads=[];facade.event_record_errors=[]
    worker,launcher=Handle('actual_retained'),Handle('owned_launcher');pair=OwnedPair(launcher,worker,facade.event);pair.authenticated=True
    pair.stop_and_wait('injected_capture_cap')
    for thread in facade.event_threads:thread.join(1.)
    require(worker.count==1 and launcher.count==1 and facade.event_record_errors and facade.budget.fault_code,'cap fault must not block retained-handle cleanup')
    records.append({'fixture':'recording_cap_preserves_retained_cleanup','status':'PASS','actual_terminations':1,'recording_faults':len(facade.event_record_errors)})
    return records

def main():
    from inputs import build_plan
    from owned_capture import supervise
    from run import final_inventory
    from final_adjudication import finalize
    started=time.monotonic();budget=Budget(HERE);receipt={'status':'INCONCLUSIVE','model_calls':0,'real_gate_scores':0,'tokenizer_calls':0}
    require(not (HERE/'FAKE_BATCH_STARTED.json').exists(),'one fake batch only, no retry')
    try:
        check_freeze();plan=build_plan();require(read(HERE/'batch_usage.json')['known_standard_used_percent']<100,'fresh known usage')
        budget.write('FAKE_BATCH_STARTED.json',{'monotonic':started,'freeze_sha256':sha((HERE/'freeze.json').read_bytes()),'single_batch':True})
        pure=pure_checks(plan);budget.write('pure_changed_paths.json',pure)
        out=HERE/'synthetic/full_pair';out.mkdir(parents=True,exist_ok=False)
        Budget(out).write_bytes('fitted_parameters.json',(HERE/'fitted_parameters.json').read_bytes())
        worker=supervise('synthetic_worker',out,90)
        require(worker['status']=='complete_valid' and worker['quiescent'],'normal integrated owned worker')
        audit=out/'audit';audit.mkdir(exist_ok=False);judged=supervise('synthetic_audit',audit,60)
        require(judged['status']=='complete_valid' and judged['quiescent'],'normal integrated independent audit')
        Budget(out).write('judge_process.json',judged);final=finalize(out,worker,judged);final_inventory(out)
        result=read(out/'judge_results.json');require(final['classification']=='PASS' and result['synthetic_only'],'independent closed synthetic judge')
        require(result['counts']=={'routes':6,'routes_correct':6,'strict_requests':4,'retentions':2,'flips':2,'off_identities':0,'skips':28,'unrun':0},'entire four-request synthetic result')
        require(read(out/'execution_receipt.json')['forward_completed']==14 and result['checks']['strict_hook_weight_cleanup']['checks']==13,'exact changed schedule/hooks')
        deadline=HERE/'synthetic/deadline';deadline.mkdir(exist_ok=False)
        killed=supervise('fake_deadline',deadline,1.)
        ownership=killed['owned_worker']
        require(killed['status']=='INCONCLUSIVE' and killed['quiescent'] and ownership['binding_authenticated']
                and ownership['retained_actual_termination_events']==1 and ownership['actual_worker_exit_code']==125
                and ownership['launcher_exit_code']==125 and not ownership['cleanup_faults'],'integrated deadline actual handle cleanup')
        receipt.update(status='PASS_FAKE_PREPARATION_ONLY',pure_changed_fixtures=len(pure),synthetic_complete_requests=4,
            synthetic_flips=2,synthetic_retentions=2,synthetic_forwards=14,synthetic_derivatives=2,
            normal_owned_process_captures=2,deadline_owned_process_captures=1,independent_closed_judge='PASS',
            first_position_flip_evidence='UNTESTED; synthetic opportunities only',production_authorization=False)
    except BaseException as error:receipt['error']=type(error).__name__+': '+str(error)
    finally:
        receipt['elapsed_seconds']=time.monotonic()-started;budget.write('fake_batch_receipt.json',receipt)
    print(json.dumps(receipt));return 0 if receipt['status']=='PASS_FAKE_PREPARATION_ONLY' else 1
if __name__=='__main__':raise SystemExit(main())
