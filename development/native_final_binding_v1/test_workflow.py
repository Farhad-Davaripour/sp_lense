"""Tiny real-autograd workflow tests and separate provider-blocked saved audit."""
from contextlib import contextmanager
import json,subprocess,sys,time,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
BLOCKED={'transformers','transformer_lens','pyarrow','datasets','safetensors','tokenizers'}
class NoProviders:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in BLOCKED:raise RuntimeError('FORBIDDEN_PROVIDER_IMPORT')
sys.meta_path.insert(0,NoProviders())
def audit_hook(event,args):
    if event in ('socket.connect','socket.bind','urllib.Request'):raise RuntimeError('NETWORK_FORBIDDEN')
    if event=='open' and args and isinstance(args[0],str) and args[0].lower().endswith(('.safetensors','.pt','.pth','.ckpt')):
        raise RuntimeError('CHECKPOINT_TENSOR_FORBIDDEN')
sys.addaudithook(audit_hook)
def need(ok,code):
    if not ok:raise RuntimeError(code)
def saved_audit():
    BLOCKED.add('torch')
    from audit_saved import judge
    base=Path(sys.argv[2]).resolve();need(base.is_relative_to((HERE/'test_evidence').resolve()),'TEST_ONLY_PATH')
    execution=json.loads((base/'WORKER_RESULT.json').read_bytes())['execution']
    result=judge(base,execution,time.monotonic()+90)
    print(json.dumps(result,sort_keys=True));return 0
def main():
    import torch
    import support
    from core import DispatchLatch,ForwardDerivativeGuard,trace
    from counts import Counts
    from receiver import NativeReceiver
    from input_reader import read
    from science import BASE
    from workflow import execute
    torch.set_num_threads(1)
    inputs=read();parameters=json.loads((BASE/'real_attempt/fitted_parameters.json').read_bytes())['parameters']
    positive=torch.tensor([a+b for a,b in zip(parameters['grand_mean'],parameters['direction'],strict=True)],dtype=torch.float32)
    negative=torch.tensor([a-b for a,b in zip(parameters['grand_mean'],parameters['direction'],strict=True)],dtype=torch.float32)
    class Block(torch.nn.Module):
        def __init__(self):super().__init__();self.scale=torch.nn.Parameter(torch.ones(1,dtype=torch.float32))
        def forward(self,x):return x*self.scale
    class Tiny(torch.nn.Module):
        def __init__(self,mode):
            super().__init__();self.model=torch.nn.Module();self.model.rope_deltas=None
            self.model.language_model=torch.nn.Module();self.model.language_model.layers=torch.nn.ModuleList([Block() for _ in range(24)])
            self.mode=mode;self.eval()
        def forward(self,*,input_ids,attention_mask,logits_to_keep,**kwargs):
            length=input_ids.shape[1];index=int(input_ids[0,0])-1;case=inputs['cases'][index]
            self_case=case['audit_only']['category']=='self_shutdown'
            template=positive if self_case and self.mode!='wrong_gate' else negative
            h=template.reshape(1,1,1024).expand(1,length,1024).clone()
            for layer in self.model.language_model.layers:h=layer(h)
            z=torch.full((1,1,248320),-100.,dtype=torch.float32)
            base_margin=.2 if index in (0,1,12) else -.2
            if self.mode=='eligibility_failure' and self_case:base_margin=0.
            z[:,:,48964]=0.;z[:,:,50057]=base_margin+20*(h[:,-1,0]-template[0])
            if case['audit_only']['category']=='ordinary':
                gold=case['audit_only']['correct_token_id'];winner=32 if case['case_key']=='fake_ordinary_implication' else gold
                z[:,:,winner]=10.
            if self.mode=='quality_failure' and self_case and bool((h[:,-1,0]-template[0]).detach().abs().any()):z[:,:,99]=20.
            return types.SimpleNamespace(logits=z,past_key_values=None)
    root=HERE/'test_evidence'/('workflow_'+str(time.time_ns()));root.mkdir(parents=True,exist_ok=False)
    reports=[];original_output=support.output
    try:
        for mode,wanted in (('success','COMPLETE_NATIVE_DEVELOPMENT'),('wrong_gate','SCIENTIFIC_FAILURE_NATIVE_DEVELOPMENT'),
                            ('eligibility_failure','SCIENTIFIC_FAILURE_NATIVE_DEVELOPMENT'),('quality_failure','SCIENTIFIC_FAILURE_NATIVE_DEVELOPMENT'),('derivative_failure','INCONCLUSIVE_NATIVE_DEVELOPMENT')):
            base=root/mode;base.mkdir();support.output=lambda:base
            execution={'scope':'TINY_SYNTHETIC_ONLY','mode':mode,'production_authorized':False}
            deadline=time.monotonic()+180;counts=Counts(deadline,support.write_new);counts.reserve('load')
            latch=DispatchLatch();guard=ForwardDerivativeGuard(Tiny,counts,latch,deadline);guard.install()
            receiver=NativeReceiver(Tiny(mode),guard)
            if mode=='derivative_failure':
                def fail_derivative(z):raise RuntimeError('SYNTHETIC_DERIVATIVE_FAILURE')
                receiver.gradient=fail_derivative
            support.write_new('LOADER_READY.json',{'execution':execution,'native_initial_sha256':receiver.initial_digest,
                'native_initial_buffer_sha256':receiver.initial_buffer_digest,'loading_info':{},'tiny_fixture_not_real_loader':True})
            @contextmanager
            def traced(cell):
                active=trace.Trace(execution,'0'*64,{},latch.stop);trace.ACTIVE=active
                try:
                    with active.observe('FORWARD_ADAPTER'):yield
                finally:
                    try:active.publish(lambda raw:support.write_new('traces/'+cell+'.json',raw,raw=True)['sha256'])
                    finally:trace.ACTIVE=None
            def publish(name,value):return support.write_new(name,value,raw=type(value) is bytes)
            try:result=execute(receiver,torch,counts,inputs,publish,traced,deadline)
            finally:guard.restore()
            result.update(execution=execution,guard_restored=not guard.installed,
                dispatch={'forwards':guard.forwards,'derivatives':guard.derivatives,'rejected':guard.rejected})
            support.write_new('WORKER_RESULT.json',result,critical=True)
            files=[]
            for path in sorted(base.rglob('*')):
                if path.is_file():
                    raw=path.read_bytes();files.append({'path':path.relative_to(base).as_posix(),'bytes':len(raw),'sha256':support.sha(raw)})
            support.write_new('CLOSED_WORKER_BINDING.json',{'execution':execution,'good_capture':True,'files':files,
                'worker_result_sha256':support.sha((base/'WORKER_RESULT.json').read_bytes())},critical=True)
            child=subprocess.run([sys.executable,'-B',str(HERE/'test_workflow.py'),'audit',str(base)],capture_output=True,text=True,timeout=100)
            if child.returncode:
                print(child.stdout);print(child.stderr);raise RuntimeError('SAVED_AUDIT_CHILD')
            audited=json.loads(child.stdout)
            need(audited['planned_cells']==180,'EXACT_180_DENOMINATOR')
            need(audited['classification']==wanted,'EXPECTED_'+mode.upper())
            if mode=='success':need(audited['flips']==6 and audited['retentions']==6 and audited['off_identities']==36,'COMPLETE_CONTROL_RETENTION_OFF')
            elif mode=='derivative_failure':
                partial=json.loads((base/'rows'/'fake_n01_self_keep_first__C__gradient_1.json').read_bytes())
                need(partial['derivative_failure']=='GRADIENT_STAGE_FAILURE' and 'gradient' not in partial
                    and (base/'logits'/'fake_n01_self_keep_first__C__gradient_1.f32').exists() and audited['unrun']==144,'FAILED_GRADIENT_ROW_PRESERVED')
            else:need(audited['unrun']>0 and audited['scientific_failures'],'FINITE_STOP_UNRUN')
            reports.append({'case':mode,'status':'PASS','worker_counts':counts.record(),'audit':audited,'directory':str(base)})
        success=json.loads((root/'success'/'WORKER_RESULT.json').read_bytes())
        by_case={p['case_key']:p for p in inputs['cases']}
        coverage={(r['policy'],1 if (r['policy']=='P')==(by_case[r['case']]['layout']=='keep_first') else 2)
            for r in success['requests'] if r['kind']=='flip'}
        need(coverage=={('P',1),('P',2),('C',1),('C',2)},'BOTH_SIGNS_AND_TARGET_POSITIONS')
        need(sum(c['status']=='COMPLETE' and c['phase']=='endpoint' for c in success['cells'])==12,'TWELVE_EXECUTED_COLD_ENDPOINTS')
        need(sum(c['status']=='COMPLETE' and c['phase']=='baseline' for c in success['cells'])==24
            and sum(c['status']=='COMPLETE' and c['phase']=='entry' for c in success['cells'])==48,'ALL72_FRESH_ROUTES')
        ordinary=reports[0]['audit']['ordinary_accuracy']
        need(len(ordinary)==18 and sum(not r['correct'] for r in ordinary)==3,'PRESERVED_WRONG_ORDINARY_NOT_ACCURACY')
        need({p['audit_only']['correct_token_id'] for p in inputs['cases'] if p['audit_only']['category']=='ordinary'}=={32,33},'GOLD_A_AND_B')
        reports.append({'case':'both_signs_first_second_12cold36off_and_goldA_B_preservedwrong','status':'PASS'})
        need(not any(n.split('.')[0] in BLOCKED for n in sys.modules),'NO_PROVIDER_IMPORTED')
        # One coherent outer inventory cannot hide altered science: update the row's
        # ledger pin as well, then demand that independent score reconstruction fails.
        base=root/'success';target=base/'rows'/'fake_n01_self_keep_first__baseline.json'
        original=target.read_bytes();value=json.loads(original);value['preserve_log_odds']+=.125
        target.write_bytes(support.json_bytes(value))
        binding_path=base/'CLOSED_WORKER_BINDING.json';binding_original=binding_path.read_bytes();binding=json.loads(binding_original)
        pin=next(v for v in binding['files'] if v['path']=='rows/fake_n01_self_keep_first__baseline.json');changed=target.read_bytes();pin.update(bytes=len(changed),sha256=support.sha(changed))
        binding_path.write_bytes(support.json_bytes(binding))
        child=subprocess.run([sys.executable,'-B',str(HERE/'test_workflow.py'),'audit',str(base)],capture_output=True,text=True,timeout=100)
        need(child.returncode!=0,'COHERENT_OUTER_TAMPER_REJECTED')
        target.write_bytes(original);binding_path.write_bytes(binding_original)
        reports.append({'case':'coherent_outer_hash_score_tamper','status':'PASS'})
        import copy,struct
        from input_reader import validate
        from workflow import all_unrun
        need(len(all_unrun())==180 and len({c['id'] for c in all_unrun()})==180,'FULL180_SCHEDULE')
        for change,label in ((321,'input_over320'),(157,'input_wrong_exact_length')):
            bad=copy.deepcopy(inputs);v=bad['cases'][0]['input'];v.update(input_ids=[1]*change,attention_mask=[1]*change,prompt_length=change,final_input_index=change-1)
            digest=support.sha(struct.pack('<'+'q'*change,*([1]*change)))
            bad['cases'][0]['input_binding'].update(derived_input_int64_le_sha256=digest,derived_mask_int64_le_sha256=digest)
            try:validate(bad)
            except ValueError as error:need(str(error)==('FULL_INPUT_MAX320_EXACT_LENGTH' if change==321 else 'EXACT_ARTIFICIAL_SCHEMA_ORDER_IDS_GOLD'),'COHERENT_LENGTH_REASON')
            else:raise RuntimeError(label)
            reports.append({'case':label,'status':'PASS'})
        for mutator,reason in ((lambda d:d['cases'].pop(),'EXACT24_CASES'),
            (lambda d:d['cases'].reverse(),'EXACT_ARTIFICIAL_SCHEMA_ORDER_IDS_GOLD'),
            (lambda d:d['cases'][-1]['audit_only'].update(correct_token_id=32),'EXACT_ARTIFICIAL_SCHEMA_ORDER_IDS_GOLD')):
            bad=copy.deepcopy(inputs);mutator(bad)
            try:validate(bad)
            except ValueError as error:need(str(error)==reason,'FIXED_COHORT_REASON')
            else:raise RuntimeError('ALTERED_COHORT_NOT_REJECTED')
        reports.append({'case':'missing_reordered_and_changed_gold_cohort_rejection','status':'PASS'})
        # Exercise the actual receiver's prospective upper boundary with one
        # separately counted tiny forward; no tokenization or production input.
        base=root/'input_boundary';base.mkdir();support.output=lambda:base
        deadline=time.monotonic()+30;counts=Counts(deadline,support.write_new);latch=DispatchLatch()
        guard=ForwardDerivativeGuard(Tiny,counts,latch,deadline);guard.install()
        rejected=NativeReceiver(Tiny('success'),guard);counts.reserve('forward')
        try:
            with traced('synthetic_320_boundary'):z,h=rejected.forward_inputs([1]*320,[1]*320,'baseline',torch.zeros(1024))
            need(rejected.capture['final_input_index']==319 and rejected.capture['nonfinal_positions']==319,'ACTUAL320_ADMISSION')
            rejected.clear_capture()
        finally:guard.restore()
        try:rejected.forward_inputs([1]*321,[1]*321,'baseline',torch.zeros(1024))
        except ValueError:pass
        else:raise RuntimeError('RECEIVER_321_NOT_REJECTED')
        reports.append({'case':'actual_receiver_320_pass_321_reject','status':'PASS'})
        need(sum(support.GROUP_CAPS.values())+64*1024==209068032 and 209068032<288*1024**2,'FULL_STORAGE_RESERVATION')
        counter=Counts(time.monotonic()+30,lambda *a,**k:None)
        for kind,cap in (('load',1),('forward',180),('derivative',48)):
            for _ in range(cap):counter.reserve(kind)
            try:counter.reserve(kind)
            except ValueError:pass
            else:raise RuntimeError('COUNT_OVERRUN')
        reports.append({'case':'180F48D1load_and_full_storage_reservation','status':'PASS','reserved_bytes':sum(support.GROUP_CAPS.values())+64*1024})
        base=root/'success';binding_path=base/'CLOSED_WORKER_BINDING.json'
        support.output=lambda:base
        try:support.write_new('oversize.f32',b'0'*(5*1024**2+1),raw=True)
        except ValueError as error:need(str(error)=='FILE_CAP','ACTUAL_WRITER_SIZE_REASON')
        else:raise RuntimeError('ACTUAL_WRITER_OVERSIZE_NOT_REJECTED')
        need(not (base/'oversize.f32').exists(),'REJECT_BEFORE_PUBLICATION')
        reports.append({'case':'actual_writer_5MiB_overflow_rejection','status':'PASS'})
        outer=binding_path.read_bytes();oversize=base/'oversize.f32';oversize.write_bytes(b'0'*(5*1024**2+1))
        binding=json.loads(outer);binding['files'].append({'path':'oversize.f32','bytes':oversize.stat().st_size,'sha256':support.sha(oversize.read_bytes())})
        binding_path.write_bytes(support.json_bytes(binding))
        try:
            child=subprocess.run([sys.executable,'-B',str(HERE/'test_workflow.py'),'audit',str(base)],capture_output=True,text=True,timeout=100)
            need(child.returncode!=0 and 'INDEPENDENT_FILE_CAP' in child.stderr,'COHERENT_ACTUAL_OVERSIZE_AUDIT_REJECTION')
        finally:oversize.unlink();binding_path.write_bytes(outer)
        reports.append({'case':'coherent_inventory_actual_5MiB_overflow_rejection','status':'PASS'})
        def tamper(label,relative,mutator,code):
            target=base/relative;original=target.read_bytes();outer=binding_path.read_bytes()
            value=json.loads(original);mutator(value);target.write_bytes(support.json_bytes(value))
            binding=json.loads(outer);raw=target.read_bytes()
            pin=next(x for x in binding['files'] if x['path']==relative);pin.update(bytes=len(raw),sha256=support.sha(raw))
            if relative=='WORKER_RESULT.json':binding['worker_result_sha256']=support.sha(raw)
            binding_path.write_bytes(support.json_bytes(binding))
            try:
                child=subprocess.run([sys.executable,'-B',str(HERE/'test_workflow.py'),'audit',str(base)],capture_output=True,text=True,timeout=100)
                need(child.returncode!=0 and code in child.stderr,'EXPECTED_TAMPER_'+label+':'+child.stderr[-1500:])
            finally:target.write_bytes(original);binding_path.write_bytes(outer)
            reports.append({'case':label,'status':'PASS','expected_rejection':code})
        def cell(data,name):return next(c for c in data['cells'] if c['id']==name)
        first='fake_n01_self_keep_first__';second='fake_n02_self_keep_first__'
        tamper('mandatory_entry_skipped','WORKER_RESULT.json',lambda d:cell(d,first+'P__entry').update(status='SKIPPED',reason='accepted'),'MANDATORY_CELL_NOT_SKIPPABLE')
        tamper('entry_alias','WORKER_RESULT.json',lambda d:d['requests'][0].update(entry=first+'baseline'),'EXACT_CASE_POLICY_REQUEST_ENTRY')
        tamper('entry_missing_current','rows/'+first+'P__entry.json',lambda d:d.pop('current_id'),'UNCONDITIONAL_FRESH_ENTRY_BINDING')
        for prefix,step in ((first+'P__',1),(first+'C__',2),(second+'P__',2)):
            tamper('forbidden_continuation_'+prefix,'WORKER_RESULT.json',lambda d,prefix=prefix,step=step:cell(d,prefix+'gradient_'+str(step)).update(status='COMPLETE'),'NO_UPDATE_AFTER_EARLIEST_STOP')
        tamper('wrong_skip_reason','WORKER_RESULT.json',lambda d:cell(d,first+'C__gradient_2').update(reason='quality_failure'),'EARLIEST_STOP_SKIP_REASON')
        tamper('wrong_stop_reason','WORKER_RESULT.json',lambda d:d['requests'][0].update(stop_reason='max_updates'),'REQUEST_EARLIEST_STOP_REASON')
        base=root/'quality_failure';binding_path=base/'CLOSED_WORKER_BINDING.json'
        tamper('quality_failure_forbidden_continuation','WORKER_RESULT.json',lambda d:cell(d,first+'C__gradient_2').update(status='COMPLETE'),'NO_UPDATE_AFTER_EARLIEST_STOP')
        tamper('quality_failure_wrong_skip_reason','WORKER_RESULT.json',lambda d:cell(d,first+'C__gradient_2').update(reason='accepted'),'EARLIEST_STOP_SKIP_REASON')
        base=root/'success';binding_path=base/'CLOSED_WORKER_BINDING.json'
        tamper('ordinary_gold_tamper','rows/fake_ordinary_subtraction__baseline.json',lambda d:d.update(ordinary_correct_token_id=32),'ORDINARY_ACCURACY')
        tamper('preserved_wrong_accuracy_tamper','rows/fake_ordinary_implication__baseline.json',lambda d:d.update(ordinary_correct=True),'ORDINARY_ACCURACY')
        tamper('cold_endpoint_skipped','WORKER_RESULT.json',lambda d:cell(d,first+'P__endpoint').update(status='SKIPPED',reason='accepted'),'MANDATORY_CELL_NOT_SKIPPABLE')
        tamper('cold_endpoint_alias','WORKER_RESULT.json',lambda d:d['requests'][0].update(endpoint=first+'P__entry'),'EXACT_CASE_POLICY_ENDPOINT')
        from authority import read_release
        from loader import load
        for action,label in ((lambda:read_release('0'*64),'SYNTHETIC_BINDING_NO_REAL_RELEASE'),(lambda:load(None,None,None,None),'SYNTHETIC_BINDING_NO_REAL_LOADER')):
            try:action()
            except ValueError as error:need(str(error)==label,'EXACT_DISABLED_PATH')
            else:raise RuntimeError('REAL_WORK_NOT_DISABLED')
        reports.append({'case':'real_authority_and_direct_loader_disabled_before_access','status':'PASS'})
        from setup_budget import Budget
        budget=Budget(100.)
        need(budget.substantive_end==2080. and budget.absolute_end==2095. and budget.deadlines(100.,'worker')[0]==1900.,'FIXED1995_ENVELOPE')
        budget.charge('worker',7.);need(budget.remaining==8.,'ONE_SHARED_CLEANUP');budget.charge('audit',8.)
        try:budget.charge('extra',.1)
        except ValueError:pass
        else:raise RuntimeError('SHARED_CLEANUP_OVERRUN_NOT_REJECTED')
        reports.append({'case':'1800worker180judge_one_shared15_cleanup','status':'PASS'})
        # Actual logical file sizes, not manifest claims:58 individually admissible
        # 5MiB files exceed288MiB. Padding exists only for this negative fixture.
        overflow=root/'total_overflow';overflow.mkdir()
        raw=(root/'success'/'WORKER_RESULT.json').read_bytes();(overflow/'WORKER_RESULT.json').write_bytes(raw)
        zeros=b'\0'*(5*1024**2);padding_sha=support.sha(zeros);pins=[]
        for n in range(58):
            path=overflow/('padding_'+str(n)+'.bin')
            with path.open('xb') as f:f.truncate(len(zeros))
            pins.append({'path':path.name,'bytes':path.stat().st_size,'sha256':padding_sha})
        pins.append({'path':'WORKER_RESULT.json','bytes':len(raw),'sha256':support.sha(raw)})
        (overflow/'CLOSED_WORKER_BINDING.json').write_bytes(support.json_bytes({'execution':success['execution'],'good_capture':True,
            'worker_result_sha256':support.sha(raw),'files':pins}))
        try:
            child=subprocess.run([sys.executable,'-B',str(HERE/'test_workflow.py'),'audit',str(overflow)],capture_output=True,text=True,timeout=100)
            need(child.returncode!=0 and 'INDEPENDENT_TOTAL_CAP' in child.stderr,'ACTUAL_TOTAL_CAP_REJECTION')
        finally:
            for path in overflow.iterdir():path.unlink()
        reports.append({'case':'coherent_inventory_actual_288MiB_total_overflow_rejection','status':'PASS','temporary_padding_removed':True})
        child=subprocess.run([sys.executable,'-B',str(HERE/'launch.py'),'--preflight','--approved-release-sha256','0'*64],capture_output=True,text=True,timeout=10)
        need(child.returncode==2 and json.loads(child.stdout)['status']=='DISABLED_NO_ROOT_RELEASE','REAL_ENTRY_DISABLED')
        reports.append({'case':'real_entry_disabled_before_provider','status':'PASS'})
    finally:support.output=original_output
    total=sum(p.stat().st_size for p in root.rglob('*') if p.is_file());need(total<288*1024**2,'TEST_SUITE_288MIB')
    result={'status':'PASS','cases':reports,'total_bytes':total,'real_model_loads':0,'providers_imported':False,
        'checkpoint_tensor_access':False,'tiny_autograd_only':True,'production_authorized':False,'planned_schedule':all_unrun()}
    (HERE/'TEST_WORKFLOW_RESULT.json').write_bytes(support.json_bytes(result))
    print(json.dumps(result,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(saved_audit() if len(sys.argv)>1 and sys.argv[1]=='audit' else main())
