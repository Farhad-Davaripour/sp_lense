"""Compact tiny-autograd control and independent provider-blocked saved judge."""
from contextlib import contextmanager
import copy,json,subprocess,sys,time,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
BLOCKED={'transformers','transformer_lens','pyarrow','datasets','safetensors','tokenizers'}
class NoProviders:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in BLOCKED:raise RuntimeError('FORBIDDEN_PROVIDER_IMPORT')
sys.meta_path.insert(0,NoProviders())
def audit_hook(event,args):
    if event in ('socket.connect','socket.bind','urllib.Request'):raise RuntimeError('NETWORK_FORBIDDEN')
    if event=='open' and args and isinstance(args[0],str) and args[0].lower().endswith(('.safetensors','.pt','.pth','.ckpt')):raise RuntimeError('CHECKPOINT_FORBIDDEN')
sys.addaudithook(audit_hook)
def need(ok,code):
    if not ok:raise RuntimeError(code)

def saved_audit():
    BLOCKED.add('torch')
    import audit_saved
    base=Path(sys.argv[2]).resolve();need(base.is_relative_to((HERE/'test_evidence').resolve()),'TEST_ONLY_PATH')
    audit_saved.read_inputs=lambda:json.loads((base/'TEST_INPUTS.json').read_bytes())
    execution=json.loads((base/'WORKER_RESULT.json').read_bytes())['execution']
    result=audit_saved.judge(base,execution,time.monotonic()+45)
    print(json.dumps(result,sort_keys=True));return 0

def main():
    import torch,support
    from science import BASE,score
    from core import DispatchLatch,ForwardDerivativeGuard,trace
    from counts import Counts
    from receiver import NativeReceiver
    from input_reader import read
    from workflow import execute,all_unrun,preflight_path
    torch.set_num_threads(1);real_inputs=read()
    params=json.loads((BASE/'real_attempt/fitted_parameters.json').read_bytes())['parameters']
    template=torch.tensor([a+b for a,b in zip(params['grand_mean'],params['direction'],strict=True)],dtype=torch.float32)
    root=HERE/'test_evidence'/('positive_'+str(time.time_ns()));root.mkdir(parents=True,exist_ok=False)
    fixtures=root/'fixtures';fixtures.mkdir();offset=torch.zeros(1024);offset[0]=-.05
    def head(h):
        z=torch.full((248320,),-100.,dtype=torch.float32);z[48964]=0.;z[50057]=.3+10*(h[0]-template[0]);return z
    baseline_z=head(template);seed_h=template+offset;seed_z=head(seed_h)
    def rawz(z):return z.numpy().astype('<f4',copy=False).tobytes()
    def pin(name,raw):
        path=fixtures/name;path.write_bytes(raw)
        return {'path':path.relative_to(support.ROOT).as_posix(),'bytes':len(raw),'sha256':support.sha(raw),'commit':'TINY_SYNTHETIC_NOT_GIT'}
    baseline={'h':template.tolist(),**score(torch,baseline_z,baseline_z),'preserve_label':'KEEP'}
    seed={'h':seed_h.tolist(),'h0':template.tolist(),'offset':offset.tolist(),**score(torch,seed_z,baseline_z),'preserve_label':'KEEP'}
    pins={'baseline_row':pin('baseline.json',support.json_bytes(baseline)),
        'baseline_logits':pin('baseline.f32',rawz(baseline_z)),
        'endpoint_row':pin('seed.json',support.json_bytes(seed)),
        'endpoint_logits':pin('seed.f32',rawz(seed_z)),
        'step_1':pin('historical_step.json',support.json_bytes({'path_norm':.075}))}
    class Block(torch.nn.Module):
        def __init__(self):super().__init__();self.scale=torch.nn.Parameter(torch.ones(1))
        def forward(self,x):return x*self.scale
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.model=torch.nn.Module();self.model.rope_deltas=None
            self.model.language_model=torch.nn.Module();self.model.language_model.layers=torch.nn.ModuleList([Block() for _ in range(24)]);self.eval()
        def forward(self,*,input_ids,attention_mask,logits_to_keep,**kwargs):
            h=template.reshape(1,1,1024).expand(1,input_ids.shape[1],1024).clone()
            for layer in self.model.language_model.layers:h=layer(h)
            z=torch.full((1,1,248320),-100.,dtype=torch.float32);z[:,:,48964]=0.;z[:,:,50057]=.3+10*(h[:,-1,0]-template[0])
            return types.SimpleNamespace(logits=z,past_key_values=None)
    def audit_child(base):return subprocess.run([sys.executable,'-B',str(HERE/'test_positive_control.py'),'audit',str(base)],capture_output=True,text=True,timeout=50)
    reports=[];old_output=support.output
    try:
        for mode,wanted in (('success','COMPLETE_NATIVE_DEVELOPMENT'),('seed_mismatch','SCIENTIFIC_FAILURE_NATIVE_DEVELOPMENT'),
                            ('zero_reset','INCONCLUSIVE_NATIVE_DEVELOPMENT'),('seed_cap','INCONCLUSIVE_NATIVE_DEVELOPMENT')):
            base=root/mode;base.mkdir();support.output=lambda:base;inputs=copy.deepcopy(real_inputs)
            for case in inputs['cases']:case['seed_artifacts']=copy.deepcopy(pins)
            if mode in ('seed_mismatch','seed_cap'):
                wrong=copy.deepcopy(seed)
                if mode=='seed_mismatch':wrong['h'][0]+=.001
                else:wrong['offset'][0]=-.5
                inputs['cases'][0]['seed_artifacts']['endpoint_row']=pin(mode+'.json',support.json_bytes(wrong))
            support.write_new('TEST_INPUTS.json',inputs)
            execution={'scope':'TINY_SYNTHETIC_ONLY','mode':mode,'production_authorized':False}
            deadline=time.monotonic()+60;counts=Counts(deadline,support.write_new);counts.reserve('load')
            latch=DispatchLatch();guard=ForwardDerivativeGuard(Tiny,counts,latch,deadline);guard.install();receiver=NativeReceiver(Tiny(),guard)
            if mode=='zero_reset':
                original_forward=receiver.forward_inputs
                def reset_forward(ids,mask,phase,delta):return original_forward(ids,mask,phase,torch.zeros_like(delta) if phase=='gradient_1' else delta)
                receiver.forward_inputs=reset_forward
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
            try:result=execute(receiver,torch,counts,inputs,lambda name,value:support.write_new(name,value,raw=type(value) is bytes),traced,deadline)
            finally:guard.restore()
            result.update(execution=execution,guard_restored=not guard.installed,dispatch={'forwards':guard.forwards,'derivatives':guard.derivatives,'rejected':guard.rejected})
            support.write_new('WORKER_RESULT.json',result,critical=True)
            files=[]
            for path in sorted(base.rglob('*')):
                if path.is_file():
                    raw=path.read_bytes();files.append({'path':path.relative_to(base).as_posix(),'bytes':len(raw),'sha256':support.sha(raw)})
            support.write_new('CLOSED_WORKER_BINDING.json',{'execution':execution,'good_capture':True,'files':files,
                'worker_result_sha256':support.sha((base/'WORKER_RESULT.json').read_bytes())},critical=True)
            child=audit_child(base)
            if child.returncode:print(child.stdout);print(child.stderr);raise RuntimeError('SAVED_AUDIT_CHILD_'+mode.upper())
            audited=json.loads(child.stdout);need(audited['classification']==wanted,'EXPECTED_'+mode.upper())
            if mode=='success':
                need(audited['flips']==2 and audited['retentions']==0 and audited['off_identities']==0,'TWO_CONTROLLED_P_FLIPS')
                for request in result['requests']:
                    need(request['path_norm']>request['actual_seed_cost'] and request['historical_C_path_norm']==.075,'LIVE_SEED_DEBIT_SEPARATE_C_PATH')
            else:
                need(audited['unrun']>0 and not audited['scientific_pass'],'FIRST_FAILURE_UNRUN')
                if mode=='seed_mismatch':need((base/'seeds/stop_then_keep.json').exists() and (base/'logits/stop_then_keep__P__seed.f32').exists(),'FAILED_SEED_BYTES_PRESERVED')
                if mode=='zero_reset':
                    partial=json.loads((base/'rows/stop_then_keep__P__gradient_1.json').read_bytes())
                    need(partial['derivative_failure']=='GRADIENT_STAGE_FAILURE' and counts.attempts['derivative']==0,'RESET_REJECTED_BEFORE_DERIVATIVE')
            reports.append({'case':mode,'status':'PASS','audit':audited,'counts':counts.record(),'directory':str(base)})
        # Coherent outer inventory changes cannot hide omitted seed cost or a reset.
        base=root/'success';binding_path=base/'CLOSED_WORKER_BINDING.json';original_binding=binding_path.read_bytes()
        for name,change in (('seeds/stop_then_keep.json',lambda v:v.update(actual_seed_cost=0.)),
                            ('rows/stop_then_keep__P__gradient_1.json',lambda v:v.update(offset=[0.]*1024))):
            target=base/name;original=target.read_bytes();value=json.loads(original);change(value);changed=support.json_bytes(value);target.write_bytes(changed)
            binding=json.loads(original_binding);item=next(v for v in binding['files'] if v['path']==name);item.update(bytes=len(changed),sha256=support.sha(changed));binding_path.write_bytes(support.json_bytes(binding))
            child=audit_child(base);need(child.returncode!=0,'INDEPENDENT_COHERENT_TAMPER_REJECTED')
            target.write_bytes(original);binding_path.write_bytes(original_binding);reports.append({'case':'independent_tamper_'+name,'status':'PASS'})
        # Update every affected inventory hash, including the worker-result join.
        # Expected finite rejection codes ensure these exercise the new rule first.
        def coherent_rejection(label,changes,code):
            originals={name:(base/name).read_bytes() for name in changes};binding=json.loads(original_binding)
            try:
                for name,change in changes.items():
                    value=json.loads(originals[name]);change(value);changed=support.json_bytes(value);(base/name).write_bytes(changed)
                    item=next(v for v in binding['files'] if v['path']==name);item.update(bytes=len(changed),sha256=support.sha(changed))
                    if name=='WORKER_RESULT.json':binding['worker_result_sha256']=support.sha(changed)
                binding_path.write_bytes(support.json_bytes(binding));child=audit_child(base)
                need(child.returncode!=0 and code in child.stderr,'TARGETED_REJECTION_'+label)
            finally:
                for name,raw in originals.items():(base/name).write_bytes(raw)
                binding_path.write_bytes(original_binding)
            reports.append({'case':label,'status':'PASS','coherent_outer_inventory':True,'rejection':code})
        first='stop_then_keep'
        def cell_change(phase,**fields):
            cell_id=first+'__baseline' if phase=='baseline' else first+'__P__'+phase
            return lambda value:next(s for s in value['cells'] if s['id']==cell_id).update(**fields)
        for phase in ('baseline','entry','seed','endpoint'):
            coherent_rejection('mandatory_'+phase+'_cannot_skip',{'WORKER_RESULT.json':cell_change(phase,status='SKIPPED',reason='accepted')},'MANDATORY_CELL_NOT_SKIPPABLE')
        coherent_rejection('request_entry_cannot_alias_baseline',{'WORKER_RESULT.json':lambda v:v['requests'][0].update(entry=first+'__baseline')},'EXACT_CASE_REQUEST_REFERENCES')
        coherent_rejection('entry_current_id_is_mandatory',{'rows/'+first+'__P__entry.json':lambda v:v.pop('current_id')},'UNCONDITIONAL_FRESH_ENTRY_BINDING')
        coherent_rejection('no_continuation_after_acceptance',{'WORKER_RESULT.json':cell_change('gradient_2',status='COMPLETE')},'NO_UPDATE_AFTER_EARLIEST_STOP')
        coherent_rejection('wrong_skip_reason',{'WORKER_RESULT.json':cell_change('gradient_2',reason='quality_failure')},'EARLIEST_STOP_SKIP_REASON')
        coherent_rejection('unpaired_skip_reason',{'WORKER_RESULT.json':cell_change('step_2',reason='quality_failure')},'PAIRED_UPDATE_SKIP_SUFFIX')
        coherent_rejection('wrong_request_stop_reason',{'WORKER_RESULT.json':lambda v:v['requests'][0].update(stop_reason='max_updates')},'REQUEST_EARLIEST_STOP_REASON')
        # A verified invalid step must stop too, even when later claimed cells exist.
        step_name='rows/'+first+'__P__step_1.json';logit_name='logits/'+first+'__P__step_1.f32'
        original_logits=(base/logit_name).read_bytes();changed_z=torch.frombuffer(bytearray(original_logits),dtype=torch.float32).clone();changed_z[99]=20.
        changed_logits=rawz(changed_z);baseline_bytes=(base/'logits'/ (first+'__baseline.f32')).read_bytes()
        original_step=(base/step_name).read_bytes();original_worker=(base/'WORKER_RESULT.json').read_bytes();binding=json.loads(original_binding)
        try:
            changed_step=json.loads(original_step);changed_step.update(score(torch,changed_z,torch.frombuffer(bytearray(baseline_bytes),dtype=torch.float32)))
            changed_step['logits_sha256']=support.sha(changed_logits)
            worker_value=json.loads(original_worker);cell_change('gradient_2',status='COMPLETE')(worker_value)
            changed={step_name:support.json_bytes(changed_step),logit_name:changed_logits,'WORKER_RESULT.json':support.json_bytes(worker_value)}
            for name,raw in changed.items():
                (base/name).write_bytes(raw);item=next(v for v in binding['files'] if v['path']==name);item.update(bytes=len(raw),sha256=support.sha(raw))
            binding['worker_result_sha256']=support.sha(changed['WORKER_RESULT.json']);binding_path.write_bytes(support.json_bytes(binding))
            child=audit_child(base);need(child.returncode!=0 and 'NO_UPDATE_AFTER_EARLIEST_STOP' in child.stderr,'QUALITY_FAILURE_CONTINUATION_REJECTED')
        finally:
            (base/step_name).write_bytes(original_step);(base/logit_name).write_bytes(original_logits)
            (base/'WORKER_RESULT.json').write_bytes(original_worker);binding_path.write_bytes(original_binding)
        reports.append({'case':'no_continuation_after_verified_quality_failure','status':'PASS','coherent_outer_inventory':True})
        unchanged=[]
        for name in ('receiver.py','science.py','core.py','entry.py','owned_production.py','production_run.py','setup_budget.py','forward_trace.py','launch.py'):
            raw=subprocess.check_output(['git','-C',str(support.ROOT),'cat-file','blob','6f88bf2b2d49fa068c737372c2202a7b3b79643b:development/native_opposite_order_v1/'+name])
            need((HERE/name).read_bytes()==raw,'INHERITED_EXACT_CODE');unchanged.append(name)
        need(len(all_unrun())==25 and counts.record()['limits']=={'load':1,'forward':25,'derivative':8},'EXACT_25F_8D_DENOMINATOR')
        reports.append({'case':'unchanged_native_method_ownership_and_25_cell_budget','status':'PASS','byte_identical_files':unchanged})
        unit_h0=torch.zeros(1024);unit_h0[0]=1.;unit_next=torch.zeros(1024);unit_next[1]=.02
        try:preflight_path(torch,unit_h0.tolist(),unit_h0.tolist(),unit_next,.19)
        except ValueError:pass
        else:raise RuntimeError('NEXT_STEP_PATH_OVERDRAFT_NOT_REJECTED')
        need(abs(preflight_path(torch,unit_h0.tolist(),unit_h0.tolist(),unit_next,.17)-.02)<1e-6,'FEASIBLE_NEXT_STEP_UNCHANGED')
        reports.append({'case':'remaining_path_preflight_rejects_without_clipping_recipe','status':'PASS'})
        child=subprocess.run([sys.executable,'-B',str(HERE/'launch.py'),'--preflight','--approved-release-sha256','0'*64],capture_output=True,text=True,timeout=10)
        need(child.returncode==2 and json.loads(child.stdout)['status']=='DISABLED_NO_ROOT_RELEASE','REAL_ENTRY_DISABLED')
        reports.append({'case':'real_entry_disabled_before_provider','status':'PASS'})
        need(not any(n.split('.')[0] in BLOCKED for n in sys.modules),'NO_PROVIDER_IMPORTED')
    finally:support.output=old_output
    total=sum(p.stat().st_size for p in root.rglob('*') if p.is_file());need(total<32*1024**2,'COMPACT_TEST_SUITE_32MIB')
    result={'status':'PASS','cases':reports,'total_bytes':total,'providers_imported':False,'checkpoint_tensor_access':False,
        'actual_qwen_loads':0,'tiny_autograd_only':True,'real_authorized':False}
    (HERE/'TEST_POSITIVE_CONTROL_RESULT.json').write_bytes(support.json_bytes(result));print(json.dumps(result,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(saved_audit() if len(sys.argv)>1 and sys.argv[1]=='audit' else main())
