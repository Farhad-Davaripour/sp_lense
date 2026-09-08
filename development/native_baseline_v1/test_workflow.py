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
    result=judge(base,execution,time.monotonic()+45)
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
            length=input_ids.shape[1];self_case=length==137
            template=positive if self_case and self.mode!='wrong_gate' else negative
            h=template.reshape(1,1,1024).expand(1,length,1024).clone()
            for layer in self.model.language_model.layers:h=layer(h)
            z=torch.full((1,1,248320),-100.,dtype=torch.float32)
            z[:,:,48964]=0.;z[:,:,50057]=-.2+20*(h[:,-1,0]-template[0])
            if length==40:z[:,:,32]=10.
            if self.mode=='quality_failure' and self_case and bool((h[:,-1,0]-template[0]).detach().abs().any()):z[:,:,99]=20.
            return types.SimpleNamespace(logits=z,past_key_values=None)
    root=HERE/'test_evidence'/('workflow_'+str(time.time_ns()));root.mkdir(parents=True,exist_ok=False)
    reports=[];original_output=support.output
    try:
        for mode,wanted in (('success','COMPLETE_NATIVE_DEVELOPMENT'),('wrong_gate','SCIENTIFIC_FAILURE_NATIVE_DEVELOPMENT'),
                            ('quality_failure','SCIENTIFIC_FAILURE_NATIVE_DEVELOPMENT'),('derivative_failure','INCONCLUSIVE_NATIVE_DEVELOPMENT')):
            base=root/mode;base.mkdir();support.output=lambda:base
            execution={'scope':'TINY_SYNTHETIC_ONLY','mode':mode,'production_authorized':False}
            deadline=time.monotonic()+60;counts=Counts(deadline,support.write_new);counts.reserve('load')
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
            child=subprocess.run([sys.executable,'-B',str(HERE/'test_workflow.py'),'audit',str(base)],capture_output=True,text=True,timeout=50)
            if child.returncode:
                print(child.stdout);print(child.stderr);raise RuntimeError('SAVED_AUDIT_CHILD')
            audited=json.loads(child.stdout)
            need(audited['classification']==wanted,'EXPECTED_'+mode.upper())
            if mode=='success':need(audited['flips']==1 and audited['retentions']==1 and audited['off_identities']==4,'COMPLETE_CONTROL_RETENTION_OFF')
            elif mode=='derivative_failure':
                partial=json.loads((base/'rows'/'self__P__gradient_1.json').read_bytes())
                need(partial['derivative_failure']=='GRADIENT_STAGE_FAILURE' and 'gradient' not in partial
                    and (base/'logits'/'self__P__gradient_1.f32').exists() and audited['unrun']==22,'FAILED_GRADIENT_ROW_PRESERVED')
            else:need(audited['unrun']>0 and audited['scientific_failures'],'FINITE_STOP_UNRUN')
            reports.append({'case':mode,'status':'PASS','worker_counts':counts.record(),'audit':audited,'directory':str(base)})
        need(not any(n.split('.')[0] in BLOCKED for n in sys.modules),'NO_PROVIDER_IMPORTED')
        # One coherent outer inventory cannot hide altered science: update the row's
        # ledger pin as well, then demand that independent score reconstruction fails.
        base=root/'success';target=base/'rows'/'self__baseline.json'
        original=target.read_bytes();value=json.loads(original);value['preserve_log_odds']+=.125
        target.write_bytes(support.json_bytes(value))
        binding_path=base/'CLOSED_WORKER_BINDING.json';binding_original=binding_path.read_bytes();binding=json.loads(binding_original)
        pin=next(v for v in binding['files'] if v['path']=='rows/self__baseline.json');changed=target.read_bytes();pin.update(bytes=len(changed),sha256=support.sha(changed))
        binding_path.write_bytes(support.json_bytes(binding))
        child=subprocess.run([sys.executable,'-B',str(HERE/'test_workflow.py'),'audit',str(base)],capture_output=True,text=True,timeout=50)
        need(child.returncode!=0,'COHERENT_OUTER_TAMPER_REJECTED')
        target.write_bytes(original);binding_path.write_bytes(binding_original)
        reports.append({'case':'coherent_outer_hash_score_tamper','status':'PASS'})
        from loader import coverage,UNUSED_MTP
        keys=json.loads((HERE/'CHECKPOINT.json').read_bytes())['checkpoint_keys']
        shapes={k:tuple(v['shape']) for k,v in keys.items() if k not in UNUSED_MTP}
        source='model.language_model.embed_tokens.weight';shapes['lm_head.weight']=shapes[source]
        objects={k:object() for k in shapes};objects['lm_head.weight']=objects[source]
        info={k:[] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs','conversion_errors')}
        proof=coverage(list(objects.items()),shapes,keys,info);need(proof['native_unique_parameters']==473,'EXACT_HEADER_COVERAGE')
        wrong={**keys,'mtp.extra.weight':{'shape':[1]}}
        try:coverage(list(objects.items()),shapes,wrong,info)
        except ValueError:pass
        else:raise RuntimeError('EXTRA_MTP_NOT_REJECTED')
        reports.append({'case':'full_header_alias_coverage_and_extra_mtp_rejection','status':'PASS'})
        child=subprocess.run([sys.executable,'-B',str(HERE/'launch.py'),'--preflight','--approved-release-sha256','0'*64],capture_output=True,text=True,timeout=10)
        need(child.returncode==2 and json.loads(child.stdout)['status']=='DISABLED_NO_ROOT_RELEASE','REAL_ENTRY_DISABLED')
        reports.append({'case':'real_entry_disabled_before_provider','status':'PASS'})
    finally:support.output=original_output
    total=sum(p.stat().st_size for p in root.rglob('*') if p.is_file());need(total<32*1024**2,'TEST_SUITE_32MIB')
    result={'status':'PASS','cases':reports,'total_bytes':total,'real_model_loads':0,'providers_imported':False,
        'checkpoint_tensor_access':False,'tiny_autograd_only':True,'production_authorized':False}
    (HERE/'TEST_WORKFLOW_RESULT.json').write_bytes(support.json_bytes(result))
    print(json.dumps(result,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(saved_audit() if len(sys.argv)>1 and sys.argv[1]=='audit' else main())
