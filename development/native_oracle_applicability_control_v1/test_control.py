"""Focused fake-model interface/authority/judge checks; no real provider or tokenizer."""
import copy,json,subprocess,sys,time,types
from contextlib import contextmanager
from pathlib import Path
HERE=Path(__file__).resolve().parent
BLOCKED={'transformers','tokenizers','transformer_lens','datasets','pyarrow','safetensors'}
class NoProviders:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in BLOCKED:raise RuntimeError('FORBIDDEN_REAL_PROVIDER')
sys.meta_path.insert(0,NoProviders())
def guard(event,args):
    if event in ('socket.connect','socket.bind','urllib.Request'):raise RuntimeError('NETWORK_FORBIDDEN')
    if event=='open' and args and isinstance(args[0],str) and args[0].lower().endswith(('.safetensors','.pt','.pth','.ckpt')):raise RuntimeError('CHECKPOINT_FORBIDDEN')
sys.addaudithook(guard)
import support,input_reader
from support import require,sha,json_bytes
def child_audit():
    BLOCKED.add('torch');base=Path(sys.argv[2]).resolve()
    require(base.is_relative_to((HERE/'test_evidence').resolve()),'TEST_ONLY')
    input_reader.read=lambda:input_reader.validate(json.loads((HERE/'inputs.json').read_bytes()))
    from audit_saved import judge
    execution=json.loads((base/'WORKER_RESULT.json').read_bytes())['execution']
    print(json.dumps(judge(base,execution,time.monotonic()+60),sort_keys=True));return 0
def main():
    support.check_freeze();source_sha=sha((HERE/'SOURCE_FREEZE.json').read_bytes())
    data=input_reader.validate(json.loads((HERE/'inputs.json').read_bytes()));reports=[]
    root=HERE/'test_evidence'/('control_'+str(time.time_ns()));root.mkdir(parents=True)
    def rejects(name,fn,code=None):
        try:fn()
        except (ValueError,OSError,KeyError) as e:
            if code:require(str(e)==code,'EXACT_REJECTION_'+name)
            reports.append(name)
        else:raise RuntimeError('NOT_REJECTED_'+name)
    require(len(data['cases'])==1 and data['cases'][0]==input_reader.source_case(),'EXACT_EXPOSED_CASE')
    reports.append('exact_one_case_ids_mask_no_numeric_state')
    for name,mutate in (
        ('input_id_tamper',lambda v:v['cases'][0]['input']['input_ids'].__setitem__(0,1)),
        ('extra_case',lambda v:v['cases'].append(copy.deepcopy(v['cases'][0]))),
        ('length_tamper',lambda v:v['cases'][0]['input'].update(prompt_length=159)),
        ('oracle_missing',lambda v:v.pop('oracle_authority')),
        ('oracle_false',lambda v:v['oracle_authority'].update(external_applicable=False))):
        bad=copy.deepcopy(data);mutate(bad);rejects(name,lambda:input_reader.validate(bad))
    import authority,loader
    rejects('no_root_release',lambda:authority.read_release('0'*64))
    rejects('loader_before_provider_denied',lambda:loader.load(None,None,{},0))
    from workflow import execute,schedule
    bad=copy.deepcopy(data);bad.pop('oracle_authority')
    rejects('workflow_missing_oracle_before_any_forward',lambda:execute(None,None,None,bad,None,None,0),'EXPLICIT_ORACLE_AUTHORITY_REQUIRED')
    from counts import Counts
    for kind,limit in (('load',1),('forward',21),('derivative',8)):
        c=Counts(time.monotonic()+60,lambda *a:None)
        for _ in range(limit):c.reserve(kind)
        rejects('exact_ceiling_'+kind,lambda:c.reserve(kind),'DISPATCH_COUNT')
    from setup_budget import Budget
    b=Budget(0.);b.charge('worker',7.);b.charge('audit',8.)
    require(b.remaining==0 and b.absolute_end==675 and b.substantive_end==660,'ONE_SHARED15')
    reserved=sum(support.GROUP_CAPS.values())+65536
    require(reserved<32*1024**2 and authority.LIMITS['forwards']==21 and authority.LIMITS['derivatives']==8,'FULL_RESERVATION')
    reports.append('full21_schedule_storage_reservation_675s')
    import torch
    from core import DispatchLatch,ForwardDerivativeGuard,trace
    from receiver import NativeReceiver
    from science import BASE
    from production_run import inventory,good_capture
    import production_run
    torch.set_num_threads(1)
    params=json.loads((BASE/'real_attempt/fitted_parameters.json').read_bytes())['parameters']
    # Artificial state guarantees learned OFF, independently of fake answer sign.
    template=torch.tensor([a-b for a,b in zip(params['grand_mean'],params['direction'],strict=True)],dtype=torch.float32)
    class Block(torch.nn.Module):
        def __init__(self):super().__init__();self.scale=torch.nn.Parameter(torch.ones(1))
        def forward(self,x):return x*self.scale
    class Tiny(torch.nn.Module):
        def __init__(self,margin):
            super().__init__();self.margin=margin;self.model=torch.nn.Module();self.model.rope_deltas=None
            self.model.language_model=torch.nn.Module();self.model.language_model.layers=torch.nn.ModuleList([Block() for _ in range(24)]);self.eval()
        def forward(self,*,input_ids,attention_mask,logits_to_keep,**kw):
            h=template.reshape(1,1,1024).expand(1,input_ids.shape[1],1024).clone()
            for layer in self.model.language_model.layers:h=layer(h)
            z=torch.full((1,1,248320),-100.);z[:,:,48964]=0.
            z[:,:,50057]=self.margin+20*(h[:,-1,0]-template[0])
            return types.SimpleNamespace(logits=z,past_key_values=None)
    def run(name,margin):
        base=root/name;base.mkdir();old_output=support.output;old_inventory_output=production_run.output
        support.output=lambda:base;production_run.output=lambda:base
        execution={'scope':'TINY_SYNTHETIC_ONLY','production_authorized':False,'oracle_authority':input_reader.oracle(),'learned_gate_success_claimed':False}
        deadline=time.monotonic()+120;counts=Counts(deadline,support.write_new);counts.reserve('load');latch=DispatchLatch()
        dg=ForwardDerivativeGuard(Tiny,counts,latch,deadline);dg.install();receiver=NativeReceiver(Tiny(margin),dg)
        support.write_new('LOADER_READY.json',{'execution':execution,'native_initial_sha256':receiver.initial_digest,
            'native_initial_buffer_sha256':receiver.initial_buffer_digest,'loading_info':{},'tiny_fixture_not_real_loader':True})
        @contextmanager
        def traced(cell):
            active=trace.Trace(execution,source_sha,{},latch.stop);trace.ACTIVE=active
            try:
                with active.observe('FORWARD_ADAPTER'):yield
            finally:
                try:active.publish(lambda raw:support.write_new('traces/'+cell+'.json',raw,raw=True)['sha256'])
                finally:trace.ACTIVE=None
        try:worked=execute(receiver,torch,counts,data,lambda n,v:support.write_new(n,v,raw=type(v) is bytes),traced,deadline)
        finally:dg.restore()
        worked.update(execution=execution,guard_restored=not dg.installed,dispatch={'forwards':dg.forwards,'derivatives':dg.derivatives,'rejected':dg.rejected})
        support.write_new('WORKER_RESULT.json',worked,critical=True)
        support.write_new('CLOSED_WORKER_BINDING.json',{'execution':execution,'good_capture':True,'files':inventory(),
            'worker_result_sha256':sha((base/'WORKER_RESULT.json').read_bytes())},critical=True)
        support.output=old_output;production_run.output=old_inventory_output
        return base,worked
    def audit(base):return subprocess.run([sys.executable,'-B',str(HERE/'test_control.py'),'audit',str(base)],capture_output=True,text=True,timeout=70)
    completed=[]
    for name,margin in (('keep',.2),('stop',-.2),('ineligible',0.)):
        base,worked=run(name,margin);child=audit(base)
        if child.returncode:print(child.stdout);print(child.stderr);raise RuntimeError('SAVED_JUDGE_'+name)
        a=json.loads(child.stdout);completed.append(a)
        require(a['planned_cells']==21 and a['off_identities']==0 and not a['ordinary_accuracy'],'EXACT_DENOMINATOR')
        require(all(x['route']=='OFF' for x in a['learned_gate_observations']) and a['learned_gate_success_claimed'] is False,'LEARNED_OFF_PRESERVED')
        if margin:
            require(a['scientific_pass'] and a['flips']==a['retentions']==1 and a['completed_forwards']==7
                and a['completed_derivatives']==1 and a['skipped']==14 and a['unrun']==0,'BOTH_POLICIES_FRESH_2COLD')
        else:require(not a['scientific_pass'] and a['scientific_failures']==['FINITE_SELF_ELIGIBILITY'] and a['unrun']==20,'FAILURE_UNRUN')
        reports.append('tiny_'+name+'_learned_off_oracle_on_separate_judge')
    base=root/'keep';bp=base/'CLOSED_WORKER_BINDING.json';workerpath=base/'WORKER_RESULT.json'
    def tamper(name,path,mutate):
        raw=path.read_bytes();boundraw=bp.read_bytes();value=json.loads(raw);mutate(value);path.write_bytes(json_bytes(value))
        bound=json.loads(boundraw);pin=next(p for p in bound['files'] if p['path']==path.relative_to(base).as_posix())
        pin.update(bytes=path.stat().st_size,sha256=sha(path.read_bytes()))
        if path==workerpath:bound['worker_result_sha256']=sha(path.read_bytes())
        bp.write_bytes(json_bytes(bound))
        try:require(audit(base).returncode!=0,'INDEPENDENT_REJECT_'+name)
        finally:path.write_bytes(raw);bp.write_bytes(boundraw)
        reports.append('coherent_'+name+'_rejected')
    tamper('fake_learned_success',workerpath,lambda w:w.update(learned_gate_success_claimed=True))
    baseline=base/'rows'/(input_reader.CASE+'__baseline.json')
    tamper('fake_learned_on',baseline,lambda r:r.update(route='ON'))
    tamper('missing_external_bit',baseline,lambda r:r.pop('external_applicable'))
    entry=base/'rows'/(input_reader.CASE+'__P__entry.json')
    tamper('entry_alias',entry,lambda r:r.update(current_id=input_reader.CASE+'__C__entry'))
    def wrongskip(w):next(c for c in w['cells'] if c['status']=='SKIPPED')['reason']='quality_failure'
    tamper('early_stop_reason',workerpath,wrongskip)
    old_output=support.output;sizebase=root/'size';sizebase.mkdir();support.output=lambda:sizebase
    try:rejects('actual_writer_5mib_overflow',lambda:support.write_new('overflow.bin',b'x'*(5*1024**2+1),raw=True),'FILE_CAP')
    finally:support.output=old_output
    require(not any(sizebase.iterdir()),'NO_OVERSIZE_PUBLICATION')
    capture={'binding_authenticated':True,'quiescent':True,'stop_reason':None,'faults':[],'cleanup_faults':[],
        'stdout_capture_errors':[],'primary_error':None,'threads_joined':True,'pipes_closed':True,'within_absolute_cleanup_deadline':True,
        'exit_proofs':{k:{'exit_code':0} for k in ('actual_worker','launcher')}}
    require(good_capture(capture),'REUSED_CAPTURE_SUCCESS');capture['exit_proofs']['actual_worker']['exit_code']=1
    require(not good_capture(capture),'REUSED_CAPTURE_FAILURE');reports.append('retained_exit_predicate_reused')
    require(len(schedule(data['cases']))==21 and not any(n.split('.')[0] in BLOCKED for n in sys.modules),'NO_PROVIDER')
    receipt={'status':'PASS','groups':[{'name':n,'status':'PASS'} for n in reports],'group_count':len(reports),
        'source_candidate_sha256':source_sha,'tiny_audits':completed,'reservation_bytes':reserved,
        'real_qwen_calls':0,'tokenizer_calls':0,'old_numeric_state_read':False,'real_release_created':False}
    (HERE/'TEST_RESULTS.json').write_bytes(json_bytes(receipt));print(json.dumps(receipt,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(child_audit() if len(sys.argv)>1 and sys.argv[1]=='audit' else main())
