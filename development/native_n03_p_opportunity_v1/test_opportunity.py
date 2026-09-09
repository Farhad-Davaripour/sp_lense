"""Focused prospective opportunity branching tests; no real model or tokenizer."""
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
    root=HERE/'test_evidence'/('opportunity_'+str(time.time_ns()));root.mkdir(parents=True)
    def rejects(name,fn):
        try:fn()
        except (ValueError,OSError,KeyError):reports.append(name)
        else:raise RuntimeError('NOT_REJECTED_'+name)
    require(data['cases']==input_reader.source_cases() and len(data['cases'])==6 and data['requested_policies']==['P'],'EXACT6_P_ONLY')
    mapping=data['oracle_authority']['applicability'];require(sum(mapping.values())==2 and len(mapping)==6,'TWO_ON_FOUR_OFF')
    reports.append('exact6_original_order_2on4off_P_only')
    for name,mutate in (
        ('missing_case',lambda v:v['cases'].pop()),('reordered_case',lambda v:v['cases'].reverse()),
        ('forbidden_C_policy',lambda v:v.update(requested_policies=['P','C'])),
        ('changed_opportunity_rule',lambda v:v.update(opportunity_rule='ALWAYS_EDIT')),
        ('oracle_control_on',lambda v:v['oracle_authority']['applicability'].update({input_reader.CASE_KEYS[2]:True})),
        ('missing_oracle',lambda v:v.pop('oracle_authority'))):
        bad=copy.deepcopy(data);mutate(bad);rejects(name,lambda:input_reader.validate(bad))
    import authority,loader
    rejects('no_real_release_loader_authority',lambda:loader.load(None,None,{},0))
    from workflow import execute,schedule
    plan=schedule(data['cases'])
    require(len(plan)==30 and all(c['phase']=='baseline' for c in plan[:6])
        and all(c['policy']=='P' for c in plan[6:]) and sum(c['phase']=='entry' for c in plan)==6
        and sum(c['phase']=='endpoint' for c in plan)==2 and sum(c['phase'].startswith('gradient_') for c in plan)==8,'FIXED_P_ONLY_SCHEDULE')
    from counts import Counts
    for kind,limit in (('load',1),('forward',30),('derivative',8)):
        c=Counts(time.monotonic()+60,lambda *a:None)
        for _ in range(limit):c.reserve(kind)
        rejects('enforced_'+kind+'_ceiling',lambda:c.reserve(kind))
    from setup_budget import Budget
    b=Budget(0.);b.charge('worker',7.);b.charge('audit',8.);reserved=sum(support.GROUP_CAPS.values())+65536
    require(b.remaining==0 and b.absolute_end==675 and b.substantive_end==660
        and authority.LIMITS['total_bytes']==48*1024**2 and reserved==35991552,'RESOURCE_BINDING')
    reports.append('30cells_6baseline_6Pentries_2cold_675s_35991552bytes')
    import torch
    from core import DispatchLatch,ForwardDerivativeGuard,trace
    from receiver import NativeReceiver
    from science import BASE
    import production_run
    torch.set_num_threads(1)
    params=json.loads((BASE/'real_attempt/fitted_parameters.json').read_bytes())['parameters']
    negative=torch.tensor([a-b for a,b in zip(params['grand_mean'],params['direction'],strict=True)],dtype=torch.float32)
    positive=torch.tensor([a+b for a,b in zip(params['grand_mean'],params['direction'],strict=True)],dtype=torch.float32)
    lookup={tuple(p['input']['input_ids']):i for i,p in enumerate(data['cases'])};require(len(lookup)==6,'UNIQUE_INPUT_IDENTITIES')
    class Block(torch.nn.Module):
        def __init__(self):super().__init__();self.scale=torch.nn.Parameter(torch.ones(1))
        def forward(self,x):return x*self.scale
    class Tiny(torch.nn.Module):
        def __init__(self,margins,technical=False):
            super().__init__();self.margins=margins;self.technical=technical;self.model=torch.nn.Module();self.model.rope_deltas=None
            self.model.language_model=torch.nn.Module();self.model.language_model.layers=torch.nn.ModuleList([Block() for _ in range(24)]);self.eval()
        def forward(self,*,input_ids,attention_mask,logits_to_keep,**kw):
            i=lookup[tuple(input_ids[0].tolist())]
            if self.technical and i==0:raise ValueError('FAKE_TECHNICAL_FORWARD_FAILURE')
            template=negative if i<2 else positive
            h=template.reshape(1,1,1024).expand(1,input_ids.shape[1],1024).clone()
            for layer in self.model.language_model.layers:h=layer(h)
            z=torch.full((1,1,248320),-100.);z[:,:,48964]=0.
            z[:,:,50057]=(self.margins[i] if i<2 else .2)+20*(h[:,-1,0]-template[0])
            return types.SimpleNamespace(logits=z,past_key_values=None)
    def run(name,margins,technical=False):
        base=root/name;base.mkdir();old_output=support.output;old_inventory=production_run.output
        support.output=lambda:base;production_run.output=lambda:base
        execution={'scope':'TINY_SYNTHETIC_ONLY','production_authorized':False,'oracle_authority':input_reader.oracle(),'learned_gate_success_claimed':False}
        deadline=time.monotonic()+120;counts=Counts(deadline,support.write_new);counts.reserve('load');latch=DispatchLatch()
        dg=ForwardDerivativeGuard(Tiny,counts,latch,deadline);dg.install();receiver=NativeReceiver(Tiny(margins,technical),dg)
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
        support.write_new('CLOSED_WORKER_BINDING.json',{'execution':execution,'good_capture':True,'files':production_run.inventory(),
            'worker_result_sha256':sha((base/'WORKER_RESULT.json').read_bytes())},critical=True)
        support.output=old_output;production_run.output=old_inventory
        return base,worked
    def audit(base):return subprocess.run([sys.executable,'-B',str(HERE/'test_opportunity.py'),'audit',str(base)],capture_output=True,text=True,timeout=70)
    completed={};worked_by_name={}
    fixtures=(('zero',(.2,.2),False),('one_first',(-.2,.2),False),('one_second',(.2,-.2),False),
        ('two',(-.2,-.2),False),('invalid',(0.,.2),False),('technical',(.2,.2),True))
    for name,margins,technical in fixtures:
        base,worked=run(name,margins,technical);worked_by_name[name]=worked;child=audit(base)
        if child.returncode:print(child.stdout);print(child.stderr);raise RuntimeError('SAVED_JUDGE_'+name)
        a=json.loads(child.stdout);completed[name]=a;require(a['planned_cells']==30 and not a['ordinary_accuracy'],'DENOMINATOR_NO_ORDINARY')
        if name=='zero':
            require(a['classification']=='NO_NATURAL_P_OPPORTUNITY' and a['scientific_pass'] is False and a['no_opportunity_observed']
                and a['completed_forwards']==6 and a['completed_derivatives']==a['off_identities']==a['flips']==a['retentions']==0
                and a['unrun']==24 and not worked['requests'] and all(c.get('reason')=='NO_NATURAL_P_OPPORTUNITY' for c in worked['cells'][6:]),'VALID_NO_OPPORTUNITY')
        elif name in ('invalid','technical'):
            expected='SCIENTIFIC_FAILURE_NATIVE_DEVELOPMENT' if name=='invalid' else 'INCONCLUSIVE_NATIVE_DEVELOPMENT'
            require(a['classification']==expected and not a['scientific_pass'] and not a['no_opportunity_observed']
                and a['opportunity_decision'] is None and a['unrun']==29,'FAILURE_NOT_NO_OPPORTUNITY')
        else:
            n=2 if name=='two' else 1
            require(a['scientific_pass'] and a['flips']==n and a['retentions']==2-n and a['off_identities']==4
                and a['completed_forwards']==14+2*n and a['completed_derivatives']==n and a['skipped']==16-2*n
                and a['unrun']==0 and len(worked['requests'])==6 and all(r['policy']=='P' for r in worked['requests']),'P_ONLY_COMPLETE')
            targets={r['target_position'] for r in a['request_outcomes'] if r['kind']=='flip'}
            require(targets==({1,2} if n==2 else {1} if name=='one_first' else {2}),'EXACT_NATURAL_FLIP_POSITION')
        reports.append('tiny_'+name+'_independent_branch')
    def tamper(name,fixture,relative,mutate):
        base=root/fixture;path=base/relative;bp=base/'CLOSED_WORKER_BINDING.json'
        raw=path.read_bytes();boundraw=bp.read_bytes();value=json.loads(raw);mutate(value);path.write_bytes(json_bytes(value))
        bound=json.loads(boundraw);pin=next(p for p in bound['files'] if p['path']==relative)
        pin.update(bytes=path.stat().st_size,sha256=sha(path.read_bytes()))
        if relative=='WORKER_RESULT.json':bound['worker_result_sha256']=sha(path.read_bytes())
        bp.write_bytes(json_bytes(bound))
        try:require(audit(base).returncode!=0,'INDEPENDENT_REJECT_'+name)
        finally:path.write_bytes(raw);bp.write_bytes(boundraw)
        reports.append('coherent_'+name+'_rejected')
    tamper('noopp_claimed_success','zero','WORKER_RESULT.json',lambda w:w.update(scientific_pass=True,classification='COMPLETE_NATIVE_DEVELOPMENT'))
    tamper('noopp_fake_OFF_preservation','zero','WORKER_RESULT.json',lambda w:w.update(off_identities=4))
    tamper('noopp_wrong_suffix_reason','zero','WORKER_RESULT.json',lambda w:w['cells'][6].update(reason='accepted'))
    def false_noopp(w):w.update(classification='NO_NATURAL_P_OPPORTUNITY',scientific_pass=False,opportunity_decision=copy.deepcopy(worked_by_name['zero']['opportunity_decision']))
    tamper('natural_STOP_hidden_as_noopp','one_first','WORKER_RESULT.json',false_noopp)
    tamper('invalid_hidden_as_noopp','invalid','WORKER_RESULT.json',false_noopp)
    def abandoned_opportunity(w):
        w.update(requests=[],flips=0,retentions=0,off_identities=0,scientific_pass=False,classification='INCONCLUSIVE_NATIVE_DEVELOPMENT')
        w['cells']=[{**c,'status':'COMPLETE'} for c in plan[:6]]+[{**c,'status':'UNRUN'} for c in plan[6:]]
        w['counts']['attempts']={'load':1,'forward':6,'derivative':0};w['dispatch']={'forwards':6,'derivatives':0,'rejected':0}
    tamper('legitimate_opportunity_unrun_without_failure','one_first','WORKER_RESULT.json',abandoned_opportunity)
    control=input_reader.CASE_KEYS[2];foreign=input_reader.CASE_KEYS[3]
    tamper('OFF_foreign_baseline','one_first','rows/'+control+'__P__entry.json',lambda r:r.update(current_id=foreign+'__baseline'))
    tamper('false_learned_gate_success','two','WORKER_RESULT.json',lambda w:w.update(learned_gate_success_claimed=True))
    large=root/'oversize';large.mkdir();(large/'WORKER_RESULT.json').write_bytes((root/'zero/WORKER_RESULT.json').read_bytes())
    for i in range(10):
        with (large/(str(i)+'.bin')).open('xb') as f:f.seek(5*1024**2-1);f.write(b'0')
    rejected=audit(large);require(rejected.returncode!=0 and 'INDEPENDENT_TOTAL_CAP' in rejected.stderr,'ACTUAL48MIB_REJECT')
    reports.append('independent_actual48mib_overflow')
    old_output=support.output;small=root/'writer';small.mkdir();support.output=lambda:small
    try:rejects('actual_writer_file_cap',lambda:support.write_new('oversize',b'x'*(5*1024**2+1),raw=True))
    finally:support.output=old_output
    require(not any(small.iterdir()) and not any(n.split('.')[0] in BLOCKED for n in sys.modules),'NO_PROVIDER_OR_OVERFLOW_WRITE')
    receipt={'status':'PASS','group_count':len(reports),'groups':[{'name':n,'status':'PASS'} for n in reports],
        'source_candidate_sha256':source_sha,'tiny_audits':completed,'reservation_bytes':reserved,
        'real_qwen_calls':0,'tokenizer_calls':0,'old_numeric_state_read':False,'real_release_created':False}
    (HERE/'TEST_RESULTS.json').write_bytes(json_bytes(receipt));print(json.dumps(receipt,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(child_audit() if len(sys.argv)>1 and sys.argv[1]=='audit' else main())
