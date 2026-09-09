"""Focused multi-case applicability/OFF/reporting checks; only tiny fake models."""
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
    print(json.dumps(judge(base,execution,time.monotonic()+90),sort_keys=True));return 0
def main():
    support.check_freeze();source_sha=sha((HERE/'SOURCE_FREEZE.json').read_bytes())
    data=input_reader.validate(json.loads((HERE/'inputs.json').read_bytes()));reports=[]
    root=HERE/'test_evidence'/('panel_'+str(time.time_ns()));root.mkdir(parents=True)
    def rejects(name,fn):
        try:fn()
        except (ValueError,OSError,KeyError):reports.append(name)
        else:raise RuntimeError('NOT_REJECTED_'+name)
    require(data['cases']==input_reader.source_cases() and len(data['cases'])==12,'EXACT12')
    mapping=data['oracle_authority']['applicability'];require(sum(mapping.values())==2 and len(mapping)==12,'TWO_ON_TEN_OFF')
    reports.append('exact12_original_order_2on10off_goldAB')
    mutations=(('missing_case',lambda v:v['cases'].pop()),('reordered_case',lambda v:v['cases'].reverse()),
        ('changed_gold',lambda v:v['cases'][-1]['audit_only'].update(correct_token_id=32)),
        ('changed_input_length',lambda v:v['cases'][0]['input'].update(prompt_length=159)),
        ('self_oracle_off',lambda v:v['oracle_authority']['applicability'].update({input_reader.CASE_KEYS[0]:False})),
        ('control_oracle_on',lambda v:v['oracle_authority']['applicability'].update({input_reader.CASE_KEYS[2]:True})),
        ('oracle_integer_not_bool',lambda v:v['oracle_authority']['applicability'].update({input_reader.CASE_KEYS[0]:1})),
        ('missing_oracle',lambda v:v.pop('oracle_authority')))
    for name,mutate in mutations:
        bad=copy.deepcopy(data);mutate(bad);rejects(name,lambda:input_reader.validate(bad))
    import authority,loader
    rejects('no_root_release_or_loader_authority',lambda:loader.load(None,None,{},0))
    from workflow import execute,schedule
    plan=schedule(data['cases'])
    require(len(plan)==72 and all(c['phase']=='baseline' for c in plan[:12])
        and sum(c['phase']=='entry' for c in plan)==24 and sum(c['phase']=='endpoint' for c in plan)==4
        and sum(c['phase'].startswith('gradient_') for c in plan)==16,'FULL_SCHEDULE')
    from counts import Counts
    for kind,limit in (('load',1),('forward',72),('derivative',16)):
        c=Counts(time.monotonic()+60,lambda *a:None)
        for _ in range(limit):c.reserve(kind)
        rejects('enforced_'+kind+'_ceiling',lambda:c.reserve(kind))
    from setup_budget import Budget
    b=Budget(0.);b.charge('worker',7.);b.charge('audit',8.)
    reserved=sum(support.GROUP_CAPS.values())+65536
    require(b.remaining==0 and b.absolute_end==1005 and b.substantive_end==990
        and authority.LIMITS['total_bytes']==96*1024**2 and reserved<96*1024**2,'ENVELOPE_RESERVATION')
    reports.append('72cells_12baseline_24entry_4cold_1005s_full_reservation')
    import torch
    from core import DispatchLatch,ForwardDerivativeGuard,trace
    from receiver import NativeReceiver
    from science import BASE
    import production_run
    torch.set_num_threads(1)
    params=json.loads((BASE/'real_attempt/fitted_parameters.json').read_bytes())['parameters']
    negative=torch.tensor([a-b for a,b in zip(params['grand_mean'],params['direction'],strict=True)],dtype=torch.float32)
    positive=torch.tensor([a+b for a,b in zip(params['grand_mean'],params['direction'],strict=True)],dtype=torch.float32)
    lookup={tuple(p['input']['input_ids']):i for i,p in enumerate(data['cases'])};require(len(lookup)==12,'UNIQUE_FAKE_INPUT_IDENTITIES')
    class Block(torch.nn.Module):
        def __init__(self):super().__init__();self.scale=torch.nn.Parameter(torch.ones(1))
        def forward(self,x):return x*self.scale
    class Tiny(torch.nn.Module):
        def __init__(self,fail=False):
            super().__init__();self.fail=fail;self.model=torch.nn.Module();self.model.rope_deltas=None
            self.model.language_model=torch.nn.Module();self.model.language_model.layers=torch.nn.ModuleList([Block() for _ in range(24)]);self.eval()
        def forward(self,*,input_ids,attention_mask,logits_to_keep,**kw):
            i=lookup[tuple(input_ids[0].tolist())];case=data['cases'][i];template=negative if i<2 else positive
            h=template.reshape(1,1,1024).expand(1,input_ids.shape[1],1024).clone()
            for layer in self.model.language_model.layers:h=layer(h)
            z=torch.full((1,1,248320),-100.);z[:,:,48964]=0.
            margin=0. if self.fail and i==0 else -.2 if i==0 else .2
            z[:,:,50057]=margin+20*(h[:,-1,0]-template[0])
            if case['audit_only']['category']=='ordinary':
                gold=case['audit_only']['correct_token_id'];winner=65-gold if case['case_key'] in ('O02','O05') else gold
                z[:,:,winner]=10.
            return types.SimpleNamespace(logits=z,past_key_values=None)
    def run(name,fail=False):
        base=root/name;base.mkdir();old_output=support.output;old_inventory=production_run.output
        support.output=lambda:base;production_run.output=lambda:base
        execution={'scope':'TINY_SYNTHETIC_ONLY','production_authorized':False,'oracle_authority':input_reader.oracle(),'learned_gate_success_claimed':False}
        deadline=time.monotonic()+180;counts=Counts(deadline,support.write_new);counts.reserve('load');latch=DispatchLatch()
        dg=ForwardDerivativeGuard(Tiny,counts,latch,deadline);dg.install();receiver=NativeReceiver(Tiny(fail),dg)
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
    def audit(base):return subprocess.run([sys.executable,'-B',str(HERE/'test_panel.py'),'audit',str(base)],capture_output=True,text=True,timeout=100)
    completed=[]
    for name,fail in (('success',False),('ineligible',True)):
        base,worked=run(name,fail);child=audit(base)
        if child.returncode:print(child.stdout);print(child.stderr);raise RuntimeError('SAVED_JUDGE_'+name)
        a=json.loads(child.stdout);completed.append(a)
        require(a['planned_cells']==72,'DENOMINATOR')
        if fail:require(not a['scientific_pass'] and a['scientific_failures']==['FINITE_SELF_ELIGIBILITY'] and a['unrun']==71,'FAILURE_UNRUN')
        else:
            require(a['scientific_pass'] and len(worked['requests'])==24 and a['off_identities']==20 and a['flips']==a['retentions']==2
                and a['completed_forwards']==44 and a['completed_derivatives']==2 and a['skipped']==28 and a['unrun']==0,'FULL_PANEL')
            require(len(a['learned_gate_observations'])==36 and all(r['route']==('OFF' if r['cell'].startswith('N02_self_') else 'ON')
                for r in a['learned_gate_observations']),'LEARNED_ORACLE_DISAGREEMENT_BOTH_WAYS')
            require(len(a['ordinary_accuracy'])==18 and sum(not r['correct'] for r in a['ordinary_accuracy'])==6
                and {r['gold_token_id'] for r in a['ordinary_accuracy']}=={32,33},'PRESERVED_WRONG_AB')
            require({(r['policy'],r['target_position']) for r in a['request_outcomes']}=={('P',1),('P',2),('C',1),('C',2)}
                and {r['policy'] for r in a['request_outcomes'] if r['kind']=='flip'}=={'P','C'},'BOTH_SIGNS_LAYOUTS_POSITION_REPORT')
        reports.append('tiny_'+name+'_separate_judge')
    base=root/'success';bp=base/'CLOSED_WORKER_BINDING.json';workerpath=base/'WORKER_RESULT.json'
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
    tamper('fake_self_learned_on',base/'rows'/(input_reader.CASE_KEYS[0]+'__baseline.json'),lambda r:r.update(route='ON'))
    tamper('off_effective_on',base/'rows'/'O01__P__entry.json',lambda r:r.update(effective_route='ON'))
    tamper('off_foreign_baseline',base/'rows'/'O02__P__entry.json',lambda r:r.update(current_id='O01__baseline'))
    tamper('ordinary_wrong_gold',base/'rows'/'O02__baseline.json',lambda r:r.update(ordinary_correct_token_id=32))
    tamper('preserved_wrong_claimed_correct',base/'rows'/'O02__baseline.json',lambda r:r.update(ordinary_correct=True))
    def wrongskip(w):next(c for c in w['cells'] if c['status']=='SKIPPED')['reason']='quality_failure'
    tamper('early_stop_reason',workerpath,wrongskip)
    # Real file-size checks, using sparse artificial files, never model artifacts.
    large=root/'oversize';large.mkdir();(large/'WORKER_RESULT.json').write_bytes(workerpath.read_bytes())
    for i in range(20):
        with (large/(str(i)+'.bin')).open('xb') as f:f.seek(5*1024**2-1);f.write(b'0')
    rejected=audit(large);require(rejected.returncode!=0 and 'INDEPENDENT_TOTAL_CAP' in rejected.stderr,'ACTUAL96MIB_REJECT')
    reports.append('independent_actual96mib_overflow')
    old_output=support.output;small=root/'writer';small.mkdir();support.output=lambda:small
    try:rejects('actual_writer_file_cap',lambda:support.write_new('oversize',b'x'*(5*1024**2+1),raw=True))
    finally:support.output=old_output
    require(not any(small.iterdir()) and not any(n.split('.')[0] in BLOCKED for n in sys.modules),'NO_PROVIDER_OR_OVERFLOW_WRITE')
    receipt={'status':'PASS','group_count':len(reports),'groups':[{'name':n,'status':'PASS'} for n in reports],
        'source_candidate_sha256':source_sha,'tiny_audits':completed,'reservation_bytes':reserved,
        'real_qwen_calls':0,'tokenizer_calls':0,'old_numeric_state_read':False,'real_release_created':False}
    (HERE/'TEST_RESULTS.json').write_bytes(json_bytes(receipt));print(json.dumps(receipt,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(child_audit() if len(sys.argv)>1 and sys.argv[1]=='audit' else main())
