"""Focused synthetic interface and tiny worker/separate-judge integration checks."""
import copy,json,os,struct,subprocess,sys,time,types
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
from prep_plan import MODEL,STUDY,HEADER,HEADER_TEXT,END,TEMPLATE_SHA256,slots,operations
def fixture(source_sha):
    cohort=json.loads((support.ROOT/'development/native_final_cohort_v1/author_packet/EMPTY_SCHEMA.json').read_bytes())
    for family in cohort['families']:
        family.update(setting_key='FAKE_SETTING_'+family['id'],mechanism_key='FAKE_MECHANISM_'+family['id'],pairing_notes='FAKE')
        for case in family['cases']:case['scenario']='ARTIFICIAL_MARKER_'+case['id']
    texts=[{'id':s['id'],'prompt':'FAKE_MARKER_'+s['id'],'prompt_sha256':sha(('FAKE_MARKER_'+s['id']).encode())} for s in slots()]
    lock={'schema':'native_final_text_lock.v1','scope':'SYNTHETIC_TEST_ONLY','study':STUDY,'model':MODEL,'cohort':cohort,
        'cohort_sha256':sha(json_bytes(cohort)),'rendered_prompts':texts,'blind_semantic_review_approved':False,
        'final_text_locked':False,'final_execution_binding':input_reader.execution_binding(source_sha)}
    identity={'text_lock_raw_sha256':sha(json_bytes(lock)),'source_freeze_sha256':input_reader.PREPARATION_SOURCE_SHA,
        'dependencies_sha256':input_reader.PREPARATION_DEPENDENCIES_SHA,'tokenizer_pins_sha256':input_reader.TOKENIZER_PINS_SHA}
    records=[];cases=[]
    for i,(s,text) in enumerate(zip(slots(),texts,strict=True)):
        ids=[i+1]+[2]*24+HEADER;mask=[1]*32;render='<FAKE_USER>'+text['prompt']+HEADER_TEXT
        r={'case_key':s['id'],'prompt_sha256':text['prompt_sha256'],'rendered_chat':render,
            'rendered_chat_utf8_sha256':sha(render.encode()),'full_token_ids':ids,'attention_mask':mask,'prompt_length':32,
            'final_input_index':31,'generation_header_start_index':25,'generation_header_suffix_ids':HEADER,
            'final_input_token_id':271,'final_input_token_decoded':'\n\n','assistant_end_token_ids':END,
            'full_suffix_token_ids':{w:[t,*END] for w,t in s['labels'].items()},'content_token_ids':s['labels'],
            'chat_template_sha256':TEMPLATE_SHA256,'enable_thinking':False,'truncation':False,
            'exact_generation_prefix':True,'exactly_one_content_token':True};records.append(r)
        audit={'category':s['category'],'expected_route':'ON' if s['category']=='self_shutdown' else 'OFF'}
        if s['category']=='ordinary':audit['correct_token_id']=32+(i-18)%2
        pack=lambda v:sha(struct.pack('<'+'q'*len(v),*v))
        cases.append({'case_key':s['id'],'family':s['family_id'],'layout':s['layout'],'audit_only':audit,
            'input':{'input_ids':ids,'attention_mask':mask,'prompt_length':32,'final_input_index':31,'token_map':s['labels']},
            'input_binding':{'prompt_sha256':r['prompt_sha256'],'rendered_chat_utf8_sha256':r['rendered_chat_utf8_sha256'],
                'chat_template_sha256':TEMPLATE_SHA256,'derived_input_int64_le_sha256':pack(ids),'derived_mask_int64_le_sha256':pack(mask)}})
    data={'schema_version':'native_final_prepared_inputs_v1','scope':'SYNTHETIC_TEST_ONLY','model':MODEL,'study':STUDY,
        'text_lock_canonical_sha256':sha(json_bytes(lock)),'source_identity':identity,'input_token_ceiling':320,
        'exact_lengths':{s['id']:32 for s in slots()},'cases':cases,'real_model_authorized':False}
    result={'status':'PASS','scope':'SYNTHETIC_TEST_ONLY','retry_allowed':False,'complete_input_publication':True,
        'completed_cases':24,'attempted_operations':313,'completed_operations':313,'planned_operations':313,
        'failed_operations':0,'unrun_operations':0,'real_model_loads':0,'real_model_forwards':0,'real_model_derivatives':0,
        'elapsed_seconds':1.,'lengths':data['exact_lengths'],'inputs_sha256':sha(json_bytes(data))}
    journal=[{'ordinal':i+1,'name':n,'status':status} for i,n in enumerate(operations()) for status in ('STARTED','COMPLETE')]
    return data,lock,records,result,journal
def bundle(base,values,source_sha):
    data,lock,records,result,journal=values;base.mkdir();(base/'preparation').mkdir()
    blobs={'inputs.json':json_bytes(data),'RESULT.json':json_bytes(result),'operations.jsonl':b''.join(json_bytes(x) for x in journal)}
    blobs.update({r['case_key']+'.json':json_bytes(r) for r in records})
    for name,raw in blobs.items():(base/'preparation'/name).write_bytes(raw)
    raw_lock=json_bytes(lock);(base/'TEXT_LOCK.json').write_bytes(raw_lock)
    closure={'schema':'root_preparation_closure.v1','quiescent':True,'exit_code':0,'timed_out':False,'one_shot':True,
        'preparation_result_sha256':sha(blobs['RESULT.json']),'text_lock_sha256':sha(raw_lock),'synthetic_fixture_only':True}
    (base/'PREPARATION_CLOSURE.json').write_bytes(json_bytes(closure))
    release={'source_freeze_sha256':source_sha,'inputs_sha256':sha(blobs['inputs.json']),'text_lock_sha256':sha(raw_lock),
        'preparation_files':{n:sha(b) for n,b in blobs.items()},'preparation_closure_sha256':sha(json_bytes(closure)),
        'synthetic_fixture_only':True}
    (base/'SYNTHETIC_BINDING.json').write_bytes(json_bytes(release));return release
def child_audit():
    BLOCKED.add('torch');base=Path(sys.argv[2]).resolve();require(base.is_relative_to((HERE/'test_evidence').resolve()),'TEST_ONLY')
    bound=json.loads((base.parent/'bundle/SYNTHETIC_BINDING.json').read_bytes())
    input_reader.read=lambda:input_reader.read_bundle(base.parent/'bundle',bound,synthetic=True)
    from audit_saved import judge
    execution=json.loads((base/'WORKER_RESULT.json').read_bytes())['execution']
    print(json.dumps(judge(base,execution,time.monotonic()+90),sort_keys=True));return 0
def main():
    frozen=support.check_freeze();source_sha=sha((HERE/'SOURCE_FREEZE.json').read_bytes())
    root=HERE/'test_evidence'/('binding_'+str(time.time_ns()));root.mkdir(parents=True);reports=[]
    vals=fixture(source_sha);data,lock,records,result,journal=vals
    def validate(v):return input_reader.validate(*v,source_sha,sha(json_bytes(v[1])),synthetic=True)
    require(validate(vals)==data,'EXACT24_PREPARED_INTERFACE');reports.append('exact24_prepared_interface')
    def negative(name,mutate):
        bad=copy.deepcopy(vals);mutate(bad)
        try:validate(bad)
        except (ValueError,KeyError,IndexError):reports.append(name)
        else:raise RuntimeError('NOT_REJECTED_'+name)
    negative('missing_case',lambda v:v[0]['cases'].pop())
    negative('reordered_case',lambda v:v[0]['cases'].reverse())
    negative('under320_changed_exact_length',lambda v:v[0]['cases'][0]['input'].update(prompt_length=31))
    negative('over320',lambda v:v[0]['cases'][0]['input'].update(input_ids=[1]*321,attention_mask=[1]*321,prompt_length=321,final_input_index=320))
    negative('own_gold',lambda v:v[0]['cases'][-1]['audit_only'].update(correct_token_id=32))
    negative('wrong_gate_category',lambda v:v[0]['cases'][0]['audit_only'].update(expected_route='OFF'))
    negative('boundary_tamper',lambda v:v[2][0]['full_suffix_token_ids'].update(KEEP=[50057,198]))
    negative('missing_operation',lambda v:v[4].pop())
    negative('source_timing_lock',lambda v:v[1]['final_execution_binding'].update(source_freeze_sha256='0'*64))
    negative('preparation_source_identity',lambda v:v[0]['source_identity'].update(source_freeze_sha256='0'*64))
    bound=bundle(root/'bundle',vals,source_sha)
    require(input_reader.read_bundle(root/'bundle',bound,synthetic=True)==data,'EXACT_SAVED_BUNDLE')
    try:input_reader.read_bundle(root/'bundle',bound)
    except ValueError:reports.append('synthetic_never_production_admissible')
    else:raise RuntimeError('FAKE_ADMITTED')
    p=root/'bundle/preparation/inputs.json';original=p.read_bytes();p.write_bytes(original+b' ')
    try:input_reader.read_bundle(root/'bundle',bound,synthetic=True)
    except ValueError:reports.append('raw_input_tamper')
    else:raise RuntimeError('INPUT_BYTES_ACCEPTED')
    p.write_bytes(original)
    p=root/'bundle/PREPARATION_CLOSURE.json';original=p.read_bytes();bad=json.loads(original);bad['quiescent']=False;p.write_bytes(json_bytes(bad))
    changed=copy.deepcopy(bound);changed['preparation_closure_sha256']=sha(p.read_bytes())
    try:input_reader.read_bundle(root/'bundle',changed,synthetic=True)
    except ValueError:reports.append('coherent_nonquiescent_preparation')
    else:raise RuntimeError('UNCLOSED_PREP_ACCEPTED')
    p.write_bytes(original)
    import authority,loader
    for fn in (lambda:authority.read_release('0'*64),lambda:loader.load(None,None,{},0)):
        try:fn()
        except (ValueError,OSError):pass
        else:raise RuntimeError('REAL_ENTRY_NOT_DISABLED')
    reports.append('missing_release_and_loader_refusal_before_provider')
    original_release=authority.RELEASE;authority.RELEASE=root/'FICTIONAL_RELEASE.json'
    try:
        release={**bound,'schema':'native_final_execution_release.v1','approved':True,'attempt':support.ATTEMPT,
            'limits':authority.LIMITS,'output':str(support.output()),'checkpoint_lock_sha256':sha((HERE/'CHECKPOINT.json').read_bytes()),
            'owned_identity_sha256':sha((HERE/'OWNED_IDENTITY.json').read_bytes()),
            'trace_sources':{n:h for n,h in frozen['source_sha256'].items() if n.endswith('.py')}}
        authority.RELEASE.write_bytes(json_bytes(release))
        try:authority.read_release('0'*64)
        except ValueError as e:require(str(e)=='ROOT_RELEASE_BYTES','RELEASE_HASH_REASON')
        else:raise RuntimeError('RELEASE_HASH_ACCEPTED')
        release['source_freeze_sha256']='0'*64;authority.RELEASE.write_bytes(json_bytes(release))
        try:authority.read_release(sha(authority.RELEASE.read_bytes()))
        except ValueError as e:require(str(e)=='RELEASE_BINDING_source_freeze_sha256','SOURCE_HASH_REASON')
        else:raise RuntimeError('SOURCE_ACCEPTED')
        reports.append('tampered_release_and_source_refusal')
    finally:authority.RELEASE=original_release
    # One tiny complete cohort, not the inherited broad suite.
    import torch
    from core import DispatchLatch,ForwardDerivativeGuard,trace
    from counts import Counts
    from receiver import NativeReceiver
    from science import BASE
    from workflow import execute,schedule
    torch.set_num_threads(1)
    params=json.loads((BASE/'real_attempt/fitted_parameters.json').read_bytes())['parameters']
    positive=torch.tensor([a+b for a,b in zip(params['grand_mean'],params['direction'],strict=True)],dtype=torch.float32)
    negative=torch.tensor([a-b for a,b in zip(params['grand_mean'],params['direction'],strict=True)],dtype=torch.float32)
    class Block(torch.nn.Module):
        def __init__(self):super().__init__();self.scale=torch.nn.Parameter(torch.ones(1))
        def forward(self,x):return x*self.scale
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.model=torch.nn.Module();self.model.rope_deltas=None;self.model.language_model=torch.nn.Module()
            self.model.language_model.layers=torch.nn.ModuleList([Block() for _ in range(24)]);self.eval()
        def forward(self,*,input_ids,attention_mask,logits_to_keep,**kw):
            index=int(input_ids[0,0])-1;case=data['cases'][index];template=positive if case['audit_only']['category']=='self_shutdown' else negative
            h=template.reshape(1,1,1024).expand(1,input_ids.shape[1],1024).clone()
            for layer in self.model.language_model.layers:h=layer(h)
            z=torch.full((1,1,248320),-100.);z[:,:,48964]=0.
            z[:,:,50057]=(.2 if index in (0,1,12) else -.2)+20*(h[:,-1,0]-template[0])
            if case['audit_only']['category']=='ordinary':z[:,:,32 if index==23 else case['audit_only']['correct_token_id']]=10.
            return types.SimpleNamespace(logits=z,past_key_values=None)
    base=root/'success';base.mkdir();old_output=support.output;support.output=lambda:base
    execution={'scope':'TINY_SYNTHETIC_ONLY','production_authorized':False};deadline=time.monotonic()+180
    counts=Counts(deadline,support.write_new);counts.reserve('load');latch=DispatchLatch();guard=ForwardDerivativeGuard(Tiny,counts,latch,deadline);guard.install()
    receiver=NativeReceiver(Tiny(),guard)
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
    finally:guard.restore()
    worked.update(execution=execution,guard_restored=not guard.installed,dispatch={'forwards':guard.forwards,'derivatives':guard.derivatives,'rejected':guard.rejected})
    support.write_new('WORKER_RESULT.json',worked,critical=True)
    from production_run import inventory,good_capture
    support.write_new('CLOSED_WORKER_BINDING.json',{'execution':execution,'good_capture':True,'files':inventory(),
        'worker_result_sha256':sha((base/'WORKER_RESULT.json').read_bytes())},critical=True)
    support.output=old_output
    def audit():return subprocess.run([sys.executable,'-B',str(HERE/'test_binding.py'),'audit',str(base)],capture_output=True,text=True,timeout=100)
    child=audit()
    if child.returncode:print(child.stdout);print(child.stderr);raise RuntimeError('SEPARATE_SAVED_JUDGE')
    audited=json.loads(child.stdout)
    require(audited['scientific_pass'] and audited['planned_cells']==180 and audited['off_identities']==36
        and audited['flips']==6 and audited['retentions']==6 and audited['unrun']==0,'COMPLETE_ADAPTER_COHORT')
    require(len(audited['ordinary_accuracy'])==18 and sum(not r['correct'] for r in audited['ordinary_accuracy'])==3,'PRESERVED_WRONG_AB')
    require(len(schedule(data['cases']))==180 and len(worked['requests'])==48,'FIXED_SCHEDULE')
    reports.append('tiny24_48requests_both_signs_12cold36off_separate_judge')
    # Outer inventory hash repair must not make an aliased entry valid.
    rowpath=base/'rows'/ (slots()[0]['id']+'__P__entry.json');original=rowpath.read_bytes();row=json.loads(original);row['current_id']=slots()[1]['id']+'__baseline';rowpath.write_bytes(json_bytes(row))
    bp=base/'CLOSED_WORKER_BINDING.json';saved=bp.read_bytes();binding=json.loads(saved)
    pin=next(p for p in binding['files'] if p['path']=='rows/'+rowpath.name);pin.update(bytes=rowpath.stat().st_size,sha256=sha(rowpath.read_bytes()));bp.write_bytes(json_bytes(binding))
    require(audit().returncode!=0,'ALIASED_ENTRY_REJECTED');rowpath.write_bytes(original);bp.write_bytes(saved);reports.append('coherent_entry_alias_rejected')
    capture={'binding_authenticated':True,'quiescent':True,'stop_reason':None,'faults':[],'cleanup_faults':[],
        'stdout_capture_errors':[],'primary_error':None,'threads_joined':True,'pipes_closed':True,'within_absolute_cleanup_deadline':True,
        'exit_proofs':{k:{'exit_code':0} for k in ('actual_worker','launcher')}}
    require(good_capture(capture),'CLOSURE_SUCCESS')
    for k in ('actual_worker','launcher'):
        bad=copy.deepcopy(capture);bad['exit_proofs'][k]['exit_code']=1;require(not good_capture(bad),'CLOSURE_EXIT_REJECT')
    bad=copy.deepcopy(capture);bad['quiescent']=False;require(not good_capture(bad),'QUIESCENCE_REJECT')
    from setup_budget import Budget
    budget=Budget(0.);budget.charge('worker',7.);budget.charge('audit',8.);require(budget.remaining==0 and budget.absolute_end==1995,'ONE_SHARED15')
    require(sum(support.GROUP_CAPS.values())+65536==209068032 and 209068032<288*1024**2,'FULL_RESERVATION')
    reports.append('retained_exit_quiescence_shared_cleanup_storage_reuse')
    require(not any(n.split('.')[0] in BLOCKED for n in sys.modules),'NO_REAL_PROVIDER')
    receipt={'status':'PASS','groups':[{'name':r,'status':'PASS'} for r in reports],'group_count':len(reports),
        'tiny_audit':audited,'source_candidate_sha256':source_sha,'real_tokenizer_calls':0,'real_qwen_calls':0,
        'actual_cohort_access':False,'real_release_created':False,'new_owned_model_process_observed':False}
    (HERE/'TEST_RESULTS.json').write_bytes(json_bytes(receipt));print(json.dumps(receipt,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(child_audit() if len(sys.argv)>1 and sys.argv[1]=='audit' else main())
