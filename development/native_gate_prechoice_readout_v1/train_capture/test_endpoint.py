"""Unmocked admission/reader/workflow/saved-audit/capture-to-fit endpoint.

Mirror substitutions are ROOT and ATTEMPT constants and resulting file hashes;
no validation/control function is replaced. Inputs are exact retained real text/
IDs (already admitted); native outputs/closure are artificial, never a model.
"""
from pathlib import Path
import ast,hashlib,importlib,json,math,os,shutil,struct,sys,tempfile,time,types
from contextlib import contextmanager
ROOT=Path(__file__).resolve().parents[3];SOURCE=Path(__file__).resolve().parent
def sha(raw):return hashlib.sha256(raw).hexdigest()
def jb(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def write(p,raw):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(raw)
def main():
    base=Path(tempfile.mkdtemp(prefix='ep_',dir=SOURCE.parent/'test_evidence'))
    copied=[];substitutions=[];source_hashes={}
    for p in SOURCE.iterdir():
        if p.is_file() and not p.name.startswith('test_') and p.name not in ('SOURCE_FREEZE.json','RELEASE_DRAFT.json'):
            raw=p.read_bytes()
            source_hashes[p.name]=sha(raw)
            if p.suffix=='.py':
                text=raw.decode();before=ast.parse(text)
                text=text.replace('ROOT=HERE.parents[2]',f'ROOT=Path({str(ROOT)!r})').replace('ROOT=Path(__file__).resolve().parents[3]',f'ROOT=Path({str(ROOT)!r})')
                text=text.replace("ATTEMPT='prechoice_train_capture_attempt_001'","ATTEMPT='t'")
                after=ast.parse(text)
                functions=lambda tree:[ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef,ast.AsyncFunctionDef))]
                assert functions(before)==functions(after),p.name
                if text!=raw.decode():substitutions.append(p.name)
                raw=text.encode()
            write(base/p.name,raw);copied.append(p.name)
    freeze={'source_sha256':{n:sha((base/n).read_bytes()) for n in copied},'external_sources':[
        {'path':'development/native_gate_frozen_transfer_v1/capture/receiver.py','sha256':'75680a94ae50ca6ea53765fb762061f36a3bc4a394fbf423de4f2f050683c152'}]}
    write(base/'SOURCE_FREEZE.json',jb(freeze))
    sys.path.insert(0,str(base))
    import support,input_reader,authority,workflow,audit_saved,index_contract
    from counts import Counts
    inputs=input_reader.build_inputs();assert len(inputs['cases'])==58
    release={'schema':'prechoice_train_capture_release.v1','approved':True,'role':'TRAIN_PRECHOICE','attempt':'t',
        'limits':authority.LIMITS,'output':str(support.output()),'reserved_bytes':67538944,
        'source_freeze_sha256':sha((base/'SOURCE_FREEZE.json').read_bytes()),'checkpoint_lock_sha256':sha((base/'CHECKPOINT.json').read_bytes()),
        'owned_identity_sha256':sha((base/'OWNED_IDENTITY.json').read_bytes()),'input_data_lock_sha256':sha((base/'DATA_LOCK.json').read_bytes()),
        'inputs_sha256':sha(support.json_bytes(inputs)),'certificate_sha256':inputs['certificate_sha256'],
        'trace_sources':{n:h for n,h in freeze['source_sha256'].items() if n.endswith('.py')}}
    raw=jb(release);write(base/'root_release/RELEASE.json',raw);approved=sha(raw)
    assert authority.read_release(approved)==release
    negatives=[]
    # Rewrite only synthetic release fixture; prospective production files untouched.
    for field,value in (('certificate_sha256','0'*64),('inputs_sha256','0'*64),('approved',False),('reserved_bytes',67538945),('source_freeze_sha256','0'*64)):
        bad={**release,field:value};(base/'root_release/RELEASE.json').write_bytes(jb(bad))
        try:authority.read_release(sha(jb(bad)))
        except (ValueError,KeyError):negatives.append(field)
        else:raise AssertionError('accepted bad '+field)
    (base/'root_release/RELEASE.json').write_bytes(raw)
    admission=authority.admit_once(approved);assert authority.authenticate()['execution']==admission['execution']
    try:authority.admit_once(approved)
    except FileExistsError:negatives.append('second_attempt')
    else:raise AssertionError('second attempt accepted')
    from array import array
    fake=ROOT/'development/native_supervised_gate_evaluation_v1/test_candidate.py';fake_raw=fake.read_bytes()
    assert sha(fake_raw)=='ef9975fe8680a32ab7800d10b7eb872486bdd30df8643a026a6fca4528ad1a56'
    scope={'struct':struct,'math':math,'array':array}
    tree=ast.parse(fake_raw);exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('f32','Vec','Difference','Logits')],type_ignores=[]),'pinned_artificial_arrays','exec'),scope)
    Vec,Logits=scope['Vec'],scope['Logits'];torch=types.SimpleNamespace(float32='float32',zeros=lambda n,**k:Vec([0.]*n))
    class FakeReceiver:
        def __init__(self):self.capture=None;self.edit_hook_registrations=0;self.i=0
        def clear_capture(self):self.capture=None
        def forward_inputs(self,ids,mask,phase,offset,*,readout_selector):
            index_contract.validate(ids,readout_selector);self.i+=1
            self.capture={'hook_calls':1,'final_input_index':len(ids)-1,'logit_count':248320,'parameter_versions_unchanged':True,
                'hook':'blocks.10.hook_out','native_target':'model.language_model.layers.10','nonfinal_positions':len(ids)-1,'unselected_sha256':'a'*64,
                'readout_index':readout_selector['readout_index'],'readout_selector_sha256':index_contract.sha(index_contract.canon(readout_selector)),
                'readout_input_ids_sha256':index_contract.int_hash(ids),'readout_nonselected_positions':len(ids)-1,
                'readout_nonselected_sha256':'b'*64,'all_positions_unchanged':True,'feature_contract':index_contract.CONTRACT}
            return Logits(.2),Vec([self.i*.125+(j%13)*.0625 for j in range(1024)])
        def finalize(self):return {'parameter_bytes_unchanged':True,'buffer_bytes_unchanged':True,
            'parameter_sha256':'2eeb7434977939f80711e1fa83a1d587c71887da0c06f1edc13a43d5ec1192e6',
            'buffer_sha256':'18f0caa55461f0f07afba9cb9809f1830745b445e5cc62283d505a0017f892fc'}
    execution=admission['execution']
    def publish(n,v):return support.write_new(n,v,raw=type(v) is bytes)
    @contextmanager
    def trace(name):
        try:yield
        finally:publish('traces/'+name+'.json',{'execution':execution,'trace_incomplete':False,'open_stages':[],'primary':None,'events':[{'edge':'ENTER'},{'edge':'RETURN'}]})
    counts=Counts(time.monotonic()+120,publish);counts.reserve('load')
    worker=workflow.execute(FakeReceiver(),torch,counts,inputs,publish,trace,time.monotonic()+120)
    assert worker['scientific_pass'],worker
    worker.update(execution=execution,guard_restored=True,dispatch={'forwards':58,'derivatives':0,'rejected':0})
    publish('WORKER_RESULT.json',worker)
    state=worker['cleanup']['state'];publish('LOADER_READY.json',{'execution':execution,'native_initial_sha256':state['parameter_sha256'],
        'native_initial_buffer_sha256':state['buffer_sha256'],'loading_info':{},'declared_class':'Qwen3_5ForConditionalGeneration',
        'coverage':{'complete_key_shape_coverage':True,'native_unique_parameters':473,'named_occurrences':474},'old_digest_equivalence_claimed':False})
    files=[{'path':p.relative_to(support.output()).as_posix(),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in support.output().rglob('*') if p.is_file()]
    publish('CLOSED_WORKER_BINDING.json',{'execution':execution,'good_capture':True,'worker_result_sha256':sha((support.output()/'WORKER_RESULT.json').read_bytes()),'files':files})
    audited=audit_saved.judge(support.output(),execution,time.monotonic()+120);assert audited['scientific_pass'],audited
    publish('AUDIT_RESULT.json',audited);publish('PARENT_FINAL.json',{'execution':execution,'audit_completed':True,'scientific_pass':True,
        'worker_quiescent':True,'audit_quiescent':True,'errors':[],'classification':'COMPLETE_NATIVE_CONSTRUCTION_CAPTURE'})
    import fit_source_auth
    manifest=fit_source_auth.build_manifest();rows,labels=fit_source_auth.extract_features(manifest,deadline=time.monotonic()+120)
    assert len(rows)==58 and len(labels)==58 and labels.count(1)==14
    result={'status':'PASS_UNMOCKED_FUNCTIONS_ARTIFICIAL_NATIVE_OUTPUTS','views':58,'negatives':negatives,'constant_substitutions':substitutions,
        'function_AST_changed':False,'actual_input_metadata':True,'model_calls':0,'tokenizer_calls':0,'fits':0,'actual_admission':False,
        'mirror':str(base),'source_sha256':source_hashes}
    write(base/'TEST_RESULT.json',jb(result));print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
