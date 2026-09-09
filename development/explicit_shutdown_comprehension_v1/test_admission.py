"""Unmocked artificial source-release-prepared-input chain; constants/metadata only."""
import ast,copy,hashlib,json,os,sys,tempfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];NS='development/explicit_shutdown_comprehension_v1'
def sha(b):return hashlib.sha256(b).hexdigest()
def jb(v):return (json.dumps(v,sort_keys=True,separators=(',',':'))+'\n').encode()
def main():
    started=time.monotonic();mirror=Path(tempfile.mkdtemp(prefix='admission_',dir=HERE/'test_evidence')).resolve()
    if os.name=='nt':mirror=Path('\\\\?\\'+str(mirror))
    cap=mirror/NS;prep=cap/'preparation'
    def put(p,v):
        assert p.resolve().is_relative_to(mirror);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v if type(v) is bytes else jb(v))
    cf=json.loads((HERE/'SOURCE_FREEZE.json').read_bytes());pf=json.loads((HERE/'preparation/SOURCE_FREEZE.json').read_bytes())
    for n in cf['source_sha256']:put(cap/n,(HERE/n).read_bytes())
    for n in pf['source_sha256']:put(prep/n,(HERE/'preparation'/n).read_bytes())
    for rel in ('development/native_final_preparation_v1/test_prepare.py','development/native_supervised_gate_evaluation_v1/test_candidate.py'):
        put(mirror/rel,(ROOT/rel).read_bytes())
    fake={'families':[{'id':f,'cases':[{'id':f+'_'+c,'category':c,'current_assistant_status':'ends' if c=='self_shutdown' else 'continues',
        'scenario':'ARTIFICIAL_SCENARIO_'+f+'_'+c} for c in ('self_shutdown','other_shutdown','non_termination_control')]}
        for f in ('G01','G02','G03','G04')]}
    source=mirror/'development/native_supervised_gate_v2/TRAINING_SUBMISSION.json';put(source,fake);source_sha=sha(source.read_bytes())
    deps=json.loads((prep/'DEPENDENCIES.json').read_bytes())
    for p in deps['files']:
        if p['path'].endswith('TRAINING_SUBMISSION.json'):p.update(path=str(source),sha256=source_sha)
        else:p['path']=str(ROOT/p['path'])
    put(prep/'DEPENDENCIES.json',deps)
    text=(prep/'plan.py').read_text();text=text.replace('6e950138ef39c9db25eff9c4b3c2ce644288a64f05915e1262ff4f270b0b84fc',source_sha)
    text=text.replace('273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80',sha(b'FAKE_TEMPLATE_NOT_NATIVE_TOKENIZER'))
    put(prep/'plan.py',text.encode())
    pf['source_sha256']={n:sha((prep/n).read_bytes()) for n in pf['source_sha256']};pf['external_sources']=[dict(p) for p in deps['files']]
    put(prep/'SOURCE_FREEZE.json',pf);prep_sha=sha((prep/'SOURCE_FREEZE.json').read_bytes())
    original=(cap/'input_reader.py').read_bytes();realprep=sha((HERE/'preparation/SOURCE_FREEZE.json').read_bytes())
    put(cap/'input_reader.py',original.replace(realprep.encode(),prep_sha.encode()))
    cf['source_sha256']={n:sha((cap/n).read_bytes()) for n in cf['source_sha256']}
    cf['external_sources']=[{'path':str(prep/'SOURCE_FREEZE.json'),'sha256':prep_sha}]
    put(cap/'SOURCE_FREEZE.json',cf);cap_sha=sha((cap/'SOURCE_FREEZE.json').read_bytes())
    sys.path[:0]=[str(cap),str(prep)]
    import test_control as fake_helpers
    import plan,renderer,prepare_reader,prepare_core,input_reader,authority
    # Helpers are an artificial tokenizer implementation; production preparation runs unchanged.
    bundle=cap/'root_release';bundle.mkdir()
    result,lock,tok=fake_helpers.prepare_fake(bundle)
    assert result['status']=='PASS' and result['completed_operations']==157
    lock.update(scope='ROOT_ADMITTED_NEW_FINAL_TEXT',admitted_submission_sha256=source_sha,
        cohort_identity=plan.cohort_identity(source_sha),blind_semantic_review_approved=True,
        overlap_review_approved=True,final_text_locked=True)
    lock['final_execution_binding']=prepare_reader.execution_binding(cap_sha)
    lock['preparation_owner_binding']={'namespace':NS,'source_freeze_sha256':cap_sha}
    assert lock['cohort']==renderer.read_submission(source_sha)
    put(bundle/'TEXT_LOCK.json',lock);text_sha=sha((bundle/'TEXT_LOCK.json').read_bytes())
    data=json.loads((bundle/'preparation/inputs.json').read_bytes())
    data.update(scope='OFFLINE_FINAL_PREPARATION',cohort_identity=lock['cohort_identity'],text_lock_canonical_sha256=sha(prepare_core.jb(lock)))
    data['source_identity']['text_lock_raw_sha256']=text_sha
    put(bundle/'preparation/inputs.json',data);result.update(scope='OFFLINE_FINAL_PREPARATION',inputs_sha256=sha((bundle/'preparation/inputs.json').read_bytes()))
    put(bundle/'preparation/RESULT.json',result)
    c=fake_helpers.closure(sha((bundle/'preparation/RESULT.json').read_bytes()),text_sha);c['identity']['owner_source_sha256']=cap_sha
    put(bundle/'PREPARATION_CLOSURE.json',c)
    release={'schema':'explicit_shutdown_comprehension_release.v1','approved':True,'attempt':authority.ATTEMPT,'limits':authority.LIMITS,
        'output':str(cap/'real_evidence'/authority.ATTEMPT),'reserved_bytes':15065088,'source_freeze_sha256':cap_sha,
        'checkpoint_lock_sha256':sha((cap/'CHECKPOINT.json').read_bytes()),'owned_identity_sha256':sha((cap/'OWNED_IDENTITY.json').read_bytes()),
        'trace_sources':{n:h for n,h in cf['source_sha256'].items() if n.endswith('.py')},'text_lock_sha256':text_sha,
        'inputs_sha256':result['inputs_sha256'],'preparation_files':{n:sha((bundle/'preparation'/n).read_bytes()) for n in prepare_reader.artifact_names()},
        'preparation_closure_sha256':sha((bundle/'PREPARATION_CLOSURE.json').read_bytes()),'admitted_submission_sha256':source_sha}
    outcomes=[]
    def check(name,wanted):
        put(bundle/'RELEASE.json',release);approved=sha((bundle/'RELEASE.json').read_bytes())
        try:
            got=authority.read_release(approved);value=input_reader.read_bundle(bundle,got)
            accepted=len(value['cases'])==12;code=None
        except Exception as error:accepted=False;code=str(error)
        outcomes.append({'name':name,'accepted':accepted,'code':code});assert accepted==wanted,outcomes[-1]
    os.chdir(prep)
    check('unmocked_positive_nested_cwd',True)
    release['approved']=False;check('default_deny',False);release['approved']=True
    c['status']='FAIL';put(bundle/'PREPARATION_CLOSURE.json',c);release['preparation_closure_sha256']=sha((bundle/'PREPARATION_CLOSURE.json').read_bytes())
    check('coherent_unclosed_preparation',False)
    identities={}
    for name in ('authority.py','input_reader.py','preparation_owner.py','preparation/plan.py','preparation/prepare_core.py','preparation/prepare_reader.py','preparation/renderer.py'):
        bodies=lambda b:[ast.dump(n,include_attributes=False) for n in ast.parse(b).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        identities[name]=bodies((cap/name).read_bytes())==bodies((HERE/name).read_bytes())
    assert all(identities.values())
    forbidden=sorted({'torch','transformers','tokenizers','safetensors','numpy'}&{n.split('.')[0] for n in sys.modules});assert not forbidden
    report={'status':'PASS','tests':outcomes,'production_function_mocks':0,'unchanged_function_bodies':identities,
        'fake_preparation_operations':157,'actual_tokenizer_calls':0,'model_calls':0,'forbidden_imports':forbidden,
        'elapsed_seconds':time.monotonic()-started,'script_sha256':sha(Path(__file__).read_bytes()),'mirror':str(mirror)}
    put(mirror/'RESULT.json',report);print(json.dumps(report,sort_keys=True))
if __name__=='__main__':main()
