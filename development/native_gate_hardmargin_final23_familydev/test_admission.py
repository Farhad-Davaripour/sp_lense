"""One unmocked final23-to-original-preparation admission chain, artificial data only."""
import ast,copy,hashlib,importlib,json,os,shutil,sys,tempfile,time
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
NEW='development/native_supervised_gate_capture_final23_v1'
OLD='development/native_supervised_gate_capture_v1'
PREP='development/native_supervised_gate_preparation_v3'
FIT='development/native_gate_hardmargin_final23_familydev'
SCHEMA='development/native_supervised_gate_v2/author_packet/EMPTY_SCHEMA.json'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def jb(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def get(path):return json.loads(path.read_bytes())
def bodies(raw):
    return [ast.dump(n,include_attributes=False) for n in ast.parse(raw).body
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]

def main():
    started=time.monotonic();mirror=Path(tempfile.mkdtemp(prefix='admission_mirror_',dir=HERE)).resolve()
    if os.name=='nt':mirror=Path('\\\\?\\'+str(mirror))
    cap=mirror/NEW;old=mirror/OLD;prep=mirror/PREP;fit=mirror/FIT
    report={'scope':'ARTIFICIAL_UNMOCKED_ADMISSION_CHAIN','production_function_mocks':0,
        'model_calls':0,'tokenizer_calls':0,'fits':0,'tests':[],'mirror':str(mirror)}
    def put(path,value):
        assert path.resolve().is_relative_to(mirror)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(value if type(value) is bytes else jb(value))
    def clone(source,target):
        assert 'author_packet' not in source.parts and source.name!='TRAINING_SUBMISSION.json'
        put(target,source.read_bytes())
    # Only source files and the checked-in artificial fixture are read.
    for namespace in (NEW,OLD,PREP):
        for name in get(ROOT/namespace/'SOURCE_FREEZE.json')['source_sha256']:
            clone(ROOT/namespace/name,mirror/namespace/name)
    for name in ('gate.py','source_auth.py'):clone(HERE/name,fit/name)
    clone(ROOT/OLD/'input_reader.py',old/'input_reader.py')
    fixture=ROOT/OLD/'fixtures/preparation_bundle/bundle'
    lock=get(fixture/'TEXT_LOCK.json');cohort=lock['cohort'];schema_raw=jb(cohort)
    put(mirror/SCHEMA,schema_raw)
    submission_sha=sha(jb(cohort))
    put(mirror/'development/native_supervised_gate_v2/TRAINING_SUBMISSION.json',jb(cohort))
    deps=get(prep/'DEPENDENCIES.json')
    for pin in deps['files']:
        if 'author_packet' in pin['path']:pin['sha256']=sha(schema_raw)
        else:clone(ROOT/pin['path'],mirror/pin['path'])
    put(prep/'DEPENDENCIES.json',deps)
    pf=get(ROOT/PREP/'SOURCE_FREEZE.json')
    pf['source_sha256']['DEPENDENCIES.json']=sha((prep/'DEPENDENCIES.json').read_bytes())
    pf['external_sources']=[{'path':str(mirror/p['path']),'sha256':p['sha256']} for p in deps['files']]
    put(prep/'SOURCE_FREEZE.json',pf);prep_sha=sha((prep/'SOURCE_FREEZE.json').read_bytes())
    # The old adapter is loaded afresh by production code, so substitute its
    # hash constant in the mirror source; every function/class AST is unchanged.
    old_raw=(old/'input_reader.py').read_bytes()
    old_pin="PREPARATION_SOURCE_SHA256='1b094732e4dc8ab60abd84a3f2a8bebffc08c42447497bfde196e00167bf6de5'"
    assert old_raw.decode().count(old_pin)==1
    put(old/'input_reader.py',old_raw.decode().replace(old_pin,"PREPARATION_SOURCE_SHA256='"+prep_sha+"'").encode())
    sys.path[:0]=[str(fit),str(cap),str(prep)]
    import source_auth,input_reader,validate,plan,authority
    validate.SCHEMA_SHA256=sha(schema_raw)
    input_reader.ORIGINAL_READER_SHA=sha((old/'input_reader.py').read_bytes())
    old_freeze=get(ROOT/OLD/'SOURCE_FREEZE.json')
    old_freeze['source_sha256']['input_reader.py']=input_reader.ORIGINAL_READER_SHA
    old_freeze['external_sources']=[{'path':str(prep/'SOURCE_FREEZE.json'),'sha256':prep_sha}]
    put(old/'SOURCE_FREEZE.json',old_freeze)
    old_source_sha=sha((old/'SOURCE_FREEZE.json').read_bytes())
    bundle=old/'root_release'
    lock.update(scope='ROOT_ADMITTED_NEW_FINAL_TEXT',admitted_submission_sha256=submission_sha,
        blind_semantic_review_approved=True,overlap_review_approved=True,final_text_locked=True)
    lock['cohort_identity']=plan.cohort_identity(submission_sha)
    lock['final_execution_binding']={'namespace':OLD,'source_freeze_sha256':old_source_sha,
        'preparation_source_freeze_sha256':prep_sha}
    lock['preparation_owner_binding']={'namespace':OLD,'source_freeze_sha256':old_source_sha}
    put(bundle/'TEXT_LOCK.json',lock);text_sha=sha((bundle/'TEXT_LOCK.json').read_bytes())
    data=get(fixture/'preparation/inputs.json')
    data.update(scope='OFFLINE_FINAL_PREPARATION',cohort_identity=lock['cohort_identity'],
        text_lock_canonical_sha256=sha(jb(lock)))
    data['source_identity'].update(text_lock_raw_sha256=text_sha,source_freeze_sha256=prep_sha,
        dependencies_sha256=sha((prep/'DEPENDENCIES.json').read_bytes()))
    for case in data['cases']:
        case['input_binding']['chat_template_sha256']=plan.TEMPLATE_SHA256
        record=get(fixture/'preparation'/(case['case_key']+'.json'))
        record['chat_template_sha256']=plan.TEMPLATE_SHA256
        put(bundle/'preparation'/(case['case_key']+'.json'),record)
    put(bundle/'preparation/inputs.json',data)
    result=get(fixture/'preparation/RESULT.json')
    result.update(scope='OFFLINE_FINAL_PREPARATION',inputs_sha256=sha((bundle/'preparation/inputs.json').read_bytes()))
    put(bundle/'preparation/RESULT.json',result)
    clone(fixture/'preparation/operations.jsonl',bundle/'preparation/operations.jsonl')
    closure=get(fixture/'PREPARATION_CLOSURE.json')
    closure.update(text_lock_sha256=text_sha,preparation_result_sha256=sha((bundle/'preparation/RESULT.json').read_bytes()))
    closure['identity'].update(text_lock_sha256=text_sha,owner_source_sha256=old_source_sha)
    put(bundle/'PREPARATION_CLOSURE.json',closure)
    bindings={'approved':False,'inputs_sha256':result['inputs_sha256'],'text_lock_sha256':text_sha,
        'preparation_files':{p.name:sha(p.read_bytes()) for p in (bundle/'preparation').iterdir()},
        'preparation_closure_sha256':sha((bundle/'PREPARATION_CLOSURE.json').read_bytes()),
        'admitted_submission_sha256':submission_sha,'source_freeze_sha256':old_source_sha}
    cf=get(ROOT/NEW/'SOURCE_FREEZE.json')
    cf['external_sources']=[{'path':str(old/'input_reader.py'),'sha256':input_reader.ORIGINAL_READER_SHA},
        {'path':str(prep/'SOURCE_FREEZE.json'),'sha256':prep_sha}]
    release_dir=cap/'root_release'
    release={'schema':'native_supervised_gate_capture_final23_release.v1','approved':True,
        'attempt':source_auth.ATTEMPT,'limits':authority.LIMITS,
        'output':str(cap/'real_evidence'/source_auth.ATTEMPT),'reserved_bytes':37879808,
        'checkpoint_lock_sha256':sha((cap/'CHECKPOINT.json').read_bytes()),
        'owned_identity_sha256':sha((cap/'OWNED_IDENTITY.json').read_bytes())}
    def republish():
        bindings['preparation_closure_sha256']=sha((bundle/'PREPARATION_CLOSURE.json').read_bytes())
        record={'schema':'final23_readonly_input_reuse.v1','approved':False,'scope':'INPUT_PROOF_REUSE_ONLY',
            'retokenize':False,'base':str(bundle),'bundle_bindings':bindings}
        put(cap/'REUSED_INPUTS.json',record)
        cf['source_sha256']['REUSED_INPUTS.json']=sha((cap/'REUSED_INPUTS.json').read_bytes())
        put(cap/'SOURCE_FREEZE.json',cf);cap_sha=sha((cap/'SOURCE_FREEZE.json').read_bytes())
        source_auth.CAPTURE_SOURCE_SHA256=cap_sha
        for own,other,key in (('PREPARED_INPUT_SHA256','INPUT_SHA','inputs_sha256'),
            ('PREPARATION_CLOSURE_SHA256','CLOSURE_SHA','preparation_closure_sha256')):
            setattr(source_auth,own,bindings[key]);setattr(input_reader,other,bindings[key])
        source_auth.PREPARATION_RESULT_SHA256=input_reader.RESULT_SHA=bindings['preparation_files']['RESULT.json']
        release.update({key:bindings[key] for key in ('inputs_sha256','text_lock_sha256','preparation_files',
            'preparation_closure_sha256','admitted_submission_sha256')})
        release.update(source_freeze_sha256=cap_sha,reused_inputs_sha256=sha((cap/'REUSED_INPUTS.json').read_bytes()),
            trace_sources={n:h for n,h in cf['source_sha256'].items() if n.endswith('.py')})
        put(release_dir/'RELEASE.json',release)
        return authority.execution(release,sha((release_dir/'RELEASE.json').read_bytes()))
    execution=republish()
    identities={}
    for relative in (FIT+'/source_auth.py',NEW+'/authority.py',NEW+'/input_reader.py',NEW+'/support.py',
        OLD+'/input_reader.py',PREP+'/prepare_reader.py',PREP+'/prepare_core.py',PREP+'/renderer.py',
        PREP+'/validate.py',PREP+'/dependencies.py'):
        identities[relative]=bodies((mirror/relative).read_bytes())==bodies((ROOT/relative).read_bytes())
    assert all(identities.values());report['unchanged_function_bodies']=identities
    report['constant_substitutions']={'source_auth':'mirror capture/preparation hashes',
        'input_reader':'mirror immutable adapter/preparation hashes',
        'old_input_reader.PREPARATION_SOURCE_SHA256':prep_sha,'validate.SCHEMA_SHA256':sha(schema_raw)}
    os.chdir(fit)
    def check(name,expected,ex=None,code=None):
        try:
            got=source_auth.authenticate_capture(execution if ex is None else ex,deadline=time.monotonic()+30)
            row={'name':name,'accepted':True,'rows':len(got['cases'])}
        except Exception as error:
            row={'name':name,'accepted':False,'code':str(error),'error_type':type(error).__name__}
        report['tests'].append(row);print(json.dumps(row),flush=True)
        assert row['accepted']==expected,row
        if code is not None:assert row.get('code')==code,row
    check('unmocked_positive_from_fit_owner_cwd',True)
    original=(cap/'authority.py').read_bytes()
    put(cap/'authority.py',original+b'\n# artificial wrong source\n')
    check('wrong_source_rejected',False,code='CAPTURE_FILE_HASH');put(cap/'authority.py',original)
    release['approved']=False;execution=republish()
    check('coherently_hashed_unapproved_release_rejected',False,code='EXACT_NATIVE_RELEASE')
    release['approved']=True;execution=republish()
    bad=copy.deepcopy(execution);bad['scope']='ROOT_APPROVED_NATIVE_CONSTRUCTION_CAPTURE_ONLY'
    check('old_namespace_execution_rejected',False,bad,code='EXACT_CAPTURE_RELEASE_EXECUTION')
    closure['status']='FAIL';put(bundle/'PREPARATION_CLOSURE.json',closure);execution=republish()
    check('coherently_hashed_unclosed_preparation_rejected',False,code='ROOT_PREPARATION_CLOSED')
    report['forbidden_imports']=sorted({'torch','transformers','tokenizers','safetensors','numpy'} &
        {name.split('.')[0] for name in sys.modules})
    assert not report['forbidden_imports']
    report['elapsed_seconds']=time.monotonic()-started;report['script_sha256']=sha(Path(__file__).read_bytes())
    put(mirror/'RESULT.json',report)
    print(json.dumps({'passed':len(report['tests']),'result':str(mirror/'RESULT.json'),
        'elapsed_seconds':report['elapsed_seconds'],'production_function_mocks':0}))

if __name__=='__main__':main()
