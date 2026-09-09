"""Independent artificial two-chain admission proof; never reads real author/evidence."""
import ast,copy,hashlib,importlib,json,os,re,struct,sys,tempfile,time
from contextlib import contextmanager
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD_CAP='development/native_supervised_gate_capture_v1';OLD_PREP='development/native_supervised_gate_preparation_v3'
NEW_CAP='development/native_supervised_gate_capture_coverage_v1';NEW_PREP='development/native_supervised_gate_preparation_coverage_v1'
FIT='development/native_gate_hardmargin_coverage_v1';LEGACY='development/native_supervised_gate_v2'
FIXTURE=ROOT/OLD_CAP/'fixtures/preparation_bundle/bundle'
NAMES=('gate','source_auth','support','input_reader','authority','dependencies','plan','storage','renderer','validate','prepare_core','prepare_reader','native_supervised_prepared_reader')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def jb(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def get(p):return json.loads(p.read_bytes())
def bodies(raw):return [ast.dump(n,include_attributes=False) for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
def packed(v):return sha(struct.pack('<'+'q'*len(v),*v))
class Deny:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in {'numpy','torch','transformers','tokenizers','safetensors','datasets','pyarrow','transformer_lens'}:raise ImportError('NO_REAL_PROVIDER_IN_ADMISSION_PROOF')
sys.meta_path.insert(0,Deny())
@contextmanager
def isolated(paths):
    saved={n:sys.modules[n] for n in NAMES if n in sys.modules};path=list(sys.path)
    for n in NAMES:sys.modules.pop(n,None)
    sys.path[:0]=[str(p) for p in paths]
    try:yield
    finally:
        for n in NAMES:sys.modules.pop(n,None)
        sys.modules.update(saved);sys.path[:]=path

def main():
    started=time.monotonic();directory=Path(tempfile.mkdtemp(prefix='two_chain_',dir=HERE)).resolve()
    mirror=Path('\\\\?\\'+str(directory)) if os.name=='nt' else directory
    def put(p,v):
        assert p.resolve().is_relative_to(mirror);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v if type(v) is bytes else jb(v))
    def clone(p,q):
        assert p.name not in ('TRAINING_SUBMISSION.json','AUTHOR_ATTESTATION.md') and 'real_evidence' not in p.parts
        put(q,p.read_bytes())
    original_bodies={};substitutions=[]
    def literal(p,name,value):
        raw=p.read_bytes();text=raw.decode();pattern=r'(?m)^'+re.escape(name)+r'=.*$'
        text,n=re.subn(pattern,name+'='+repr(value),text);assert n==1,(p,name)
        put(p,text.encode());assert bodies(raw)==bodies(p.read_bytes());substitutions.append(str(p.relative_to(mirror))+':'+name)
    def chain(cap_ns,prep_ns,new):
        cap=mirror/cap_ns;prep=mirror/prep_ns
        for ns in (cap_ns,prep_ns):
            freeze=get(ROOT/ns/'SOURCE_FREEZE.json')
            for name in freeze['source_sha256']:clone(ROOT/ns/name,mirror/ns/name)
        deps=get(prep/'DEPENDENCIES.json')
        for pin in deps['files']:
            p=Path(pin['path']);relative=p.relative_to(ROOT) if p.is_absolute() else p
            clone(ROOT/relative,mirror/relative);pin['path']=str(mirror/relative)
        put(prep/'DEPENDENCIES.json',deps)
        pf=get(ROOT/prep_ns/'SOURCE_FREEZE.json');pf['source_sha256']['DEPENDENCIES.json']=sha((prep/'DEPENDENCIES.json').read_bytes());pf['external_sources']=[dict(p) for p in deps['files']]
        put(prep/'SOURCE_FREEZE.json',pf);prep_sha=sha((prep/'SOURCE_FREEZE.json').read_bytes())
        literal(cap/'input_reader.py','PREPARATION_SOURCE_SHA256',prep_sha)
        cf=get(ROOT/cap_ns/'SOURCE_FREEZE.json');cf['source_sha256']['input_reader.py']=sha((cap/'input_reader.py').read_bytes());cf['external_sources']=[{'path':str(prep/'SOURCE_FREEZE.json'),'sha256':prep_sha}]
        put(cap/'SOURCE_FREEZE.json',cf);cap_sha=sha((cap/'SOURCE_FREEZE.json').read_bytes())
        with isolated((cap,prep)):
            import plan,renderer,validate,prepare_reader,authority,input_reader
            lock=get(FIXTURE/'TEXT_LOCK.json');cohort=copy.deepcopy(lock['cohort'])
            if new:
                cohort['schema_version']=get(mirror/'development/native_supervised_gate_coverage_v1/author_packet/EMPTY_SCHEMA.json')['schema_version']
                cohort['families']=cohort['families'][:2];cohort['ordinary']=[]
                for i,f in enumerate(cohort['families'],5):
                    old=f['id'];f['id']='G0'+str(i)
                    for c in f['cases']:c['id']=c['id'].replace(old,f['id']);c['scenario']=c['scenario'].replace(old,f['id'])
            submission_sha=sha(jb(cohort));put(mirror/plan.COHORT_IDENTITY['cohort_namespace']/'TRAINING_SUBMISSION.json',cohort)
            rendered=validate.validate(cohort)['prompts'];slots=plan.slots()
            lock.update(cohort=cohort,cohort_sha256=sha(jb(cohort)),rendered_prompts=rendered,scope='ROOT_ADMITTED_NEW_FINAL_TEXT',admitted_submission_sha256=submission_sha,
                cohort_identity=plan.cohort_identity(submission_sha),model=plan.MODEL,study=plan.STUDY,blind_semantic_review_approved=True,overlap_review_approved=True,final_text_locked=True,
                final_execution_binding=prepare_reader.execution_binding(cap_sha),preparation_owner_binding={'namespace':cap_ns,'source_freeze_sha256':cap_sha})
            bundle=cap/'root_release';put(bundle/'TEXT_LOCK.json',lock);text_sha=sha((bundle/'TEXT_LOCK.json').read_bytes())
            data=get(FIXTURE/'preparation/inputs.json');cases=[]
            for i,(slot,text) in enumerate(zip(slots,rendered,strict=True)):
                p=copy.deepcopy(data['cases'][i]);record=get(FIXTURE/'preparation'/(p['case_key']+'.json'))
                p.update(case_key=slot['id'],family=slot['family_id'],layout=slot['layout'])
                if new:p['input']['input_ids'][0]+=8192
                p['audit_only']['category']=slot['category'];p['audit_only']['expected_route']='ON' if slot['category']=='self_shutdown' else 'OFF'
                chat='<FAKE_USER>'+text['prompt']+'<FAKE_END>\n'+plan.HEADER_TEXT
                record.update(case_key=slot['id'],prompt_sha256=text['prompt_sha256'],rendered_chat=chat,rendered_chat_utf8_sha256=sha(chat.encode()),full_token_ids=p['input']['input_ids'],chat_template_sha256=plan.TEMPLATE_SHA256)
                p['input_binding']={'prompt_sha256':text['prompt_sha256'],'rendered_chat_utf8_sha256':sha(chat.encode()),'chat_template_sha256':plan.TEMPLATE_SHA256,'derived_input_int64_le_sha256':packed(p['input']['input_ids']),'derived_mask_int64_le_sha256':packed(p['input']['attention_mask'])}
                put(bundle/'preparation'/(slot['id']+'.json'),record);cases.append(p)
            data.update(cases=cases,scope='OFFLINE_FINAL_PREPARATION',cohort_identity=lock['cohort_identity'],model=plan.MODEL,study=plan.STUDY,text_lock_canonical_sha256=sha(jb(lock)),exact_lengths={p['case_key']:p['input']['prompt_length'] for p in cases})
            data['source_identity'].update(text_lock_raw_sha256=text_sha,source_freeze_sha256=prep_sha,dependencies_sha256=sha((prep/'DEPENDENCIES.json').read_bytes()))
            put(bundle/'preparation/inputs.json',data);result=get(FIXTURE/'preparation/RESULT.json');count=len(plan.operations())
            result.update(scope='OFFLINE_FINAL_PREPARATION',inputs_sha256=sha((bundle/'preparation/inputs.json').read_bytes()),lengths=data['exact_lengths'],attempted_operations=count,completed_operations=count,planned_operations=count,completed_cases=len(cases))
            put(bundle/'preparation/RESULT.json',result)
            journal=b''.join(jb({'ordinal':i+1,'name':name,'status':status}) for i,name in enumerate(plan.operations()) for status in ('STARTED','COMPLETE'))
            put(bundle/'preparation/operations.jsonl',journal);closure=get(FIXTURE/'PREPARATION_CLOSURE.json');closure.update(text_lock_sha256=text_sha,preparation_result_sha256=sha((bundle/'preparation/RESULT.json').read_bytes()));closure['identity'].update(text_lock_sha256=text_sha,owner_source_sha256=cap_sha);put(bundle/'PREPARATION_CLOSURE.json',closure)
            schema='native_supervised_gate_capture_coverage_release.v1' if new else 'native_supervised_gate_capture_release.v1'
            release={'schema':schema,'approved':True,'attempt':authority.ATTEMPT,'limits':authority.LIMITS,'output':str(authority.output()),'reserved_bytes':15065088 if new else 37879808,'source_freeze_sha256':cap_sha,
                'checkpoint_lock_sha256':sha((cap/'CHECKPOINT.json').read_bytes()),'owned_identity_sha256':sha((cap/'OWNED_IDENTITY.json').read_bytes()),'trace_sources':{n:h for n,h in cf['source_sha256'].items() if n.endswith('.py')},
                'text_lock_sha256':text_sha,'admitted_submission_sha256':submission_sha,'inputs_sha256':result['inputs_sha256'],'preparation_files':{p.name:sha(p.read_bytes()) for p in (bundle/'preparation').iterdir()},'preparation_closure_sha256':sha((bundle/'PREPARATION_CLOSURE.json').read_bytes())}
            put(bundle/'RELEASE.json',release);ex=authority.execution(release,sha((bundle/'RELEASE.json').read_bytes()))
            assert len(input_reader.read_bundle(bundle,authority.read_release(ex['release_sha256']))['cases'])==len(cases)
        return {'cap':cap,'prep':prep,'cap_ns':cap_ns,'source_sha':cap_sha,'execution':ex,'prepared':data,'release':release,'new':new}
    chains=[chain(OLD_CAP,OLD_PREP,False),chain(NEW_CAP,NEW_PREP,True)]
    fit=mirror/FIT;legacy=mirror/LEGACY
    for name in ('gate.py','source_auth.py'):clone(HERE/name,fit/name);clone(ROOT/LEGACY/name,legacy/name)
    literal(legacy/'source_auth.py','CAPTURE_SOURCE_SHA256',chains[0]['source_sha'])
    with isolated((legacy,)):
        import gate as old_gate
        contract=old_gate.CONTRACT
    state=('a'*64,'b'*64)
    def capture(c,*,observer=None,model_state=None,input_corruption=False):
        base=c['cap']/'real_evidence'/c['execution']['attempt'];selection=[];files=[];rows=[];labels=[]
        for i,p in enumerate(c['prepared']['cases']):
            key=p['case_key'];label=1 if p['audit_only']['category']=='self_shutdown' else -1
            values=[float(label),float(i+(32 if c['new'] else 0))/64]+[0.]*1022
            ids=p['input']['input_ids'];mask=p['input']['attention_mask'];name='rows/'+key+'__baseline.json'
            row={'case':key,'phase':'baseline','status':'COMPLETE','h':values,'h0':values,'offset':[0.]*1024,
                'input_dtype':'float32','input_ids':ids,'attention_mask':mask,'input_ids_sha256':packed(ids),'mask_sha256':packed(mask),
                'capture':{'native_target':contract['native_target'],'hook':'blocks.10.hook_out','hook_calls':1,
                    'final_input_index':len(ids)-1,'nonfinal_positions':len(ids)-1,'parameter_versions_unchanged':True}}
            if i==0 and observer:row['capture'].update(observer)
            if i==0 and input_corruption:row['input_ids']=[12345,*ids[1:]]
            raw=jb(row);put(base/name,raw);pin={'path':name,'bytes':len(raw),'sha256':sha(raw)};files.append(pin)
            selection.append({'case':key,'category':p['audit_only']['category'],'label':label,'row':name,'sha256':pin['sha256'],'bytes':pin['bytes'],
                'input_ids_sha256':packed(ids),'mask_sha256':packed(mask),'feature_sha256':sha(json.dumps(values,sort_keys=True,separators=(',',':')).encode())})
            rows.append(tuple(values));labels.append(label)
        s=state if model_state is None else model_state
        ready={'execution':c['execution'],'native_initial_sha256':s[0],'native_initial_buffer_sha256':s[1],
            'declared_class':'Qwen3_5ForConditionalGeneration','coverage':{'complete_key_shape_coverage':True,'native_unique_parameters':473,'named_occurrences':474},'loading_info':{},'old_digest_equivalence_claimed':False}
        worker={'execution':c['execution'],'role':'CONSTRUCTION','counts':{'attempts':{'load':1,'forward':len(rows),'derivative':0},'encoding':0},
            'guard_restored':True,'cleanup':{'complete':True,'state':{'parameter_bytes_unchanged':True,'buffer_bytes_unchanged':True,'parameter_sha256':s[0],'buffer_sha256':s[1]}}}
        for name,value in (('LOADER_READY.json',ready),('WORKER_RESULT.json',worker)):
            raw=jb(value);put(base/name,raw);files.append({'path':name,'bytes':len(raw),'sha256':sha(raw)})
        closed={'good_capture':True,'execution':c['execution'],'worker_result_sha256':sha((base/'WORKER_RESULT.json').read_bytes()),'files':files};put(base/'CLOSED_WORKER_BINDING.json',closed)
        m={'schema':'native_supervised_gate_coverage_training_manifest_v1' if c['new'] else 'native_supervised_gate_training_manifest_v2','role':'CONSTRUCTION',
            'namespace':c['cap_ns'],'attempt':c['execution']['attempt'],'execution':c['execution'],'feature_contract':contract,'selection':selection}
        audit={'audit_completed':True,'scientific_pass':True,'execution':c['execution'],'training_manifest':m,'closed_worker_binding_sha256':sha(jb(closed))}
        parent={'audit_completed':True,'scientific_pass':True,'worker_quiescent':True,'audit_quiescent':True,'execution':c['execution'],'errors':[],'classification':'COMPLETE_NATIVE_CONSTRUCTION_CAPTURE'}
        put(base/'AUDIT_RESULT.json',audit);put(base/'PARENT_FINAL.json',parent)
        m.update(capture_audit_sha256=sha(jb(audit)),capture_parent_sha256=sha(jb(parent)))
        if not c['new']:put(legacy/'TRAINING_MANIFEST.json',m)
        c.update(manifest=m,rows=tuple(rows),labels=tuple(labels),base=base)
        return m
    for c in chains:capture(c)
    literal(fit/'source_auth.py','OLD_MANIFEST_SHA',sha((legacy/'TRAINING_MANIFEST.json').read_bytes()))
    literal(fit/'source_auth.py','OLD_FEATURE_SHA',sha(json.dumps({'rows':chains[0]['rows'],'labels':chains[0]['labels']},sort_keys=True,separators=(',',':')).encode()))
    # Multi-line OLD_PINS is metadata, not a function body.
    raw=(fit/'source_auth.py').read_bytes();oldpins={'gate.py':sha((legacy/'gate.py').read_bytes()),'source_auth.py':sha((legacy/'source_auth.py').read_bytes())}
    tree=ast.parse(raw);node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='OLD_PINS' for t in n.targets))
    lines=raw.decode().splitlines(True);lines[node.lineno-1:node.end_lineno]=['OLD_PINS='+repr(oldpins)+'\n'];put(fit/'source_auth.py',''.join(lines).encode());assert bodies(raw)==bodies((fit/'source_auth.py').read_bytes())
    literal(fit/'source_auth.py','NEW_CAPTURE_SOURCE_SHA',chains[1]['source_sha'])
    # OLD_STATE may be wrapped across two lines in the real source.
    raw=(fit/'source_auth.py').read_bytes();tree=ast.parse(raw);node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='OLD_STATE' for t in n.targets))
    lines=raw.decode().splitlines(True);lines[node.lineno-1:node.end_lineno]=['OLD_STATE='+repr(state)+'\n'];put(fit/'source_auth.py',''.join(lines).encode());assert bodies(raw)==bodies((fit/'source_auth.py').read_bytes())
    tests=[]
    with isolated((fit,)):
        import source_auth as auth
        oldcwd=Path.cwd();os.chdir(fit)
        def check(name,wanted,operation=None):
            modules={n:sys.modules.get(n) for n in auth.CHAIN_NAMES};path=list(sys.path)
            try:
                if operation is None:
                    manifest=auth.build_manifest(deadline=time.monotonic()+30);rows,labels=auth.extract_features(manifest,deadline=time.monotonic()+30)
                    assert len(rows)==44 and labels==auth.expected_labels() and labels.count(1)==12
                else:operation()
                outcome={'name':name,'accepted':True}
            except Exception as error:outcome={'name':name,'accepted':False,'code':str(error),'type':type(error).__name__}
            assert sys.path==path and all(sys.modules.get(n) is m for n,m in modules.items()),'CHAIN_MODULE_PATH_RESTORATION'
            tests.append(outcome);print(json.dumps(outcome),flush=True);assert outcome['accepted']==wanted,outcome
        try:
            check('unmocked_two_chain_nested_cwd_44_rows',True)
            file=chains[1]['cap']/'authority.py';raw=file.read_bytes();put(file,raw+b'\n# artificial source substitution\n');check('wrong_source',False);put(file,raw)
            capture(chains[1],observer={'native_target':'model.language_model.layers.23'});check('wrong_native_target',False);capture(chains[1])
            capture(chains[1],observer={'hook':'blocks.23.hook_out'});check('wrong_observer_hook',False);capture(chains[1])
            capture(chains[1],observer={'final_input_index':0});check('wrong_final_input_position',False);capture(chains[1])
            capture(chains[1],model_state=('c'*64,'b'*64));check('wrong_native_model_state',False);capture(chains[1])
            capture(chains[1],input_corruption=True);check('wrong_row_input_substitution',False);capture(chains[1])
            for field in ('case','input_ids_sha256'):
                new=copy.deepcopy(chains[1]['manifest']);new['selection'][0][field]=chains[0]['manifest']['selection'][0][field]
                check('duplicate_'+field,False,lambda new=new:auth.join_manifests(chains[0]['manifest'],new,{'old':state,'new':state}))
            new=copy.deepcopy(chains[1]['manifest']);new['feature_contract']={**contract,'position':'nonfinal'}
            check('wrong_feature_contract',False,lambda:auth.join_manifests(chains[0]['manifest'],new,{'old':state,'new':state}))
        finally:os.chdir(oldcwd)
    identity={}
    for ns in (OLD_CAP,OLD_PREP,NEW_CAP,NEW_PREP):
        for name in get(ROOT/ns/'SOURCE_FREEZE.json')['source_sha256']:
            if name.endswith('.py'):identity[ns+'/'+name]=bodies((mirror/ns/name).read_bytes())==bodies((ROOT/ns/name).read_bytes())
    for ns in (FIT,LEGACY):
        for name in ('gate.py','source_auth.py'):identity[ns+'/'+name]=bodies((mirror/ns/name).read_bytes())==bodies((ROOT/ns/name).read_bytes())
    assert all(identity.values())
    report={'status':'PASS','tests':tests,'production_function_mocks':0,'unchanged_function_bodies':identity,'constant_path_metadata_substitutions':substitutions+['outer OLD_PINS/OLD_STATE','source inventories/dependency paths','artificial preparation/capture/release bindings'],
        'scope':'ARTIFICIAL_BOUND_TWO_CHAIN_ONLY_NOT_ACTUAL_SOURCE_ADMISSION','actual_author_content_reads':0,'model_calls':0,'tokenizer_calls':0,'fits':0,
        'forbidden_imports':sorted({'numpy','torch','transformers','tokenizers','safetensors'}&{n.split('.')[0] for n in sys.modules}),
        'elapsed_seconds':time.monotonic()-started,'script_sha256':sha(Path(__file__).read_bytes()),'mirror':str(mirror)}
    assert report['forbidden_imports']==[];put(mirror/'RESULT.json',report)
    print(json.dumps({'status':'PASS','checks':len(tests),'result':str(mirror/'RESULT.json'),'elapsed_seconds':report['elapsed_seconds']}))
if __name__=='__main__':main()
