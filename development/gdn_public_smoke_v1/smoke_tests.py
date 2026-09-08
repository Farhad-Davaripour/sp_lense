"""Narrow ordinary engineering tests: inert callbacks and saved bytes only."""
import ast
import copy
import json
import pathlib
import sys
import time
import types
HERE=pathlib.Path(__file__).resolve().parent
class Block:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','transformers','transformer_lens','numpy','tokenizers','safetensors','sp_lense'}:
            raise RuntimeError('FORBIDDEN_MODEL_IMPORT')
sys.meta_path.insert(0,Block())
import support
import constructor_operands as operands
import constructor_reader as reader
def need(value,code):
    if not value:raise RuntimeError(code)
def rejected(call):
    try:call()
    except (ValueError,RuntimeError,OSError,KeyError):return True
    raise RuntimeError('EXPECTED_REJECTION')
def run():
    started=time.monotonic();results=[]
    support.check_freeze()
    from admission import admit
    from plan import build_plan
    from diagnostic_support import exact_input,CELL_ID
    from diagnostic_counter import Counter
    binding=admit();plan=build_plan();locked,cell=exact_input(plan)
    need(locked['input_ids']==locked['attention_mask']==[1] and locked['final_input_index']==0 and len(plan['cells'])==1 and not plan['requests'],'PUBLIC_ONLY')
    counter=Counter(CELL_ID);counter.reserve_load();counter.consume_load();counter.reserve_forward(CELL_ID)
    rejected(lambda:counter.reserve_load());rejected(lambda:counter.reserve_forward(CELL_ID))
    from helper_binding import ensure_package
    package=ensure_package()
    need(package['admission'].__file__.endswith('admission.py'),'PRIVATE_ADMISSION_BINDING')
    need(sum(operands.FILE_CAPS.values())==65536,'EXACT_RECEIPT_RESERVE')
    from launch import preflight
    disabled=preflight('0'*64);need(not disabled['production_authorized'],'DISABLED_REAL_AUTHORITY')
    results.append({'case':'public_input_counter_source_admission','status':'PASS','preflight':disabled})
    tree=ast.parse((HERE/'candidate_loader.py').read_bytes());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_diagnostic_load_adapter')
    body=next(x for x in fn.body if isinstance(x,ast.Try)).body
    begin=next(i for i,x in enumerate(body) if isinstance(x,ast.ImportFrom) and x.module=='constructor_operands')
    selected=body[begin:begin+4]
    need(len(selected)==4 and isinstance(selected[2],ast.Assign),'ACTUAL_LOAD_WINDOW')
    original_snapshot=operands.loaded_snapshot
    clean=None
    for name in ('clean','post_constructor_rejection','partial_publication'):
        folder=HERE/'test_evidence'/('run_'+str(time.time_ns()))/name;control=folder/'control';control.mkdir(parents=True)
        execution={'mode':'INERT_FIXTURE','actual_model_work':False,'case':name};events=[]
        support.run_dir=lambda:folder
        native=support.write_new
        def publish(key,payload,**kw):
            if name=='partial_publication' and key=='CONSTRUCTOR_AFTER.json':
                from real_boundary import write_exclusive
                write_exclusive(control/key,payload[:19],folder)
                raise OSError('INERT_IO')
            return native(key,payload,**kw)
        recorder=operands.Operands(execution,publish,lambda code:events.append('stop'))
        count=[0]
        def snapshot():
            count[0]+=1;events.append('before' if count[0]==1 else 'after')
            return {'observational_only':True,'conditions':{'function_type':True,'code_equal':not(name=='post_constructor_rejection' and count[0]==2),'resolved_path_equal':True,'different_code_fields':[]}}
        operands.loaded_snapshot=snapshot
        token=object()
        def load(cfg,with_lens):
            need(cfg=='INERT_CONFIG' and with_lens is False,'UNCHANGED_LOAD_ARGUMENTS');events.append('load');return token
        env={'writer':types.SimpleNamespace(constructor_operands=recorder),'backend_module':types.SimpleNamespace(ResearchBackend=types.SimpleNamespace(load=load)),
             'config':types.SimpleNamespace(load_config=lambda path:'INERT_CONFIG'),'ROOT':HERE,'spec':{'model':{'config_path':'INERT_NO_WEIGHTS'}}}
        try:
            exec(compile(ast.Module(body=copy.deepcopy(selected),type_ignores=[]),str(HERE/'candidate_loader.py'),'exec'),env)
            need(env['backend'] is token and events[:3]==['before','load','after'],'ORIGINAL_RETURN_AND_PHASE_ORDER')
            if name=='post_constructor_rejection':
                error=package['source_contract'].Rejected('CALLABLE_CODE_IDENTITY')
                recorder.fail('LOAD_OR_ADMISSION',error);recorder.finish();events.append('cleanup_failure');recorder.fail('OUTER_CLOSEOUT',OSError())
                need(recorder.primary['predicate']=='CALLABLE_CODE_IDENTITY','FIRST_CAUSE_SURVIVES_CLEANUP')
            else:recorder.finish()
        except OSError:
            need(name=='partial_publication','ONLY_EXPECTED_IO');recorder.finish()
        status=recorder.status();proof=reader.verify_operands(control,execution,status,copy.deepcopy(status))
        if name=='partial_publication':need(not proof['binding_verified'] and recorder.io_failed and (control/'CONSTRUCTOR_AFTER.json').stat().st_size==19,'PARTIAL_PRESERVED_NO_SUCCESS')
        else:need(proof['binding_verified'],'COMPLETE_PHASE_JOIN')
        if name=='clean':clean=(control,execution,status)
        results.append({'case':name,'status':'PASS','events':events,'proof':proof,'live_status':status})
    operands.loaded_snapshot=original_snapshot
    control,execution,status=clean
    # Coherent outer rehash reaches the inner phase check, not just raw hash.
    after=control/'CONSTRUCTOR_AFTER.json';raw=after.read_bytes();value=json.loads(raw);value['phase']='BEFORE'
    changed=(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode()
    # Saved-only synthetic transformation uses a fresh output, never rewrites clean evidence.
    target=HERE/'test_evidence'/('tamper_'+str(time.time_ns()))/'control';target.mkdir(parents=True)
    tampered=copy.deepcopy(status)
    for key,pointer in tampered['pointers'].items():
        data=changed if key=='CONSTRUCTOR_AFTER.json' else (control/key).read_bytes()
        if key=='CONSTRUCTOR_TERMINAL.json':
            terminal=json.loads(data);terminal['prior_pointers']={k:v for k,v in tampered['pointers'].items() if k!=key};data=(json.dumps(terminal,sort_keys=True,separators=(',',':'))+'\n').encode()
        (target/key).write_bytes(data)
        pointer.update(path=(target/key).relative_to(HERE).as_posix(),bytes=len(data),sha256=support.sha(data))
    rejected(lambda:reader.verify_operands(target,execution,tampered,copy.deepcopy(tampered)))
    results.append({'case':'coherent_saved_phase_tamper','status':'PASS','synthetic_identity_rebinding':True})
    need(time.monotonic()-started<45,'ENGINEERING_TEST_DEADLINE')
    return {'status':'PASS_MODEL_FREE_HANDOFF_ONLY','elapsed_seconds':time.monotonic()-started,'checks':results,
            'model_imports':0,'model_loads':0,'weight_accesses':0,'actual_forwards':0,'derivatives':0,'encodings':0,'real_authority':False,'scientific_pass':False}
if __name__=='__main__':
    result=run();print(json.dumps(result,sort_keys=True))
