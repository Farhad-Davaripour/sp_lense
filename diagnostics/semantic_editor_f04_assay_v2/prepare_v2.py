"""Bind the unchanged parent, then prospectively freeze only this cleanup successor."""
import ast,difflib,json,sys,time
from core import HERE,ROOT,Budget,read,require,sha,check_freeze
from parent_binding import reuse,PARENT,COMMIT,INVENTORY

def preflight():
    from admission import exact_cohort,environment,admit_production
    from inputs import build_plan
    lock=check_freeze();plan=build_plan();exact_cohort(plan);environment(plan)
    authorization=read(HERE/'authorization.json')
    require(authorization['run_authorized'] is False and authorization['root_release_sha256'] is None,'production remains disabled')
    try:admit_production()
    except ValueError as error:require('NO REAL MODEL RELEASE' in str(error),'expected disabled admission failure')
    else:raise ValueError('production unexpectedly admitted')
    require(not any(n=='torch' or n.startswith('transformer_lens') for n in sys.modules),'no model library imports')
    return {'status':'PASS_ZERO_MODEL_PREFLIGHT','source_sha256':sha((HERE/'freeze.json').read_bytes()),
            'input_lock_sha256':sha((HERE/'input_lock.json').read_bytes()),'plan_unchanged':True,
            'counts':{'future_forwards':42,'future_derivatives':16,'fresh_routes':6,'endpoints':4},
            'model_calls':0,'tokenizer_calls':0,'gate_scores':0,'production_authorized':False}

def main():
    started=time.monotonic();budget=Budget(HERE)
    if sys.argv[1:]==['preflight']:
        result=preflight();result['elapsed_seconds']=time.monotonic()-started
        budget.write('zero_model_preflight.json',result);print(json.dumps(result));return
    require(sys.argv[1:]==['freeze'] and not (HERE/'freeze.json').exists(),'one prospective source freeze')
    for path in HERE.glob('*.py'):ast.parse(path.read_bytes(),filename=path.name)
    receipt=reuse();budget.write('parent_reuse_receipt.json',receipt)
    require(read(HERE/'production_plan.json')['limits']=={'forwards':42,'derivatives':16,'loads':1,'worker_seconds':300,'cleanup_seconds':15,'audit_seconds':90,'bytes':96*1024**2},'future envelope unchanged')
    diff=''
    for name in ('native.py','owned.py','owned_capture.py','admission.py'):
        diff+=''.join(difflib.unified_diff((ROOT/PARENT/name).read_text().splitlines(True),(HERE/name).read_text().splitlines(True),fromfile='immutable_v1/'+name,tofile='successor_v2/'+name))
    budget.write_bytes('NARROW_DIFF.patch',diff.encode())
    budget.write('successor_manifest.json',{'parent_commit':COMMIT,'parent_inventory_sha256':INVENTORY,
        'changed_runtime_only':['native.py','owned.py','owned_capture.py'],
        'necessary_provenance_binding_only':['admission.py','parent_binding.py'],
        'scientific_input_token_method_plan_unchanged':True,'pure_fixtures':12,'live_deadline_fixtures':1,
        'prior_suites_rerun':False,'prep_invoked_seconds':180,'prep_bytes':32*1024**2,'file_bytes':5*1024**2,
        'future_limits_unchanged':read(HERE/'production_plan.json')['limits'],
        'ownership_record_bound':{'normal_events':5,'deadline_events_conservative':24,'writer_threads_conservative':26,'unchanged_event_cap':64,'unchanged_writer_cap':66},
        'production_authorized':False})
    names=sorted(p.name for p in HERE.iterdir() if p.is_file() and p.name not in ('authorization.json','approved_root_release.json','freeze.json'))
    budget.write('freeze.json',{'source_sha256':{n:sha((HERE/n).read_bytes()) for n in names},
        'scope':'cleanup-only f04 successor; no model authorization','production_authorized':False,
        'one_batch_no_retry':True,'fake_batch':'exactly12 pure cleanup tests plus one live facade deadline; no other traversal'})
    print(json.dumps({'status':'PROSPECTIVE_SOURCE_LOCKED','freeze_sha256':sha((HERE/'freeze.json').read_bytes()),'files':len(names),'elapsed_seconds':time.monotonic()-started}))
if __name__=='__main__':main()
