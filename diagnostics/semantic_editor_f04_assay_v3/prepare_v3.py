"""Prospective V3 wrapper-grace lock; reuse previous tests, never execute their batches."""
import ast,difflib,json,sys,time
from core import HERE,ROOT,Budget,read,require,sha,git,check_freeze
from parent_binding import reuse
PARENT='diagnostics/semantic_editor_f04_assay_v2'
COMMIT='5cdebf01b917d5e813fa3ef068d679888b8ad5cf'
INVENTORY='434739235f2faaac1a0e35724fa3a92c70161726b79d51aed4897169c02d4c45'
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
    require(sys.argv[1:]==['freeze'] and not (HERE/'freeze.json').exists(),'one new prospective lock')
    for path in HERE.glob('*.py'):ast.parse(path.read_bytes(),filename=path.name)
    inherited=reuse()
    raw=git('show',COMMIT+':'+PARENT+'/FINAL_INVENTORY.json');require(sha(raw)==INVENTORY,'immutable failed V2 inventory')
    entries={r['path']:r for r in json.loads(raw)['files']}
    same={}
    def parent(name):
        data=(ROOT/PARENT/name).read_bytes();row=entries[name]
        require(sha(data)==row['sha256'] and len(data)==row['bytes'],'authenticated V2 '+name)
        return data
    for path in HERE.iterdir():
        if path.is_file() and path.name in entries:
            original=parent(path.name)
            if path.read_bytes()==original:same[path.name]=sha(original)
    require(all(n in same for n in ('native.py','editor.py','run.py','entry.py','saved_judge.py','admission.py','inputs.py','input_lock.json','inputs.json','production_plan.json','tokens_01.json','tokens_02.json','source_bindings.json','fitted_parameters.json','final_adjudication.py','guard_candidate.py','hook_record.py')),'unchanged native/science/input/runtime')
    tests=json.loads(parent('pure_cleanup_results.json'))
    require(len(tests)==12 and all(r['status']=='PASS' for r in tests),'twelve inherited cleanup tests')
    require(json.loads(parent('cleanup_batch_receipt.json'))['status']=='INCONCLUSIVE_FIXTURE','V2 failure remains final')
    budget.write('parent_reuse_receipt.json',{'parent_commit':COMMIT,'inventory_sha256':INVENTORY,'byte_identical_v2_files':same,
        'inherited_pure_tests':{'count':12,'sha256':sha(parent('pure_cleanup_results.json')),'rerun':False},
        'original_f04_matrix_and_normal_captures':inherited,'no_model_tokenizer_scores_or_fits':True})
    diff=''
    for name in ('owned.py','owned_capture.py'):
        diff+=''.join(difflib.unified_diff(parent(name).decode().splitlines(True),(HERE/name).read_text().splitlines(True),fromfile='immutable_v2/'+name,tofile='successor_v3/'+name))
    budget.write_bytes('NARROW_DIFF.patch',diff.encode())
    budget.write('supervision_resource_proof.json',{'stop_episode':{'absolute_seconds':3,'worker_verification_max':1,'wrapper_grace_max':1,'forced_launcher_verification_max':1,'grace_requires_actual_exit_proof':True,'nonrenewable':True},
        'unchanged_outer_waits':{'terminate':3,'kill':3,'capture_joins':4,'evidence_join':1},
        'conservative_wait_total_seconds':14,'unchanged_cleanup_envelope_seconds':15,'overhead_headroom_seconds':1,
        'overhead_is_measured_not_guaranteed':True,'observed_overrun_is_INCONCLUSIVE':True,
        'future_limits':read(HERE/'production_plan.json')['limits'],'future_record_bound_bytes':86325248,
        'ownership_event_cap_unchanged':64,'ownership_writer_cap_unchanged':66,'normal_stop_event_bound':26,
        'prep_seconds':180,'prep_bytes':32*1024**2,'file_bytes':5*1024**2})
    names=sorted(p.name for p in HERE.iterdir() if p.is_file() and p.name not in ('authorization.json','approved_root_release.json','freeze.json'))
    budget.write('freeze.json',{'source_sha256':{n:sha((HERE/n).read_bytes()) for n in names},'scope':'owned wrapper completion grace only; no model authorization',
        'production_authorized':False,'fake_batch':'six fixed deterministic new cases plus one unchanged actual-facade fake deadline','one_batch_no_retry':True})
    print(json.dumps({'status':'PROSPECTIVE_SOURCE_LOCKED','freeze_sha256':sha((HERE/'freeze.json').read_bytes()),'files':len(names),'elapsed_seconds':time.monotonic()-started}))
if __name__=='__main__':main()
