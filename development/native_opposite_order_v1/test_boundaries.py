"""Provider-free actual authority/counter/budget tests in an isolated fixture tree."""
import json,os,sys,time
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
class NoModels:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in {'torch','transformers','transformer_lens','pyarrow','datasets','numpy','safetensors'}:raise RuntimeError('MODEL_FREE_BOUNDARY')
sys.meta_path.insert(0,NoModels())
def need(ok,code):
    if not ok:raise RuntimeError(code)
def denied(call):
    try:call()
    except (OSError,ValueError):return
    raise RuntimeError('REJECTION_REQUIRED')
def main():
    import authority,support
    from counts import Counts
    from setup_budget import Budget
    from loader import DenyUnusedDependencies
    deny=DenyUnusedDependencies()
    for name in ('transformer_lens','datasets.arrow_dataset','pyarrow.lib'):
        try:deny.find_spec(name)
        except ImportError:pass
        else:raise RuntimeError('DEPENDENCY_FINDER_MUST_DENY')
    root=HERE/'test_evidence'/('boundaries_'+str(time.time_ns()));root.mkdir(parents=True,exist_ok=False)
    attempt=root/'attempt';names=('inputs.json','CHECKPOINT.json','OWNED_IDENTITY.json')
    for name in names:(root/name).write_bytes(b'{}\n')
    source=root/'frozen.py';source.write_bytes(b'BOUNDARY_FIXTURE = True\n')
    freeze={'source_sha256':{'frozen.py':support.sha(source.read_bytes())},'external_sources':[]}
    (root/'SOURCE_FREEZE.json').write_bytes(support.json_bytes(freeze))
    release={'schema':'native_baseline_release.v1','approved':False,'attempt':support.ATTEMPT,'limits':authority.LIMITS,'output':str(attempt),
        'source_freeze_sha256':support.sha((root/'SOURCE_FREEZE.json').read_bytes()),
        'inputs_sha256':support.sha((root/'inputs.json').read_bytes()),'checkpoint_lock_sha256':support.sha((root/'CHECKPOINT.json').read_bytes()),
        'owned_identity_sha256':support.sha((root/'OWNED_IDENTITY.json').read_bytes())}
    release_path=root/'ISOLATED_TEST_RELEASE.json';release_path.write_bytes(support.json_bytes(release));results=['blocked_dependency_finder_rejects_before_resolution']
    saved_env={k:os.environ.get(k) for k in ('SP_NATIVE_RELEASE_SHA','SP_NATIVE_ADMISSION_SHA')}
    try:
        with patch.object(authority,'HERE',root),patch.object(authority,'RELEASE',release_path),patch.object(support,'HERE',root), \
             patch.object(authority,'output',lambda:attempt),patch.object(support,'output',lambda:attempt):
            denied(lambda:authority.read_release(support.sha(release_path.read_bytes())))
            results.append('unapproved_release_rejected')
            release['approved']=True;release_path.write_bytes(support.json_bytes(release));approved=support.sha(release_path.read_bytes())
            denied(lambda:authority.read_release('0'*64));results.append('wrong_release_hash_rejected')
            before=source.read_bytes();source.write_bytes(b'BOUNDARY_FIXTURE = False\n');denied(lambda:authority.read_release(approved));source.write_bytes(before)
            results.append('frozen_source_mutation_rejected')
            admission=authority.admit_once(approved);need(authority.authenticate()['execution']==admission['execution'],'EXACT_ADMITTED_IDENTITY')
            denied(lambda:authority.admit_once(approved));results.append('one_shot_admission_and_no_retry')
            os.environ['SP_NATIVE_ADMISSION_SHA']='0'*64;denied(authority.authenticate);results.append('admission_hash_rejected')
    finally:
        for key,value in saved_env.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
    recorded=[];counts=Counts(time.monotonic()+5,lambda name,row:recorded.append((name,row)))
    for kind,cap in (('load',1),('forward',28),('derivative',8)):
        for _ in range(cap):counts.reserve(kind)
        denied(lambda:counts.reserve(kind))
    need(len(recorded)==37,'EXACT_MAX_DISPATCH_RECORDS');results.append('one_load_28_forward_8_derivative_caps')
    expired=Counts(time.monotonic()-1,lambda *a:None);denied(lambda:expired.reserve('forward'));results.append('expired_dispatch_rejected')
    budget=Budget(0.);worker,cleanup=budget.deadlines(1.,'worker');need(worker==601. and cleanup==616.,'WORKER_ENVELOPE')
    budget.charge('worker',10.);audit,audit_cleanup=budget.deadlines(603.,'audit');need(audit==660. and audit_cleanup==665.,'SHARED_NOT_RENEWED')
    budget.charge('audit',5.);denied(lambda:budget.deadlines(604.,'audit'));results.append('one_shared_cleanup_allowance')
    from workflow import all_unrun
    need(len(all_unrun())==28 and all(v['status']=='UNRUN' for v in all_unrun()),'PRE_SCHEDULE_UNRUN')
    need(sum(support.GROUP_CAPS.values())==33382400 and sum(support.GROUP_CAPS.values())<32*1024**2,'COMPLETE_STORAGE_RESERVATION')
    results.extend(('pre_schedule_28_unrun','bounded_storage_groups_fit_32mib'))
    result={'status':'PASS','cases':results,'case_count':len(results),'providers_imported':False,'model_work':False,
        'fixture_authority_only':True,'real_release_created':False,'directory':str(root)}
    (HERE/'TEST_BOUNDARIES_RESULT.json').write_bytes(support.json_bytes(result));print(json.dumps(result,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
