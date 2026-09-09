"""Actual Windows owner tests with local fake children only."""
import json,subprocess,sys,time
from pathlib import Path
BLOCKED={'torch','transformers','tokenizers','transformer_lens','datasets','pyarrow','safetensors'}
class NoProviders:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in BLOCKED:raise RuntimeError('NO_REAL_PROVIDER')
sys.meta_path.insert(0,NoProviders())
from owner import HERE,run,verify,sha,jb,sizes,require
def main():
    source_sha=verify();config=json.loads((HERE/'OWNED_IDENTITY.json').read_bytes())
    root=HERE/'test_evidence'/('owned_'+str(time.time_ns()));root.mkdir(parents=True);reports=[]
    for mode in ('success','nonzero','timeout','stdout_overflow','stderr_overflow'):
        out=root/mode;prep=root/(mode+'_fake_preparation')
        command=[config['launch_image'],'-B',str(HERE/'fake_child.py'),mode,str(prep)]
        result=run(command,config,out,prep,{'owner_source_sha256':source_sha,'text_lock_sha256':'0'*64,'synthetic_only':True},
            wait_seconds=1. if mode=='timeout' else 5.,cleanup_seconds=2.)
        require((out/'CLOSURE.json').is_file(),'ACTUAL_CLOSURE_PUBLISHED_'+mode)
        require(result['status']==('PASS' if mode=='success' else 'FAIL'),'EXPECTED_'+mode+':'+str(result))
        require(result['actual_authenticated'] and result['quiescent'] and result['within_deadline'],'ACTUAL_HANDLES_CLOSED_'+mode+':'+str(result))
        require(all(p['valid_retained_handle'] and p['signaled'] and p['query_success'] for p in result['exit_proofs'].values()),'RETAINED_EXIT_PROOFS')
        require(result['binding']['actual_worker']['pid']!=result['binding']['launcher']['pid'],'VENV_REAL_CHILD')
        require(sizes(out)<=32768 and sizes(out)+sizes(prep)<=16*1024**2,'ACTUAL_COMBINED_STORAGE')
        if mode=='success':require(result['exit_code']==0 and result['termination'] is None,'NATURAL_CLEAN_EXIT')
        if mode=='nonzero':require(result['exit_code']==7,'EXACT_NONZERO_EXIT')
        if mode=='timeout':require(result['timed_out'] and result['termination']['api_success'],'BOUNDED_JOB_TIMEOUT')
        if 'overflow' in mode:require(any(d['overflow'] and d['retained_bytes']==8192 for d in result['drains']),'BOUNDED_CAPTURE_PREFIX')
        reports.append({'case':mode,'status':'PASS','closure_sha256':sha((out/'CLOSURE.json').read_bytes()),
            'actual_exit_codes':{k:p['exit_code'] for k,p in result['exit_proofs'].items()},'observed':result})
    before=(root/'success/CLOSURE.json').read_bytes()
    try:run([],config,root/'success',root/'never_created',{},wait_seconds=1.,cleanup_seconds=1.)
    except FileExistsError:pass
    else:raise RuntimeError('RETRY_ACCEPTED')
    require((root/'success/CLOSURE.json').read_bytes()==before and not (root/'never_created').exists(),'ONE_SHOT_NO_OVERWRITE')
    reports.append({'case':'exclusive_one_shot_no_retry','status':'PASS'})
    wrong=dict(config);wrong['launch_sha256']='0'*64
    refused=run([config['launch_image'],'-B',str(HERE/'fake_child.py'),'success',str(root/'not_started')],wrong,
        root/'rejected_suspended_launcher',root/'not_started',{'synthetic_only':True},wait_seconds=2.,cleanup_seconds=2.)
    require(refused['status']=='FAIL' and refused['assigned_before_resume'] is False and not (root/'not_started').exists()
        and refused['exit_proofs']['launcher']['signaled'] and refused['exit_proofs']['launcher']['query_success']
        and refused['exit_proofs']['launcher']['exit_code']==125 and refused['job_empty_before_close'],'PRE_ASSIGNMENT_RETAINED_CLEANUP')
    reports.append({'case':'preassignment_suspended_launcher_refusal_cleanup','status':'PASS','observed':refused})
    child=subprocess.run([sys.executable,'-B',str(HERE/'owner.py')],capture_output=True,text=True,timeout=10)
    require(child.returncode==2 and json.loads(child.stdout)['status']=='DISABLED_NO_ROOT_PREPARATION_RELEASE','NO_REAL_ADMISSION')
    reports.append({'case':'disabled_without_root_release','status':'PASS'})
    child=subprocess.run([sys.executable,'-B',str(HERE/'owner.py'),'--owner-source-sha256',source_sha,
        '--approved-preparation-sha256','0'*64],capture_output=True,text=True,timeout=10)
    require(child.returncode==2 and json.loads(child.stdout)['status']=='OWNER_ADMISSION_FAILURE'
        and not (HERE/'ownership_attempt_001').exists(),'STORAGE_BLOCK_PREVENTS_REAL_LAUNCH')
    reports.append({'case':'storage_blocked_production_admission','status':'PASS'})
    require(not any(n.split('.')[0] in BLOCKED for n in sys.modules),'MODEL_TOKENIZER_FREE')
    result={'status':'PASS','groups':reports,'group_count':len(reports),'source_sha256':source_sha,
        'real_preparation_launches':0,'tokenizer_calls':0,'model_calls':0,'actual_cohort_access':False,'real_release_created':False}
    (HERE/'TEST_RESULTS.json').write_bytes(jb(result));print(json.dumps({'status':'PASS','groups':len(reports),'source_sha256':source_sha,'test_root':str(root)}));return 0
if __name__=='__main__':raise SystemExit(main())
