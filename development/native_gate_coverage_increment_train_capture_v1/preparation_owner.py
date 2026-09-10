"""One bounded hidden Windows preparation owner; no tokenizer/model imports."""
import hashlib,json,os,subprocess,sys,threading,time
from pathlib import Path
from windows_job import Job,native,require
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PREP=ROOT/'development/native_gate_coverage_increment_train_preparation_v1'
TOTAL=32*1024**2;FILE=5*1024**2;CAPTURE=8192;OWNER_CAP=32768
PREPARATION_CAP=TOTAL-OWNER_CAP
COMBINED_STORAGE_ADMISSION_READY=True
def sha(raw):return hashlib.sha256(raw).hexdigest()
def jb(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def verify(expected=None):
    raw=(HERE/'SOURCE_FREEZE.json').read_bytes()
    if expected is not None:require(sha(raw)==expected,'OWNER_SOURCE_HASH')
    lock=json.loads(raw)
    for n,h in lock['source_sha256'].items():require(sha((HERE/n).read_bytes())==h,'OWNER_SOURCE_BYTES')
    for pin in lock['external_sources']:require(sha(Path(pin['path']).read_bytes())==pin['sha256'],'OWNER_EXTERNAL_SOURCE_BYTES')
    return sha(raw)
def sizes(base):
    values=[p.stat().st_size for p in base.rglob('*') if p.is_file()] if base.exists() else []
    require(all(n<=FILE for n in values),'COMBINED_PER_FILE_CAP');return sum(values)
def owner_write(out,name,value,*,raw=False):
    limits={'ADMISSION.json':4096,'stdout.bin':8192,'stderr.bin':8192,'CLOSURE.json':12288}
    require(name in limits,'EXACT_OWNER_ARTIFACT_NAME')
    data=value if raw else jb(value)
    require(type(data) is bytes and len(data)<=limits[name],'OWNER_ARTIFACT_CAP')
    require(sizes(out)+len(data)<=OWNER_CAP,'OWNER_PREWRITE_TOTAL_CAP')
    with (out/name).open('xb') as f:
        require(f.write(data)==len(data),'OWNER_SHORT_WRITE');f.flush();os.fsync(f.fileno())
class Drain:
    def __init__(self,pipe,stop):
        self.pipe,self.stop=pipe,stop;self.data=bytearray();self.seen=0;self.overflow=False;self.error=None;self.eof=False
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    def run(self):
        try:
            while True:
                raw=os.read(self.pipe.fileno(),4096)
                if not raw:self.eof=True;break
                self.seen+=len(raw);self.data.extend(raw[:max(0,CAPTURE-len(self.data))])
                if self.seen>CAPTURE:self.overflow=True;self.stop.set()
        except BaseException as error:self.error=type(error).__name__;self.stop.set()
    def record(self):return {'observed_bytes':self.seen,'retained_bytes':len(self.data),'overflow':self.overflow,'error_type':self.error,
        'eof':self.eof,'thread_joined':not self.thread.is_alive()}
def run(command,config,out,prep_out,identity,*,wait_seconds=350.,cleanup_seconds=5.):
    require(0<wait_seconds<=350 and 0<cleanup_seconds<=5,'FINITE_OWNER_LIMITS')
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic();wait_end=started+wait_seconds;absolute_end=wait_end+cleanup_seconds
    require(not prep_out.exists(),'NO_PREVIOUS_PREPARATION_ATTEMPT')
    receipt={'schema':'root_preparation_closure.v1','status':'FAIL','one_shot':True,'command':command,'identity':identity,
        'text_lock_sha256':identity.get('text_lock_sha256'),'started_monotonic':started,'wait_deadline':wait_end,'absolute_deadline':absolute_end,
        'quiescent':False,'timed_out':False,'exit_code':None,'primary_error':None,'cleanup_errors':[],
        'creationflags':'CREATE_NO_WINDOW|CREATE_SUSPENDED','assigned_before_resume':False,'actual_authenticated':False}
    owner_write(out,'ADMISSION.json',{'started':started,'command':command,'identity':identity,'one_shot':True})
    process=job=launcher=actual=helper=pending=None;drains=[];binding=None;stop=threading.Event();termination=None;cleanup_started=None;helper_identity=None
    try:
        n=native();job=Job()
        process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW|0x4,close_fds=True)
        launcher=n.NativeHandle(int(process._handle),'preparation_launcher_retained',True)
        launched=launcher.snapshot();require(launched['pid']==process.pid and launched['ppid']==os.getpid()
            and os.path.normcase(launched['image'])==os.path.normcase(config['launch_image'])
            and launched['image_sha256']==config['launch_sha256'],'PINNED_LAUNCHER_HANDLE_IDENTITY')
        drains=[Drain(process.stdout,stop),Drain(process.stderr,stop)]
        job.assign_resume(launcher.handle);receipt['assigned_before_resume']=True
        while True:
            now=time.monotonic()
            require(sizes(prep_out)<=PREPARATION_CAP and sizes(out)<=OWNER_CAP,'DISJOINT_PREPARATION_OWNER_PARTITIONS')
            if now>=wait_end:receipt['timed_out']=True;raise ValueError('EXTERNAL_PREPARATION_DEADLINE')
            if stop.is_set():raise ValueError('CAPTURE_OVERFLOW_OR_DRAIN_FAILURE')
            members=job.pids()
            known={process.pid}
            if actual is not None:known.add(binding['actual_worker']['pid'])
            if helper is not None:known.add(helper_identity['pid'])
            for child_pid in set(members)-known:
                    candidate=n.open_claimed_worker(child_pid);pending=candidate;observed=candidate.snapshot()
                    receipt['candidate_identity']=observed
                    is_helper=os.path.normcase(observed['image'])==os.path.normcase(config['console_image'])
                    image_key='console' if is_helper else 'base'
                    receipt['candidate_checks']={'owned_job':job.contains(candidate.handle),'parent':observed['ppid']==launched['pid'],
                        'creation_order':observed['creation_filetime']>=launched['creation_filetime'],'live':observed['live'],
                        'image':os.path.normcase(observed['image'])==os.path.normcase(config[image_key+'_image']),
                        'image_sha256':observed['image_sha256']==config[image_key+'_sha256'],
                        'single_role':helper is None if is_helper else actual is None}
                    try:
                        require(all(receipt['candidate_checks'].values()),'OWNED_JOB_ACTUAL_CREATION_IMAGE_IDENTITY')
                    except BaseException:candidate.close();raise
                    if is_helper:
                        helper=candidate;helper.tag='owned_console_helper_retained';helper_identity=observed
                    else:
                        actual=candidate;binding={'launcher':launched,'actual_worker':observed,'owned_job_membership':True,
                            'creation_identity_before_exit':True,'no_pid_only_authority':True};receipt['actual_authenticated']=True
                    pending=None
            if actual is not None:
                if actual.exited() and launcher.exited() and (helper is None or helper.exited()):break
            elif launcher.exited():raise ValueError('NO_RETAINED_ACTUAL_IDENTITY_BEFORE_EXIT')
            stop.wait(.005)
    except BaseException as error:receipt['primary_error']={'type':type(error).__name__,'code':str(error) if type(error) is ValueError else 'OWNER_FAILURE'}
    finally:
        cleanup_started=time.monotonic();cleanup_end=min(absolute_end,cleanup_started+cleanup_seconds)
        try:
            if job is not None:
                if job.pids():termination=job.terminate()
                # Assignment/image checks can fail while our launcher is still
                # suspended outside the job; retain-handle cleanup must cover it.
                if launcher is not None and not job.contains(launcher.handle) and not launcher.exited():termination=launcher.terminate()
            elif launcher is not None and not launcher.exited():termination=launcher.terminate()
            if process is not None and launcher is None and process.poll() is None:
                process.terminate() # Windows Popen uses its original retained handle.
        except BaseException as error:receipt['cleanup_errors'].append({'phase':'terminate','type':type(error).__name__})
        proofs={}
        for key,handle in (('actual_worker',actual),('launcher',launcher),('console_helper',helper)):
            if handle is not None:
                try:proofs[key]=handle.exit_proof(max(0.,cleanup_end-time.monotonic()-.4))
                except BaseException as error:receipt['cleanup_errors'].append({'phase':'exit_'+key,'type':type(error).__name__})
        for drain in drains:drain.thread.join(max(0.,cleanup_end-time.monotonic()-.2))
        pipes_closed=True
        if process is not None:
            for pipe in (process.stdout,process.stderr):
                try:pipe.close()
                except BaseException:pipes_closed=False
        empty=False
        if job is not None:
            try:empty=not job.pids();job.close()
            except BaseException as error:receipt['cleanup_errors'].append({'phase':'job_close','type':type(error).__name__})
        for handle in (actual,helper,pending):
            if handle is not None:
                try:handle.close()
                except BaseException as error:receipt['cleanup_errors'].append({'phase':'actual_handle_close','type':type(error).__name__})
        if process is not None:
            try:process.poll();process._handle.Close()
            except BaseException as error:receipt['cleanup_errors'].append({'phase':'launcher_handle_close','type':type(error).__name__})
        drained=[d.record() for d in drains]
        receipt.update(binding=binding,console_helper_identity=helper_identity,termination=termination,exit_proofs=proofs,drains=drained,pipes_closed=pipes_closed,
            job_empty_before_close=empty,cleanup_started_monotonic=cleanup_started,cleanup_seconds=time.monotonic()-cleanup_started)
        receipt['quiescent']=bool(receipt['actual_authenticated'] and len(proofs)==(3 if helper is not None else 2) and all(p['valid_retained_handle'] and p['signaled'] and p['query_success'] for p in proofs.values())
            and empty and len(drained)==2 and all(d['eof'] and d['thread_joined'] for d in drained) and pipes_closed)
        receipt['exit_code']=proofs.get('actual_worker',{}).get('exit_code')
        for name,d in zip(('stdout.bin','stderr.bin'),drains):owner_write(out,name,bytes(d.data),raw=True)
        try:
            result_path=prep_out/'RESULT.json'
            if result_path.exists():
                require(result_path.stat().st_size<=8192,'SMALL_TERMINAL_PREPARATION_RECEIPT');raw=result_path.read_bytes()
                receipt['preparation_result_sha256']=sha(raw);receipt['preparation_status']=json.loads(raw).get('status')
            else:receipt['preparation_result_sha256']=None;receipt['preparation_status']=None
            receipt['elapsed_seconds']=time.monotonic()-started
            receipt['within_deadline']=receipt['elapsed_seconds']<=wait_seconds+cleanup_seconds and receipt['cleanup_seconds']<=cleanup_seconds
            success=receipt['quiescent'] and receipt['within_deadline'] and not receipt['timed_out'] and receipt['primary_error'] is None
            success=success and not receipt['cleanup_errors'] and all(p['exit_code']==0 for p in proofs.values())
            success=success and all(not d['overflow'] and d['error_type'] is None for d in drained) and receipt['preparation_status']=='PASS'
            receipt['status']='PASS' if success else 'FAIL';receipt['combined_bytes_before_closure']=sizes(prep_out)+sizes(out)
            raw=jb(receipt);require(len(raw)<=12288 and sizes(out)+len(raw)<=OWNER_CAP,'SMALL_OWNER_RESERVATION')
            require(sizes(prep_out)<=PREPARATION_CAP and sizes(prep_out)+sizes(out)+len(raw)<=TOTAL,'FINAL_COMBINED16MIB')
            owner_write(out,'CLOSURE.json',raw,raw=True)
        except BaseException:
            receipt['status']='FAIL';receipt['closure_publication_failed']=True
    return receipt
def main():
    if len(sys.argv)!=5 or sys.argv[1]!='--owner-source-sha256' or sys.argv[3]!='--approved-preparation-sha256':
        print(json.dumps({'status':'DISABLED_NO_ROOT_PREPARATION_RELEASE','tokenizer_calls':0,'model_calls':0}));return 2
    try:
        require(COMBINED_STORAGE_ADMISSION_READY,'BLOCKED_UNPROVEN_COMBINED_STORAGE_CAP')
        owner_sha=verify(sys.argv[2]);raw=(PREP/'root_release/PREPARATION_RELEASE.json').read_bytes()
        require(sha(raw)==sys.argv[4],'EXPLICIT_PREPARATION_RELEASE_HASH');release=json.loads(raw)
        require(release['approved'] is True and release['scope']=='ONE_OFFLINE_FINAL_PREPARATION','ROOT_PREPARATION_AUTHORITY')
        text_raw=(PREP/'root_release/TEXT_LOCK.json').read_bytes();require(sha(text_raw)==release['text_lock_sha256'],'OWNER_EXACT_TEXT_LOCK')
        text_lock=json.loads(text_raw)
        require(text_lock['preparation_owner_binding']=={'namespace':'development/native_gate_coverage_increment_train_capture_v1','source_freeze_sha256':owner_sha},'OWNER_FIXED_BEFORE_TOKENIZATION')
        config=json.loads((HERE/'PREPARATION_OWNED_IDENTITY.json').read_bytes())
        command=[config['launch_image'],'-B',str(PREP/'prepare_offline.py'),'--approved-preparation-sha256',sys.argv[4]]
        value=run(command,config,HERE/'ownership_attempt_001',PREP/'preparation_attempt_001',
            {'owner_source_sha256':owner_sha,'preparation_release_sha256':sys.argv[4],'text_lock_sha256':release['text_lock_sha256']})
        print(json.dumps({k:value.get(k) for k in ('status','quiescent','exit_code','timed_out','elapsed_seconds','closure_publication_failed')}))
        return 0 if value['status']=='PASS' else 1
    except BaseException as error:
        print(json.dumps({'status':'OWNER_ADMISSION_FAILURE','error_type':type(error).__name__,'tokenizer_calls':0,'model_calls':0}));return 2
if __name__=='__main__':raise SystemExit(main())
