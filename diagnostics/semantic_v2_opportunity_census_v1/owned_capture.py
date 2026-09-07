"""Minimal owned-handle facade at the existing bounded run_capture boundary."""
import json,os,secrets,subprocess,sys,threading,time
from pathlib import Path
from core import HERE,ROOT,Budget,read,require
from owned import OwnedPair,strict_json
from native import NativeHandle,open_claimed_worker
sys.path.insert(0,str(ROOT))
from scripts.three_family_bounded_capture import run_capture

def cleanup_envelope(capture,origin,now):
    """Account scheduler/recording overhead, not just requested wait durations."""
    row={'origin_monotonic':origin,'absolute_deadline':origin+15.,'observed_at':now,
         'elapsed_seconds':max(0.,now-origin),'within_15_seconds':now<=origin+15.}
    if not row['within_15_seconds'] and capture is not None:
        capture['status']='INCONCLUSIVE'
        capture['technical_recording_fault']=capture.get('technical_recording_fault') or 'OWNED_CLEANUP_ENVELOPE_EXCEEDED'
    return row

def validate_claim(raw,nonce):
    claim=strict_json(raw)
    require(claim.get('kind')=='claim' and claim.get('nonce')==nonce and type(claim.get('pid')) is int and claim['pid']>0,'one private nonce claim before worker OpenProcess or permission')
    return claim

def execution_envelope(frame,lane,output,release_sha):
    require(frame.get('scope')=='FAKE_WORK_ONLY' and frame.get('may_load') is True,'unchanged core ownership attestation')
    require(lane in ('worker','audit','synthetic_worker','synthetic_audit','fake_deadline'),'fixed execution lane')
    real=lane in ('worker','audit')
    require((release_sha is not None)==real,'fake ownership is never scientific authorization')
    return {'ownership_only':frame,'lane':lane,'output':str(Path(output).resolve()),
            'execution_scope':'PRODUCTION_V2_CENSUS' if real else 'SYNTHETIC_ONLY','root_release_sha256':release_sha}

class OwnedProcess:
    def __init__(self,command,kwargs,budget,deadline,lane,output,release_sha):
        self.budget,self.deadline=budget,deadline;self.pair=None;self.actual=None;self.handshake_thread=None;self.handshake_error=None;self.events=[]
        self.event_threads=[];self.event_record_errors=[]
        require(command==[str(Path(os.sys.executable)), '-B',str(HERE/'entry.py'),lane,str(Path(output).resolve())],'fixed local bootstrap command')
        require(kwargs.get('stdin')==subprocess.DEVNULL and kwargs.get('stdout')==subprocess.PIPE and kwargs.get('stderr')==subprocess.STDOUT and not kwargs.get('shell',False),'fixed binary capture boundary')
        actual_kwargs={**kwargs,'stdin':subprocess.PIPE}
        self.proc=subprocess.Popen(command,**actual_kwargs);self.pid=self.proc.pid;self.stdout=self.proc.stdout
        self.launcher=NativeHandle(int(self.proc._handle),'owned_launcher_original',True)
        self.lane,self.output,self.release_sha=lane,output,release_sha

    def persist(self,name,value,append=False,before_permission=False):
        # A full/blocked disk must never prevent retained-handle termination.
        # Before acknowledgement persistence must finish; otherwise no permission.
        previous=self.event_threads[-1] if self.event_threads else None
        def save():
            if previous is not None:previous.join()  # daemon writer chain preserves exact journal order
            try:
                if append:self.budget.event(name,value)
                else:self.budget.write(name,value)
            except BaseException as error:
                self.event_record_errors.append(type(error).__name__+': '+str(error)[:512]);self.budget.fault('OWNED_EVIDENCE_FAULT')
        require(len(self.event_threads)<66,'bounded ownership writer count')
        thread=threading.Thread(target=save,name='owned-bounded-evidence',daemon=True);self.event_threads.append(thread);thread.start()
        if before_permission:
            thread.join(1.);require(not thread.is_alive() and not self.event_record_errors,'complete identity evidence before permission')

    def event(self,kind,data):
        before=kind=='authenticated_before_permission'
        try:
            require(len(self.events)<64,'bounded ownership events')
            row={'kind':kind,'data':data,'monotonic':time.monotonic()};self.events.append(row)
            self.persist('ownership_events.jsonl',row,append=True,before_permission=before)
        except BaseException as error:
            self.event_record_errors.append(type(error).__name__+': '+str(error)[:512]);self.budget.fault('OWNED_EVIDENCE_FAULT')
            if before:raise

    def authenticate(self):
        nonce=secrets.token_hex(32);frame=[];done=threading.Event()
        def receive():
            try:
                raw=self.stdout.readline(8193);require(0<len(raw)<=8192 and raw.endswith(b'\n'),'missing/oversized handshake')
                frame.append(validate_claim(raw,nonce))
            except BaseException as exc:self.handshake_error=type(exc).__name__+': '+str(exc)
            finally:done.set()
        self.handshake_thread=threading.Thread(target=receive,name='owned-private-handshake',daemon=True)
        self.handshake_thread.start()
        try:
            self.proc.stdin.write(json.dumps({'nonce':nonce}).encode()+b'\n');self.proc.stdin.flush()
            require(done.wait(min(3.,max(0.,self.deadline-time.monotonic()))),'private handshake deadline')
            self.handshake_thread.join(.1)
            require(not self.handshake_thread.is_alive() and self.handshake_error is None and len(frame)==1,'one complete private handshake')
            self.actual=open_claimed_worker(frame[0]['pid'])
            self.pair=OwnedPair(self.launcher,self.actual,self.event)
            def acknowledge(value):
                self.persist('process_identity.json',{'binding':self.pair.binding,'launcher_pid':self.proc.pid,
                    'actual_worker_pid':frame[0]['pid'],'handle_retained_before_ack':True,'lane':self.lane,
                    'ownership_scope_is_not_execution_authority':True,'no_later_pid_targeting':True},before_permission=True)
                envelope=execution_envelope(value,self.lane,self.output,self.release_sha)
                self.proc.stdin.write(json.dumps(envelope).encode()+b'\n');self.proc.stdin.flush()
            self.pair.authorize(frame,read(HERE/'source_bindings.json')['owned_identity'],nonce,os.getpid(),acknowledge)
            self.proc.stdin.close()
        except BaseException:
            # Close the private pipe first: an unacknowledged bootstrap cannot load.
            if not self.proc.stdin.closed:self.proc.stdin.close()
            if self.pair is not None:self.pair.stop_and_wait('handshake_rejected',timeout=1.)
            else:
                if not self.launcher.exited():self.launcher.terminate()
                self.launcher.wait(1.)
            if self.handshake_thread is not None:self.handshake_thread.join(1.)
            raise
        return self

    def poll(self):
        require(self.pair is not None and self.pair.authenticated,'no unauthenticated facade polling')
        state=self.pair.observe(self.deadline)
        if self.pair.faults:self.budget.fault('OWNED_WORKER_LIFECYCLE_FAULT')
        if state in ('STOPPED','BOTH_EXITED'):
            code=self.actual.exit_code();launch=self.launcher.exit_code()
            return code if code else launch
        return None

    def wait(self,timeout=None):
        until=time.monotonic()+(0 if timeout is None else timeout)
        while True:
            value=self.poll()
            if value is not None:self.proc.wait(timeout=0);return value
            if time.monotonic()>=until:raise subprocess.TimeoutExpired('owned-pair',timeout)
            time.sleep(min(.01,max(0.,until-time.monotonic())))

    def terminate(self):
        require(self.pair is not None,'only authenticated retained worker target')
        self.pair.stop_and_wait('capture_cleanup',timeout=1.)
    def kill(self):self.terminate()  # Same retained handles; no PID/tree lookup.

    def finish(self,capture):
        # Record all exit/handle-close observations BEFORE joining the writer chain.
        if self.pair is not None:self.pair.observe_completion(require_exit=True)
        if self.actual is not None:
            try:self.actual.close()
            except BaseException as error:
                if self.pair is not None:self.pair.cleanup_fault('actual_handle_close',error)
        joined=self.handshake_thread is None or not self.handshake_thread.is_alive()
        origin=min(self.deadline,self.pair.stop_started_at if self.pair is not None and self.pair.stop_started_at is not None else self.deadline)
        journal_deadline=min(time.monotonic()+1.,origin+15.)
        for thread in self.event_threads:thread.join(max(0.,journal_deadline-time.monotonic()))
        evidence_joined=all(not thread.is_alive() for thread in self.event_threads)
        pipes=self.proc.stdin.closed and self.stdout.closed
        proofs={} if self.pair is None else self.pair.exit_proofs
        worker_exit=proofs.get('actual_worker',{}).get('exit_code')
        launcher_exit=proofs.get('launcher',{}).get('exit_code')
        quiet=False if self.pair is None else self.pair.quiescent(joined and evidence_joined,pipes)
        result={'binding_authenticated':self.pair is not None and self.pair.authenticated,
            'actual_worker_exit_code':worker_exit,'launcher_exit_code':launcher_exit,'handshake_thread_joined':joined,
            'pipes_closed':pipes,'quiescent':quiet,'handshake_error':self.handshake_error,
            'evidence_threads_joined':evidence_joined,'evidence_errors':self.event_record_errors,
            'faults':[] if self.pair is None else self.pair.faults,'cleanup_faults':[] if self.pair is None else self.pair.cleanup_faults,
            'stop_reason':None if self.pair is None else self.pair.stop_reason,
            'stop_episode':None if self.pair is None else {'started_at':self.pair.stop_started_at,'deadline':self.pair.stop_deadline,'completed_at':self.pair.stop_completed_at,'overrun':self.pair.episode_overrun},
            'wrapper_grace':None if self.pair is None else self.pair.wrapper_grace,
            'termination_attempts':[] if self.pair is None else self.pair.termination_attempts,
            'exit_proofs':proofs,'actual_handle_closed':self.actual is not None and self.actual.closed,
            'retained_actual_termination_events':sum(x['kind']=='terminate_actual_retained_handle' for x in self.events)}
        self.budget.write('ownership_final.json',result)
        capture['owned_worker']=result;capture['quiescent']=bool(capture['quiescent'] and quiet)
        if not quiet or self.handshake_error or self.event_record_errors or result['faults'] or result['cleanup_faults'] or worker_exit!=0 or launcher_exit!=0:
            capture['status']='INCONCLUSIVE';capture['technical_recording_fault']=capture['technical_recording_fault'] or 'OWNED_WORKER_IDENTITY_OR_LIFECYCLE'
        capture['cleanup_envelope']=cleanup_envelope(capture,origin,time.monotonic())
        return capture

def supervise(lane,output,seconds,release_sha=None):
    output=Path(output).resolve();budget=Budget(output);started=time.monotonic();deadline=started+seconds;owned={}
    require(lane in ('worker','audit','synthetic_worker','synthetic_audit','fake_deadline'),'fixed lane')
    command=[os.sys.executable,'-B',str(HERE/'entry.py'),lane,str(output)]
    budget.write('RUN_STARTED.json',{'command':command,'lane':lane,'started_monotonic':started,'deadline_monotonic':deadline,
        'scope':'PRODUCTION_V2_CENSUS' if release_sha is not None else 'SYNTHETIC_ONLY','root_release_sha256':release_sha})
    def spawn(actual,**kwargs):
        require(actual==command,'exact owned command')
        proc=OwnedProcess(actual,kwargs,budget,deadline,lane,output,release_sha);owned['process']=proc
        return proc.authenticate()
    capture=None
    try:
        capture=run_capture(command,budget,deadline,cwd=ROOT,process_factory=spawn,terminate_timeout=3,kill_timeout=3,reader_join_timeout=1)
        if 'process' in owned:capture=owned['process'].finish(capture)
        budget.write('capture.json',capture);return capture
    finally:
        proc=owned.get('process');pair=None if proc is None else proc.pair
        origin=min(deadline,pair.stop_started_at if pair is not None and pair.stop_started_at is not None else deadline)
        envelope=cleanup_envelope(capture,origin,time.monotonic())
        budget.write('supervisor_final.json',{'elapsed_seconds_including_cleanup':time.monotonic()-started,
            'capture_returned':capture is not None,'quiescent':False if capture is None else capture['quiescent'],
            'cleanup_envelope':envelope,'authoritative_return_status':None if capture is None else capture['status'],
            'owned_actual_handle_pre_ack':True,'both_processes_required':True,'no_pid_tree_kill':True})
