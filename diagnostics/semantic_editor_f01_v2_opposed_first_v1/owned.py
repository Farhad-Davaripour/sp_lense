"""Pure authorization/lifecycle core. PID lookups never occur during cleanup."""
import json,os,time

def require(ok,message):
    if not ok:raise ValueError(message)
def canonical(path):return os.path.normcase(os.path.abspath(path))
def strict_json(raw):
    def pairs(items):
        out={}
        for key,value in items:require(key not in out,"duplicate handshake key");out[key]=value
        return out
    return json.loads(raw,object_pairs_hook=pairs)

def validate(claims,launcher,actual,config,nonce,driver_pid,termination_rights):
    require(len(claims)==1,"exactly one claim before acknowledgement")
    claim=claims[0];require(claim.get("kind")=="claim" and claim.get("nonce")==nonce,"private nonce claim")
    require(termination_rights is True,"termination-capable actual handle required before acknowledgement")
    require(launcher["live"] and actual["live"],"both identities live")
    require(launcher["ppid"]==driver_pid,"owned launcher parent")
    require(all(type(x) is int and x>0 for x in (launcher["pid"],actual["pid"],actual["creation_filetime"])),"finite process identity")
    require(claim["pid"]==actual["pid"] and claim["ppid"]==actual["ppid"] and claim["os_identity"]==actual,"OS/self identity and creation match")
    require(canonical(launcher["image"])==canonical(config["launch_image"]) and launcher["image_sha256"]==config["launch_sha256"],"owned launch image/hash")
    require(canonical(actual["image"])==canonical(config["base_image"]) and actual["image_sha256"]==config["base_sha256"],"actual Python image/hash")
    require(canonical(claim["base_executable"])==canonical(config["base_image"]) and canonical(claim["sys_executable"])==canonical(config["launch_image"]),"self executable contract")
    if launcher["pid"]==actual["pid"]:
        require(launcher==actual and canonical(config["launch_image"])==canonical(config["base_image"]),"strict direct-process binding")
        mode="direct"
    else:
        require(actual["ppid"]==launcher["pid"] and launcher["creation_filetime"]<=actual["creation_filetime"],"direct launcher child/creation order")
        mode="wrapper_child"
    return {"mode":mode,"launcher":launcher,"actual_worker":actual,"termination_rights_confirmed":True,"nonce":nonce}

class OwnedPair:
    """Retains original handles. No PID-based fallback or unauthenticated kill."""
    def __init__(self,launcher,worker,event,clock=time.monotonic):
        self.launcher,self.worker,self.event=launcher,worker,event
        self.authenticated=False;self.acknowledged=False;self.faults=[];self.cleanup_faults=[];self.binding=None
        self.stop_reason=None;self.attempted_handles=set();self.termination_attempts=[];self.exit_proofs={}
        self.clock=clock;self.stop_started_at=None;self.stop_deadline=None;self.stop_completed_at=None
        self.wrapper_grace=None;self.episode_overrun=False
    def fault(self,kind,detail):
        row={"kind":kind,"detail":detail};self.faults.append(row);self.event("primary_fault",row)
    def authorize(self,claims,config,nonce,driver_pid,acknowledge):
        require(not self.authenticated and not self.acknowledged,"one authorization per owned handle pair")
        try:self.binding=validate(claims,self.launcher.snapshot(),self.worker.snapshot(),config,nonce,driver_pid,self.worker.termination_rights)
        except BaseException as error:self.fault("identity_rejected",str(error));raise
        self.authenticated=True
        self.event("authenticated_before_permission",self.binding)
        acknowledge({"may_load":True,"nonce":nonce,"scope":"FAKE_WORK_ONLY"})
        self.acknowledged=True;self.event("permission_sent",{"fake_only":True})
    def cleanup_fault(self,phase,error):
        row={'phase':phase,'error':str(error),'receipt':getattr(error,'receipt',None)}
        if row not in self.cleanup_faults:
            self.cleanup_faults.append(row);self.event('cleanup_fault',row)
    def observe_completion(self,timeout=0.,require_exit=False):
        """Bounded observation only: never re-enters termination or clears any prior fault."""
        for phase,handle,allowed in (('actual_worker',self.worker,self.authenticated),('launcher',self.launcher,True)):
            if not allowed:continue
            try:
                proof=handle.exit_proof(timeout)
                valid=proof['valid_retained_handle'] and proof['signaled'] and proof['query_success']
                if valid:
                    if phase not in self.exit_proofs:
                        self.exit_proofs[phase]=proof;self.event('retained_exit_proof',{'phase':phase,**proof})
                elif require_exit or not proof['valid_retained_handle'] or proof['wait_result'] not in (0,258) or proof['signaled']:
                    from native import NativeFailure
                    raise NativeFailure('owned process exit not proven',proof)
            except BaseException as error:self.cleanup_fault(phase,error)
        complete=self.authenticated and all(p in self.exit_proofs for p in ('actual_worker','launcher'))
        if self.stop_deadline is not None:
            now=self.clock()
            if complete and self.stop_completed_at is None:self.stop_completed_at=now
            if (self.stop_completed_at if complete else now)>self.stop_deadline:
                self.episode_overrun=True
                self.cleanup_fault('stop_episode',ValueError('absolute three-second stop deadline exceeded'))
        return complete
    def quiescent(self,threads_joined,pipes_closed):
        # finish() records handle observations before joining the evidence-writer chain.
        return bool(self.authenticated and len(self.exit_proofs)==2 and threads_joined and pipes_closed)
    def _remaining(self,reserve=0.):
        return max(0.,min(1.,self.stop_deadline-self.clock()-reserve))
    def _stop_phase(self,phase,handle,event):
        try:
            if not handle.exited():
                require(self._remaining()>0.,'stop episode exhausted before termination')
                key=id(handle);require(key not in self.attempted_handles,'retained handle already attempted')
                self.attempted_handles.add(key);self.event(event,{'handle_tag':handle.tag})
                try:result=handle.terminate()
                except BaseException as error:
                    result=getattr(error,'receipt',{'classification':'cleanup_failure','error':str(error)})
                    self.termination_attempts.append({'phase':phase,**result})
                    self.event('retained_termination_result',{'phase':phase,**result});raise
                else:
                    self.termination_attempts.append({'phase':phase,**result})
                    self.event('retained_termination_result',{'phase':phase,**result})
            proof=handle.exit_proof(self._remaining())
            if not (proof['valid_retained_handle'] and proof['signaled'] and proof['query_success']):
                from native import NativeFailure
                raise NativeFailure('owned process completion not proven',proof)
            self.exit_proofs[phase]=proof;self.event('retained_exit_proof',{'phase':phase,**proof})
        except BaseException as error:self.cleanup_fault(phase,error)
    def _wrapper_grace(self):
        # Only after a successful actual-worker exit proof, never for a direct process.
        require(self.authenticated and self.binding['mode']=='wrapper_child' and 'actual_worker' in self.exit_proofs,'wrapper grace requires authenticated actual exit proof')
        require(self.wrapper_grace is None,'one nonrenewable wrapper grace')
        allowance=self._remaining(reserve=1.)
        self.wrapper_grace={'started_at':self.clock(),'allowance_seconds':allowance,'episode_deadline':self.stop_deadline,
                            'forced_verification_reserved_seconds':1.,'result':None}
        self.event('wrapper_grace_started',dict(self.wrapper_grace))
        try:
            proof=self.launcher.exit_proof(allowance)
            self.wrapper_grace['proof']=proof
            if proof['valid_retained_handle'] and proof['signaled'] and proof['query_success']:
                self.exit_proofs['launcher']=proof
                self.event('retained_exit_proof',{'phase':'launcher',**proof})
                self.wrapper_grace['result']='natural_wrapper_exit'
            elif proof['valid_retained_handle'] and proof['wait_result']==258:
                self.wrapper_grace['result']='grace_expired_still_live'
            else:
                from native import NativeFailure
                raise NativeFailure('wrapper grace exit proof failed',proof)
        except BaseException as error:
            self.wrapper_grace['result']='proof_failure';self.wrapper_grace['error']=str(error)
            self.cleanup_fault('wrapper_grace',error)
        finally:
            self.wrapper_grace['finished_at']=self.clock()
            self.event('wrapper_grace_result',dict(self.wrapper_grace))
    def stop_and_wait(self,reason,timeout=2.):
        if self.stop_reason is not None:return self.observe_completion(timeout=0.)
        self.stop_reason=reason;self.stop_started_at=self.clock();self.stop_deadline=self.stop_started_at+3.
        self.fault(reason,'first stop cause; fixed three-second episode, repeated calls only observe')
        if self.authenticated:self._stop_phase('actual_worker',self.worker,'terminate_actual_retained_handle')
        else:self.event('actual_worker_not_targeted',{'reason':'not authenticated'})
        direct=self.authenticated and self.binding is not None and self.binding['mode']=='direct'
        wrapper=self.authenticated and self.binding is not None and self.binding['mode']=='wrapper_child'
        if wrapper and 'actual_worker' in self.exit_proofs:self._wrapper_grace()
        if not direct and 'launcher' not in self.exit_proofs:
            self._stop_phase('launcher',self.launcher,'terminate_owned_launcher')
        # In direct mode the actual-worker attempt is the only termination phase.
        # Verify the separately retained launcher handle without another grace/kill.
        return self.observe_completion(timeout=0.,require_exit=direct)
    def observe(self,deadline,now=time.monotonic):
        if not self.authenticated:return 'NOT_AUTHENTICATED'
        if self.stop_reason is not None:
            return 'STOPPED' if self.observe_completion() else 'UNJOINED'
        try:launcher_dead=self.launcher.exited();worker_dead=self.worker.exited()
        except BaseException as error:
            self.cleanup_fault('poll',error)
            self.stop_and_wait('observation_failure',timeout=1.)
            return 'STOPPED' if len(self.exit_proofs)==2 else 'UNJOINED'
        expired=now()>=deadline
        if (launcher_dead and not worker_dead) or expired:
            self.stop_and_wait('deadline' if expired else 'launcher_exit',timeout=1.)
            return 'STOPPED' if len(self.exit_proofs)==2 else 'UNJOINED'
        if launcher_dead and worker_dead:
            return 'BOTH_EXITED' if self.observe_completion() else 'UNJOINED'
        return 'RUNNING'
