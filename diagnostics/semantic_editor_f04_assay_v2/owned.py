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
    def __init__(self,launcher,worker,event):
        self.launcher,self.worker,self.event=launcher,worker,event
        self.authenticated=False;self.acknowledged=False;self.faults=[];self.cleanup_faults=[];self.binding=None
        self.stop_reason=None;self.attempted_handles=set();self.termination_attempts=[];self.exit_proofs={}
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
        return self.authenticated and all(p in self.exit_proofs for p in ('actual_worker','launcher'))
    def quiescent(self,threads_joined,pipes_closed):
        # finish() records handle observations before joining the evidence-writer chain.
        return bool(self.authenticated and len(self.exit_proofs)==2 and threads_joined and pipes_closed)
    def stop_and_wait(self,reason,timeout=2.):
        if self.stop_reason is not None:
            return self.observe_completion(timeout=0.)
        self.stop_reason=reason
        self.fault(reason,'first stop cause; subsequent cleanup calls only observe retained handles')
        # Actual worker first, including when the owned launcher has already exited.
        for phase,handle,allowed,event in (
            ('actual_worker',self.worker,self.authenticated,'terminate_actual_retained_handle'),
            ('launcher',self.launcher,True,'terminate_owned_launcher')):
            if not allowed:
                self.event('actual_worker_not_targeted',{'reason':'not authenticated'});continue
            try:
                if not handle.exited():
                    key=id(handle)
                    require(key not in self.attempted_handles,'retained handle already attempted')
                    self.attempted_handles.add(key)
                    self.event(event,{'handle_tag':handle.tag})
                    try:result=handle.terminate()
                    except BaseException as error:
                        result=getattr(error,'receipt',{'classification':'cleanup_failure','error':str(error)})
                        self.termination_attempts.append({'phase':phase,**result})
                        self.event('retained_termination_result',{'phase':phase,**result})
                        raise
                    else:
                        self.termination_attempts.append({'phase':phase,**result})
                        self.event('retained_termination_result',{'phase':phase,**result})
                proof=handle.exit_proof(timeout)
                if not (proof['valid_retained_handle'] and proof['signaled'] and proof['query_success']):
                    from native import NativeFailure
                    raise NativeFailure('owned process completion not proven',proof)
                self.exit_proofs[phase]=proof
                self.event('retained_exit_proof',{'phase':phase,**proof})
            except BaseException as error:self.cleanup_fault(phase,error)
        return self.observe_completion(timeout=0.)
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
