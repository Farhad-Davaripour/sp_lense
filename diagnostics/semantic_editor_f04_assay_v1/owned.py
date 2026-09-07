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
    def quiescent(self,threads_joined,pipes_closed):
        return bool(self.authenticated and self.worker.exited() and self.launcher.exited() and threads_joined and pipes_closed)
    def stop_and_wait(self,reason,timeout=2.):
        self.fault(reason,"stop authenticated actual worker even when launcher has already exited")
        # Never use launcher.poll() as permission to skip the actual worker.
        if self.authenticated:
            try:
                if not self.worker.exited():
                    self.event("terminate_actual_retained_handle",{"handle_tag":self.worker.tag});self.worker.terminate()
                require(self.worker.wait(timeout),"actual worker did not exit")
                self.event("actual_worker_exit_observed",{"handle_tag":self.worker.tag})
            except BaseException as error:
                row={"phase":"actual_worker","error":str(error)};self.cleanup_faults.append(row);self.event("cleanup_fault",row)
        else:self.event("actual_worker_not_targeted",{"reason":"not authenticated"})
        try:
            if not self.launcher.exited():self.event("terminate_owned_launcher",{"handle_tag":self.launcher.tag});self.launcher.terminate()
            require(self.launcher.wait(timeout),"owned launcher did not exit")
        except BaseException as error:
            row={"phase":"launcher","error":str(error)};self.cleanup_faults.append(row);self.event("cleanup_fault",row)
    def observe(self,deadline,now=time.monotonic):
        """Future polling adapter: launcher-exit and deadline are both sticky faults."""
        if not self.authenticated:return "NOT_AUTHENTICATED"
        launcher_dead=self.launcher.exited();worker_dead=self.worker.exited();expired=now()>=deadline
        if (launcher_dead and not worker_dead) or expired:
            if launcher_dead and not worker_dead:self.fault("launcher_exit_while_worker_alive","EOF cannot establish quiescence")
            self.stop_and_wait("deadline" if expired else "launcher_exit",timeout=1.)
            return "STOPPED" if self.worker.exited() and self.launcher.exited() else "UNJOINED"
        return "BOTH_EXITED" if launcher_dead and worker_dead else "RUNNING"
