"""Six deterministic new stop-episode fixtures using unchanged V2 native methods."""
import ctypes
from core import require
from native import NativeHandle
from owned import OwnedPair

class Clock:
    def __init__(self):self.t=0.
    def __call__(self):return self.t

class Kernel:
    def __init__(self,clock,exit_at=None,error=None,query=True,overhead=0.):
        self.clock,self.exit_at,self.error,self.query,self.overhead=clock,exit_at,error,query,overhead
        self.live=True;self.terminations=0
    def TerminateProcess(self,handle,code):
        self.terminations+=1
        if self.error is not None:ctypes.set_last_error(self.error);return 0
        self.live=False;return 1
    def WaitForSingleObject(self,handle,milliseconds):
        duration=milliseconds/1000.
        if self.live and self.exit_at is not None and self.exit_at<=self.clock.t+duration:
            self.clock.t=max(self.clock.t,self.exit_at);self.live=False
        elif self.live:self.clock.t+=duration
        if not self.live and milliseconds>0 and self.overhead:
            self.clock.t+=self.overhead;self.overhead=0.
        return 258 if self.live else 0
    def GetExitCodeProcess(self,handle,code):
        if not self.query:ctypes.set_last_error(31);return 0
        code._obj.value=125;return 1

def handle(kernel,tag,value):
    h=NativeHandle.__new__(NativeHandle);h.kernel=kernel;h.handle=value;h.tag=tag
    h.termination_rights=True;h.owned_close=False;h.closed=False;h.termination_receipt=None
    return h

def fixture(clock,worker,launcher,mode='wrapper_child'):
    events=[];pair=OwnedPair(handle(launcher,'launcher',642),handle(worker,'actual',641),lambda k,r:events.append({'kind':k,'data':r}),clock=clock)
    pair.authenticated=True;pair.binding={'mode':mode}
    return pair,events

def result(pair,events,wk,lk):
    return {'stop_started_at':pair.stop_started_at,'absolute_deadline':pair.stop_deadline,'completed_at':pair.stop_completed_at,
            'overrun':pair.episode_overrun,'grace':pair.wrapper_grace,'primary_faults':pair.faults,'cleanup_faults':pair.cleanup_faults,
            'actual_terminations':sum(r['phase']=='actual_worker' for r in pair.termination_attempts),
            'launcher_terminations':sum(r['phase']=='launcher' for r in pair.termination_attempts),'events':events}

def run():
    rows=[]
    def record(name,pair,events,wk,lk):rows.append({'fixture':name,'status':'PASS','evidence':result(pair,events,wk,lk)})
    c=Clock();wk,lk=Kernel(c),Kernel(c,exit_at=.25);p,e=fixture(c,wk,lk);p.stop_and_wait('deadline')
    require(p.wrapper_grace['result']=='natural_wrapper_exit' and wk.terminations==1 and lk.terminations==0 and not p.cleanup_faults and p.quiescent(True,True),'natural wrapper exit avoids kill')
    record('natural_wrapper_exit_no_kill',p,e,wk,lk)

    c=Clock();wk,lk=Kernel(c),Kernel(c);p,e=fixture(c,wk,lk);p.stop_and_wait('deadline')
    require(p.wrapper_grace['result']=='grace_expired_still_live' and wk.terminations==lk.terminations==1 and not p.cleanup_faults and p.stop_completed_at<=3.,'expired grace gets one bounded forced completion')
    record('grace_expiry_one_forced_attempt',p,e,wk,lk)

    c=Clock();wk,lk=Kernel(c),Kernel(c,error=5);p,e=fixture(c,wk,lk);p.stop_and_wait('deadline')
    failures=list(p.cleanup_faults);attempt=p.termination_attempts[-1]
    require(failures and attempt['winerror']==5 and attempt['exit_proof']['wait_result']==258,'unchanged native live error5 fails')
    lk.live=False;p.observe_completion();p.stop_and_wait('later_call')
    require(p.cleanup_faults==failures and lk.terminations==1,'later proof cannot erase genuine fault or retry')
    record('strict_error5_timeout_failure_remains',p,e,wk,lk)

    c=Clock();wk,lk=Kernel(c),Kernel(c,exit_at=.2);p,e=fixture(c,wk,lk);p.stop_and_wait('deadline');deadline=p.stop_deadline
    c.t=8.;p.stop_and_wait('capture_cleanup');p.observe(0.,now=c);p.stop_and_wait('kill');p.observe_completion()
    require(p.stop_deadline==deadline==3. and len(p.faults)==1 and wk.terminations==1 and lk.terminations==0,'nonrenewable completed stop episode')
    require(sum(r['kind']=='wrapper_grace_started' for r in e)==sum(r['kind']=='wrapper_grace_result' for r in e)==1 and not p.episode_overrun,'grace once, not restarted by late observation')
    record('repeated_calls_deadline_and_grace_idempotent',p,e,wk,lk)

    c=Clock();shared=Kernel(c);p,e=fixture(c,shared,shared,mode='direct');p.stop_and_wait('deadline')
    require(shared.terminations==1 and p.wrapper_grace is None and len(p.termination_attempts)==1 and p.quiescent(True,True) and not p.cleanup_faults,'direct process has no duplicate phase')
    record('direct_process_no_grace_or_duplicate_termination',p,e,shared,shared)

    failures=[]
    for mode in ('time_exhausted','failed_worker_proof'):
        c=Clock();wk=Kernel(c,overhead=4. if mode=='time_exhausted' else 0.,query=mode!='failed_worker_proof');lk=Kernel(c)
        p,e=fixture(c,wk,lk);p.stop_and_wait('deadline')
        require(p.cleanup_faults and not p.quiescent(True,True),'overrun/failed proof cannot pass')
        if mode=='time_exhausted':require(p.episode_overrun and lk.terminations==0 and p.stop_deadline==3.,'no forced attempt beyond absolute stop deadline')
        else:require(p.wrapper_grace is None,'no grace without actual-worker exit proof')
        failures.append({'mode':mode,**result(p,e,wk,lk)})
    from owned_capture import cleanup_envelope
    capture={'status':'complete_valid','technical_recording_fault':None};envelope=cleanup_envelope(capture,0.,15.001)
    require(capture['status']=='INCONCLUSIVE' and not envelope['within_15_seconds'],'total cleanup overrun fails closed')
    rows.append({'fixture':'exhausted_time_or_failed_proof_never_passes','status':'PASS','evidence':failures,'cleanup_overrun':envelope})
    require(len(rows)==6,'exact six new pure fixtures')
    return rows
