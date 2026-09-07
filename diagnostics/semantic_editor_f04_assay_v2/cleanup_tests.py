"""Twelve fixed pure cleanup regressions, no model, tokenizer, scores or process launch."""
import ctypes
from core import require
from native import NativeHandle,NativeFailure
from owned import OwnedPair

class Kernel:
    def __init__(self,error=None,wait=0,query=True,live=False):
        self.error,self.wait_result,self.query_ok,self.live=error,wait,query,live
        self.calls=[];self.terminations=0;self.query_count=0;self.wait_clobber=997
    def TerminateProcess(self,handle,code):
        self.calls.append(('terminate',handle));self.terminations+=1
        if self.error is not None:ctypes.set_last_error(self.error);return 0
        self.live=False;return 1
    def WaitForSingleObject(self,handle,timeout):
        self.calls.append(('wait',handle));ctypes.set_last_error(self.wait_clobber)
        return 258 if self.live else self.wait_result
    def GetExitCodeProcess(self,handle,code):
        self.calls.append(('query',handle));self.query_count+=1;ctypes.set_last_error(31)
        if self.query_ok:code._obj.value=125
        return self.query_ok

def handle(kernel,tag='retained',value=321):
    h=NativeHandle.__new__(NativeHandle)
    h.handle,h.tag,h.termination_rights,h.owned_close=value,tag,True,False
    h.closed=False;h.termination_receipt=None;h.kernel=kernel
    return h

def failed(h):
    try:h.terminate()
    except NativeFailure as error:
        require(error.receipt['classification']=='cleanup_failure','failure must retain raw receipt')
        return error.receipt
    raise ValueError('cleanup fault silently accepted')

def run():
    rows=[]
    def record(name,data):rows.append({'fixture':name,'status':'PASS','evidence':data})
    # Use the actual facade methods repeatedly, not a parallel stop implementation.
    from owned_capture import OwnedProcess
    events=[];wk,lk=Kernel(live=True),Kernel(live=True)
    pair=OwnedPair(handle(lk,'launcher',322),handle(wk,'actual',321),lambda kind,row:events.append({'kind':kind,'data':row}))
    pair.authenticated=True;facade=OwnedProcess.__new__(OwnedProcess);facade.pair=pair
    pair.observe(0.,now=lambda:1.)
    facade.terminate();facade.kill();pair.observe(0.,now=lambda:2.);pair.stop_and_wait('later_different_cause')
    require(pair.stop_reason=='deadline' and len(pair.faults)==1 and wk.terminations==lk.terminations==1 and not pair.cleanup_faults,'sticky first cause and one attempt per handle')
    record('repeat_stop_poll_terminate_kill_single_cause_attempt',{'cause':pair.stop_reason,'faults':pair.faults,'worker_calls':wk.terminations,'launcher_calls':lk.terminations})

    k=Kernel(error=5);h=handle(k);r=h.terminate()
    require(r['classification']=='observed_already_exited_race' and r['winerror']==5 and r['exit_proof']['exit_code']==125,'only proved error5 race accepted')
    require(all(call[1]==h.handle for call in k.calls),'same retained handle for terminate/wait/query')
    record('error5_signaled_same_handle_exit_query_race',r)

    r=failed(handle(Kernel(error=5,live=True)));require(r['exit_proof']['wait_result']==258,'live denial')
    record('error5_live_denial_fails',r)

    r=failed(handle(Kernel(error=6,wait=0xffffffff)));require(r['winerror']==6 and not r['exit_proof']['query_success'],'invalid OS handle fails')
    record('invalid_retained_os_handle_fails',r)

    r=failed(handle(Kernel(error=87)));require(r['exit_proof']['signaled'] and r['exit_proof']['query_success'],'unknown error cannot be excused by exit')
    record('unknown_error_even_with_exit_proof_fails',r)

    k=Kernel(error=5,wait=0xffffffff);r=failed(handle(k));require(k.query_count==0 and r['exit_proof']['wait_error']==997,'failed wait retained')
    record('error5_failed_wait_fails',r)

    r=failed(handle(Kernel(error=5,query=False)));require(r['exit_proof']['signaled'] and r['exit_proof']['query_error']==31,'failed exit query retained')
    record('error5_failed_query_fails',r)

    r=failed(handle(Kernel(error=5,wait=None)));require(not r['exit_proof']['signaled'],'missing successful wait proof')
    record('error5_missing_exit_proof_fails',r)

    k=Kernel(error=5);r=handle(k).terminate()
    require(r['winerror']==5 and ctypes.get_last_error()==31 and k.calls==[('terminate',321),('wait',321),('query',321)],'capture GetLastError immediately before later calls clobber it')
    record('last_error_preserved_across_clobbering_wait_query',r)

    wk,lk=Kernel(error=5,live=True),Kernel(live=True);events=[]
    pair=OwnedPair(handle(lk,'launcher',322),handle(wk,'actual',321),lambda k,r:events.append((k,r)));pair.authenticated=True
    pair.stop_and_wait('deadline',0.);original=list(pair.cleanup_faults)
    require(original and wk.terminations==1,'real initial fault')
    wk.live=False;pair.stop_and_wait('late_cleanup',0.);pair.observe(0.,now=lambda:5.)
    require(pair.cleanup_faults==original and pair.stop_reason=='deadline' and wk.terminations==1 and pair.quiescent(True,True),'later exit cannot erase denial or cause a retry')
    record('later_exit_retains_genuine_failure',{'faults':original,'termination_attempts':pair.termination_attempts,'event_count':len(events)})

    k=Kernel(error=5,live=True);h=handle(k);first=failed(h);k.live=False;second=failed(h)
    require(first is second and k.terminations==1 and first['exit_proof']['wait_result']==258,'native direct repeated termination cannot relabel an earlier failure')
    record('native_duplicate_attempt_is_sticky',first)

    k=Kernel();h=handle(k,value=0);r=failed(h)
    require(k.terminations==0 and not r['attempted'],'absent handle never targeted')
    require(not pair.quiescent(False,True) and not pair.quiescent(True,False),'I/O and writer completion required even after both exits')
    record('absent_handle_and_unjoined_io_never_accepted',r)
    require(len(rows)==12,'exact prospective pure case count')
    return rows
