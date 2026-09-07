"""Narrow Windows handle operations, reusing the authenticated probe identity reader."""
import ctypes,hashlib,importlib.util,sys
from ctypes import wintypes
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PROBE=ROOT/"diagnostics/windows_worker_identity_probe_v1/probe.py"
PROBE_SHA="2eebb8671603837bfde1c7381406379220af941d0a57f5affb68011de8728e56"
if hashlib.sha256(PROBE.read_bytes()).hexdigest()!=PROBE_SHA:raise ValueError("authenticated identity-reader source changed")
spec=importlib.util.spec_from_file_location("verified_worker_identity_reader",PROBE);source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)

class NativeHandle:
    def __init__(self,handle,tag,termination_rights,owned_close=False):
        self.handle,self.tag,self.termination_rights,self.owned_close=handle,tag,termination_rights,owned_close
        self.closed=False;self.termination_receipt=None
        self.kernel,_=source.api()
        self.kernel.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT];self.kernel.TerminateProcess.restype=wintypes.BOOL
        self.kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)];self.kernel.GetExitCodeProcess.restype=wintypes.BOOL
    def snapshot(self):return source.identity(self.handle)
    def _valid(self):return bool(self.handle) and not self.closed
    def exit_proof(self,timeout=0.):
        """Only this original handle can prove exit; wait and query failures stay visible."""
        row={'handle':int(self.handle or 0),'handle_tag':self.tag,'valid_retained_handle':self._valid(),
             'wait_result':None,'wait_error':None,'signaled':False,'query_success':False,'query_error':None,'exit_code':None}
        if not row['valid_retained_handle']:return row
        wait=self.kernel.WaitForSingleObject(self.handle,int(max(0.,timeout)*1000))
        error=ctypes.get_last_error() if wait==0xffffffff else None
        row.update(wait_result=wait,wait_error=error,signaled=wait==0)
        if wait==0:
            code=wintypes.DWORD();success=self.kernel.GetExitCodeProcess(self.handle,ctypes.byref(code))
            error=ctypes.get_last_error() if not success else None
            row.update(query_success=bool(success),query_error=error,exit_code=int(code.value) if success else None)
        return row
    def exited(self):
        proof=self.exit_proof()
        if not proof['valid_retained_handle'] or proof['wait_result'] not in (0,258) or (proof['signaled'] and not proof['query_success']):
            raise NativeFailure('retained handle exit observation failed',proof)
        return proof['signaled']
    def wait(self,timeout):
        proof=self.exit_proof(timeout)
        return proof['valid_retained_handle'] and proof['signaled'] and proof['query_success']
    def terminate(self):
        # Sticky per-handle receipt is set BEFORE the call. Even direct callers cannot retry.
        if self.termination_receipt is not None:
            if self.termination_receipt['classification'] in ('termination_requested','observed_already_exited_race'):return self.termination_receipt
            raise NativeFailure('retained termination attempt previously failed',self.termination_receipt)
        row={'handle':int(self.handle or 0),'handle_tag':self.tag,'attempted':False,'api_success':False,
             'winerror':None,'classification':'cleanup_failure','exit_proof':None}
        self.termination_receipt=row
        if not self._valid() or not self.termination_rights:raise NativeFailure('invalid handle or missing termination rights',row)
        row['attempted']=True
        success=self.kernel.TerminateProcess(self.handle,125)
        error=ctypes.get_last_error() if not success else None  # IMMEDIATE: no intervening API/formatting call.
        row.update(api_success=bool(success),winerror=error)
        if success:row['classification']='termination_requested';return row
        proof=self.exit_proof(0.);row['exit_proof']=proof
        if error==5 and proof['valid_retained_handle'] and proof['signaled'] and proof['query_success']:
            row['classification']='observed_already_exited_race';return row
        raise NativeFailure('TerminateProcess retained handle failed',row)
    def exit_code(self):
        code=wintypes.DWORD();source.require(self.kernel.GetExitCodeProcess(self.handle,ctypes.byref(code)),"exit code");return code.value
    def close(self):
        if self.owned_close:
            success=self.kernel.CloseHandle(self.handle)
            error=ctypes.get_last_error() if not success else None
            if not success:raise NativeFailure('retained handle close failed',{'handle_tag':self.tag,'winerror':error})
            self.owned_close=False;self.closed=True

class NativeFailure(RuntimeError):
    def __init__(self,message,receipt):
        self.receipt=receipt
        super().__init__(message+': '+str(receipt))

def open_claimed_worker(pid):
    kernel,_=source.api()
    # Required access succeeds atomically; no query-only fallback or later reopen.
    handle=kernel.OpenProcess(0x1|0x400|0x1000|0x100000,False,pid)
    source.require(handle,"actual worker termination/query/synchronize rights unavailable")
    return NativeHandle(handle,"actual_worker_retained",True,True)
