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
        self.kernel,_=source.api()
        self.kernel.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT];self.kernel.TerminateProcess.restype=wintypes.BOOL
        self.kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)];self.kernel.GetExitCodeProcess.restype=wintypes.BOOL
    def snapshot(self):return source.identity(self.handle)
    def exited(self):return self.kernel.WaitForSingleObject(self.handle,0)==0
    def wait(self,timeout):return self.kernel.WaitForSingleObject(self.handle,int(timeout*1000))==0
    def terminate(self):
        source.require(self.termination_rights,"no termination rights")
        source.require(self.kernel.TerminateProcess(self.handle,125),"TerminateProcess retained handle failed")
    def exit_code(self):
        code=wintypes.DWORD();source.require(self.kernel.GetExitCodeProcess(self.handle,ctypes.byref(code)),"exit code");return code.value
    def close(self):
        if self.owned_close:self.kernel.CloseHandle(self.handle);self.owned_close=False

def open_claimed_worker(pid):
    kernel,_=source.api()
    # Required access succeeds atomically; no query-only fallback or later reopen.
    handle=kernel.OpenProcess(0x1|0x400|0x1000|0x100000,False,pid)
    source.require(handle,"actual worker termination/query/synchronize rights unavailable")
    return NativeHandle(handle,"actual_worker_retained",True,True)
