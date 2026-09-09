"""Small suspended-launch containment, plus the reviewed retained-handle identity reader."""
import ctypes,importlib.util,json,os,sys
from ctypes import wintypes as W
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def require(ok,code):
    if not ok:raise ValueError(code)
def native():
    # The caller verifies SOURCE_FREEZE including these external source bytes first.
    path=ROOT/'diagnostics/semantic_editor_f03_v2_first_C_v2/native.py'
    spec=importlib.util.spec_from_file_location('preparation_owned_native',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class Basic(ctypes.Structure):
    _fields_=[('PerProcessUserTimeLimit',ctypes.c_longlong),('PerJobUserTimeLimit',ctypes.c_longlong),
        ('LimitFlags',W.DWORD),('MinimumWorkingSetSize',ctypes.c_size_t),('MaximumWorkingSetSize',ctypes.c_size_t),
        ('ActiveProcessLimit',W.DWORD),('Affinity',ctypes.c_size_t),('PriorityClass',W.DWORD),('SchedulingClass',W.DWORD)]
class IO(ctypes.Structure):
    _fields_=[(n,ctypes.c_ulonglong) for n in ('ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount')]
class Extended(ctypes.Structure):
    _fields_=[('BasicLimitInformation',Basic),('IoInfo',IO),('ProcessMemoryLimit',ctypes.c_size_t),
        ('JobMemoryLimit',ctypes.c_size_t),('PeakProcessMemoryUsed',ctypes.c_size_t),('PeakJobMemoryUsed',ctypes.c_size_t)]
class Pids(ctypes.Structure):
    _fields_=[('assigned',W.DWORD),('listed',W.DWORD),('ids',ctypes.c_size_t*8)]
class Job:
    def __init__(self):
        require(os.name=='nt','WINDOWS_ONLY');self.k=ctypes.WinDLL('kernel32',use_last_error=True);self.nt=ctypes.WinDLL('ntdll')
        self.k.CreateJobObjectW.argtypes=[ctypes.c_void_p,W.LPCWSTR];self.k.CreateJobObjectW.restype=W.HANDLE
        self.k.SetInformationJobObject.argtypes=[W.HANDLE,ctypes.c_int,ctypes.c_void_p,W.DWORD];self.k.SetInformationJobObject.restype=W.BOOL
        self.k.AssignProcessToJobObject.argtypes=[W.HANDLE,W.HANDLE];self.k.AssignProcessToJobObject.restype=W.BOOL
        self.k.QueryInformationJobObject.argtypes=[W.HANDLE,ctypes.c_int,ctypes.c_void_p,W.DWORD,ctypes.c_void_p];self.k.QueryInformationJobObject.restype=W.BOOL
        self.k.IsProcessInJob.argtypes=[W.HANDLE,W.HANDLE,ctypes.POINTER(W.BOOL)];self.k.IsProcessInJob.restype=W.BOOL
        self.k.TerminateJobObject.argtypes=[W.HANDLE,W.UINT];self.k.TerminateJobObject.restype=W.BOOL
        self.k.CloseHandle.argtypes=[W.HANDLE];self.k.CloseHandle.restype=W.BOOL
        self.nt.NtResumeProcess.argtypes=[W.HANDLE];self.nt.NtResumeProcess.restype=ctypes.c_long
        self.handle=self.k.CreateJobObjectW(None,None);require(self.handle,'CREATE_OWNED_JOB')
        self.terminated=False;self.closed=False
        limits=Extended();limits.BasicLimitInformation.LimitFlags=0x2000 # kill on last job handle close; no breakaway flags
        if not self.k.SetInformationJobObject(self.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            self.k.CloseHandle(self.handle);self.closed=True;raise ValueError('JOB_KILL_ON_CLOSE')
    def assign_resume(self,handle):
        require(self.k.AssignProcessToJobObject(self.handle,handle),'ASSIGN_SUSPENDED_BEFORE_EXECUTION')
        require(self.contains(handle),'LAUNCHER_IN_OWNED_JOB')
        require(self.nt.NtResumeProcess(handle)==0,'RESUME_OWNED_SUSPENDED_PROCESS')
    def contains(self,handle):
        yes=W.BOOL();require(self.k.IsProcessInJob(handle,self.handle,ctypes.byref(yes)),'JOB_MEMBERSHIP_QUERY');return bool(yes.value)
    def pids(self):
        values=Pids();require(self.k.QueryInformationJobObject(self.handle,3,ctypes.byref(values),ctypes.sizeof(values),None),'JOB_PROCESS_LIST')
        require(values.assigned==values.listed and values.listed<=8,'BOUNDED_JOB_PROCESS_LIST');return list(values.ids[:values.listed])
    def terminate(self):
        require(not self.terminated,'ONE_JOB_TERMINATION');self.terminated=True
        ok=self.k.TerminateJobObject(self.handle,125);error=ctypes.get_last_error() if not ok else None
        return {'attempted':True,'api_success':bool(ok),'winerror':error,'owned_job_not_pid_target':True}
    def close(self):
        if not self.closed:
            ok=self.k.CloseHandle(self.handle);require(ok,'CLOSE_OWNED_JOB');self.closed=True
