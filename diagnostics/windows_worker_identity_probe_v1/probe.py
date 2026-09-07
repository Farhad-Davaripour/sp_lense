"""One trivial owned child. Windows process identity only; no model/data imports."""
import ctypes,hashlib,json,math,os,queue,secrets,subprocess,sys,threading,time
from ctypes import wintypes
from pathlib import Path
HERE=Path(__file__).resolve().parent
MAX_FILE=1024**2
MAX_TOTAL=4*1024**2

def require(ok,why):
    if not ok:raise ValueError(why)
def sha(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def write(name,value):
    data=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    require(len(data)<=MAX_FILE and sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file())+len(data)<=MAX_TOTAL,"artifact cap")
    with (HERE/name).open("xb") as stream:stream.write(data)
def normpath(value):return os.path.normcase(os.path.abspath(value))

class BASIC(ctypes.Structure):
    _fields_=[("ExitStatus",ctypes.c_long),("PebBaseAddress",ctypes.c_void_p),("AffinityMask",ctypes.c_size_t),
        ("BasePriority",ctypes.c_long),("UniqueProcessId",ctypes.c_size_t),("InheritedFromUniqueProcessId",ctypes.c_size_t)]

def api():
    require(os.name=="nt","Windows only")
    kernel=ctypes.WinDLL("kernel32",use_last_error=True);nt=ctypes.WinDLL("ntdll")
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    kernel.GetProcessId.argtypes=[wintypes.HANDLE];kernel.GetProcessId.restype=wintypes.DWORD
    kernel.QueryFullProcessImageNameW.argtypes=[wintypes.HANDLE,wintypes.DWORD,wintypes.LPWSTR,ctypes.POINTER(wintypes.DWORD)]
    kernel.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.WaitForSingleObject.restype=wintypes.DWORD
    nt.NtQueryInformationProcess.argtypes=[wintypes.HANDLE,wintypes.ULONG,ctypes.c_void_p,wintypes.ULONG,ctypes.POINTER(wintypes.ULONG)]
    nt.NtQueryInformationProcess.restype=ctypes.c_long
    return kernel,nt

def identity(handle):
    kernel,nt=api();basic=BASIC();returned=wintypes.ULONG()
    status=nt.NtQueryInformationProcess(handle,0,ctypes.byref(basic),ctypes.sizeof(basic),ctypes.byref(returned))
    require(status==0 and returned.value==ctypes.sizeof(basic),"ProcessBasicInformation unavailable")
    size=wintypes.DWORD(32768);image=ctypes.create_unicode_buffer(size.value)
    require(kernel.QueryFullProcessImageNameW(handle,0,image,ctypes.byref(size)),"OS image unavailable")
    times=[wintypes.FILETIME() for _ in range(4)]
    require(kernel.GetProcessTimes(handle,*[ctypes.byref(t) for t in times]),"creation time unavailable")
    pid=kernel.GetProcessId(handle);require(pid==basic.UniqueProcessId and pid>0,"OS PID consistency")
    return {"pid":pid,"ppid":int(basic.InheritedFromUniqueProcessId),"image":image.value,"image_sha256":sha(Path(image.value).read_bytes()),
        "creation_filetime":(times[0].dwHighDateTime<<32)|times[0].dwLowDateTime,"live":kernel.WaitForSingleObject(handle,0)==258}

def child():
    watchdog=threading.Timer(7.,lambda:os._exit(124));watchdog.daemon=True;watchdog.start()
    try:
        challenge=sys.stdin.buffer.readline(4097);require(len(challenge)<=4096,"challenge cap")
        nonce=json.loads(challenge)["nonce"];require(isinstance(nonce,str) and len(nonce)==64,"nonce")
        kernel,_=api();me=identity(kernel.GetCurrentProcess())
        claim={"nonce":nonce,"pid":os.getpid(),"ppid":os.getppid(),"sys_executable":sys.executable,"base_executable":sys._base_executable,
            "os_identity":me,"model_imports":False,"role":"sole trivial probe child"}
        print(json.dumps(claim,sort_keys=True),flush=True)
        ack=sys.stdin.buffer.readline(4097);require(len(ack)<=4096 and json.loads(ack)=={"ack":nonce},"matching private-pipe acknowledgement")
    finally:watchdog.cancel()

def relationship(wrapper,actual,claim,plan,nonce,driver_pid):
    checks={"distinct_wrapper_child":wrapper["pid"]!=actual["pid"],"direct_child":actual["ppid"]==wrapper["pid"],"owned_wrapper_parent":wrapper["ppid"]==driver_pid,
        "nonce":claim["nonce"]==nonce,"self_pid_ppid":claim["pid"]==actual["pid"] and claim["ppid"]==actual["ppid"],
        "same_live_identity":claim["os_identity"]==actual,"live":wrapper["live"] and actual["live"],"creation_order":wrapper["creation_filetime"]<=actual["creation_filetime"],
        "wrapper_image":normpath(wrapper["image"])==normpath(plan["venv_executable"]) and wrapper["image_sha256"]==plan["external_sha256"][plan["venv_executable"]],
        "base_image":normpath(actual["image"])==normpath(plan["base_executable"]) and actual["image_sha256"]==plan["external_sha256"][plan["base_executable"]],
        "self_executable_contract":normpath(claim["sys_executable"])==normpath(plan["venv_executable"]) and normpath(claim["base_executable"])==normpath(plan["base_executable"])}
    return {"checks":checks,"all_pass":all(checks.values())}

def probe():
    plan=read(HERE/"plan.json");freeze=read(HERE/"freeze.json")
    for path,digest in freeze["files"].items():require(sha((HERE/path).read_bytes())==digest,"frozen source "+path)
    for path,digest in plan["external_sha256"].items():require(sha(Path(path).read_bytes())==digest,"pinned source "+path)
    require(normpath(sys.executable)==normpath(plan["venv_executable"]),"driver uses pinned same executable")
    require(not any(k in sys.modules for k in ("torch","transformers","transformer_lens")),"no model imports")
    usage=read(HERE/"batch_usage.json");require(usage["known_standard_used_percent"]<100,"usage exhausted")
    nonce=secrets.token_hex(32);started=time.monotonic();deadline=started+10;proc=None;child_handle=None;reader=None
    result={"status":"UNKNOWN","probe_children_started":0,"model_loads":0,"forwards":0,"derivatives":0,"tokenizer_calls":0,"gate_scores":0,
        "primary_error":None,"cleanup_errors":[],"stdout":"","stderr":"","nonce":nonce,"driver_pid":os.getpid(),"driver_executable":sys.executable}
    try:
        command=[sys.executable,"-B",str(HERE/"probe.py"),"child"]
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
        result.update(probe_children_started=1,command=command,owned_Popen_pid=proc.pid)
        proc.stdin.write(json.dumps({"nonce":nonce}).encode()+b"\n");proc.stdin.flush()
        received=queue.Queue(maxsize=1)
        reader=threading.Thread(target=lambda:received.put(proc.stdout.readline(16385)),daemon=True);reader.start()
        line=received.get(timeout=max(.01,deadline-time.monotonic()-2));reader.join(.1)
        require(len(line)<=16384 and line.endswith(b"\n"),"bounded complete child claim")
        result["stdout"]=line.decode();claim=json.loads(line);result["claim"]=claim
        require(claim["nonce"]==nonce and type(claim["pid"]) is int and claim["pid"]>0,"claim authenticated on own private pipe before OS child lookup")
        wrapper=identity(int(proc._handle));require(wrapper["pid"]==proc.pid,"owned OS handle PID")
        kernel,_=api();child_handle=kernel.OpenProcess(0x400|0x1000|0x100000,False,claim["pid"])
        require(child_handle,"query/synchronize child handle unavailable")
        actual=identity(child_handle);binding=relationship(wrapper,actual,claim,plan,nonce,os.getpid())
        result.update(wrapper=wrapper,actual_python=actual,binding=binding)
        require(binding["all_pass"],"prospective wrapper/child relationship failed")
        proc.stdin.write(json.dumps({"ack":nonce}).encode()+b"\n");proc.stdin.flush();proc.stdin.close()
        result["worker_exit_code"]=proc.wait(timeout=max(.01,deadline-time.monotonic()-1))
        require(result["worker_exit_code"]==0,"clean owned wrapper exit")
        require(kernel.WaitForSingleObject(child_handle,100)==0,"actual child exited before closeout")
        extra=proc.stdout.read(16385);error=proc.stderr.read(8193)
        require(not extra and len(error)<=8192,"bounded EOF/diagnostic output")
        result.update(stderr=error.decode(errors="replace"),eof_observed=True,status="WRAPPER_CHILD_RELATIONSHIP_CONFIRMED")
    except BaseException as error:result["primary_error"]=type(error).__name__+": "+str(error)
    finally:
        if proc is not None:
            try:
                if proc.poll() is None:
                    result["owned_terminate_attempted"]=True;proc.terminate()
                    try:proc.wait(timeout=.5)
                    except subprocess.TimeoutExpired:result["owned_kill_attempted"]=True;proc.kill();proc.wait(timeout=.5)
                result["owned_process_exited"]=proc.poll() is not None
                for pipe in (proc.stdin,proc.stdout,proc.stderr):
                    if pipe is not None:pipe.close()
            except BaseException as error:result["cleanup_errors"].append(type(error).__name__+": "+str(error))
        if reader is not None:reader.join(.5);result["reader_joined"]=not reader.is_alive()
        if child_handle is not None:
            kernel,_=api();result["actual_child_exited"]=kernel.WaitForSingleObject(child_handle,0)==0;kernel.CloseHandle(child_handle)
        result["elapsed_seconds"]=time.monotonic()-started
        if result["elapsed_seconds"]>10 or result["cleanup_errors"]:result["status"]="UNKNOWN"
        write("probe_result.json",result)
    print(json.dumps({k:result.get(k) for k in ("status","probe_children_started","elapsed_seconds","owned_Popen_pid","primary_error","cleanup_errors")}))
    return 0 if result["status"]=="WRAPPER_CHILD_RELATIONSHIP_CONFIRMED" else 1

if __name__=="__main__":
    if sys.argv[1:]==["child"]:child()
    elif sys.argv[1:]==["probe"]:raise SystemExit(probe())
    else:raise SystemExit("one probe/owned child only")
