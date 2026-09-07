"""Private retained-process bootstrap; execution mode is separately authenticated."""
import json
import os
import sys
import threading
import time
from support import HERE,require,ownership,check_freeze,sha,write_new


def main():
    require(len(sys.argv)==3 and sys.argv[1] in ("production_worker","production_audit"),"fixed production entrypoint lanes")
    lane,deadline=sys.argv[1],float(sys.argv[2])
    check_freeze()
    native,owned=ownership()
    watchdog=threading.Timer(9.,lambda:os._exit(124));watchdog.daemon=True;watchdog.start()
    try:
        raw=sys.stdin.buffer.readline(8193)
        require(0<len(raw)<=8192 and raw.endswith(b"\n"),"bounded challenge")
        nonce=owned.strict_json(raw)["nonce"]
        require(type(nonce) is str and len(nonce)==64,"private nonce")
        kernel,_=native.source.api()
        identity=native.source.identity(kernel.GetCurrentProcess())
        print(json.dumps({"kind":"claim","nonce":nonce,"pid":os.getpid(),"ppid":os.getppid(),
            "sys_executable":sys.executable,"base_executable":sys._base_executable,"os_identity":identity}),flush=True)
        raw=sys.stdin.buffer.readline(8193)
        require(0<len(raw)<=8192 and raw.endswith(b"\n"),"bounded permission")
        from authority import authenticate
        execution=authenticate()["execution"]
        require(owned.strict_json(raw)=={"ownership":{"may_load":True,"nonce":nonce,"scope":"RETAINED_PROCESS_ONLY"},
            "lane":lane,"absolute_deadline":deadline,"source_freeze_sha256":sha((HERE/"SOURCE_FREEZE.json").read_bytes()),
            "execution":execution},"exact independently authenticated mode plus ownership")
        require(time.monotonic()<deadline,"no substantive work after cutoff")
        write_new("owned/"+lane+"/BOOTSTRAP.json",{"actual_pid":os.getpid(),"parent_pid":os.getppid(),
            "permission_received":True,"backend_imports_before_permission":False,"execution":execution,"absolute_deadline":deadline})
    finally: watchdog.cancel()
    from production_run import worker,audit
    return worker(deadline) if lane=="production_worker" else audit(deadline)


if __name__=="__main__": raise SystemExit(main())
