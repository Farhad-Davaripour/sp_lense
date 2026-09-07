"""One fake worker only; waits for permission, closes output, stays alive <=9s."""
import json,os,sys,threading
from native import source
from owned import strict_json,require
watchdog=threading.Timer(9.,lambda:os._exit(124));watchdog.daemon=True;watchdog.start()
try:
    line=sys.stdin.buffer.readline(8193);require(len(line)<=8192 and line.endswith(b"\n"),"challenge/pipe loss")
    nonce=strict_json(line)["nonce"];require(isinstance(nonce,str) and len(nonce)==64,"nonce")
    kernel,_=source.api();identity=source.identity(kernel.GetCurrentProcess())
    print(json.dumps({"kind":"claim","nonce":nonce,"pid":os.getpid(),"ppid":os.getppid(),"sys_executable":sys.executable,
        "base_executable":sys._base_executable,"os_identity":identity}),flush=True)
    line=sys.stdin.buffer.readline(8193);require(len(line)<=8192 and line.endswith(b"\n"),"permission/pipe loss")
    permission=strict_json(line);require(permission=={"may_load":True,"nonce":nonce,"scope":"FAKE_WORK_ONLY"},"fake permission")
    print(json.dumps({"kind":"ready_to_close_outputs","nonce":nonce}),flush=True)
    os.close(1);os.close(2)
    threading.Event().wait(9.)
finally:watchdog.cancel()
