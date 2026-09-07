"""Candidate private bootstrap; no real lane and no backend import before ACK."""
import json
import os
import sys
import threading
import time
from support import HERE, check_freeze, ownership, require, sha, write_new


def main():
    require(len(sys.argv) == 3 and sys.argv[1] == "runtime_worker", "fixed candidate lane")
    lane,deadline = sys.argv[1],float(sys.argv[2])
    check_freeze()
    native,owned = ownership()
    watchdog = threading.Timer(9.,lambda:os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    try:
        raw = sys.stdin.buffer.readline(8193)
        require(len(raw) <= 8192 and raw.endswith(b"\n"), "bounded private challenge")
        nonce = owned.strict_json(raw)["nonce"]
        require(isinstance(nonce,str) and len(nonce) == 64, "private nonce")
        kernel,_ = native.source.api()
        identity = native.source.identity(kernel.GetCurrentProcess())
        print(json.dumps({"kind":"claim","nonce":nonce,"pid":os.getpid(),"ppid":os.getppid(),
            "sys_executable":sys.executable,"base_executable":sys._base_executable,"os_identity":identity}),flush=True)
        raw = sys.stdin.buffer.readline(8193)
        require(len(raw) <= 8192 and raw.endswith(b"\n"), "bounded candidate permission")
        require(owned.strict_json(raw) == {"ownership":{"may_load":True,"nonce":nonce,"scope":"FAKE_WORK_ONLY"},
            "lane":lane,"absolute_deadline":deadline,"source_freeze_sha256":sha((HERE/"SOURCE_FREEZE.json").read_bytes())},
            "exact ownership, lane, deadline and source permission")
        require(time.monotonic() < deadline, "no work beyond parent cutoff")
        write_new("owned/runtime_worker/BOOTSTRAP.json",{"actual_pid":os.getpid(),"parent_pid":os.getppid(),
            "permission_received":True,"backend_imports_before_permission":False,"absolute_deadline":deadline,
            "production_authorized":False})
    finally:
        watchdog.cancel()
    from candidate import run
    return run(deadline, execution_mode="SYNTHETIC_ONLY")


if __name__ == "__main__":
    raise SystemExit(main())
