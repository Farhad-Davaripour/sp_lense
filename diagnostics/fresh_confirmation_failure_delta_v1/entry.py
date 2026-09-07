"""Private nonce bootstrap; only two fake lanes, no model imports before ACK."""
import json
import os
import sys
import threading
import time
from support import HERE, check_freeze, ownership, require, sha, write_new


def main():
    require(len(sys.argv) == 3 and sys.argv[1] in ("forced_stall","failure_worker"), "fixed fake lane")
    lane,deadline = sys.argv[1],float(sys.argv[2])
    check_freeze()
    native,owned = ownership()
    watchdog = threading.Timer(9.,lambda:os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    try:
        raw = sys.stdin.buffer.readline(8193)
        require(len(raw) <= 8192 and raw.endswith(b"\n"), "bounded nonce challenge")
        nonce = owned.strict_json(raw)["nonce"]
        require(isinstance(nonce,str) and len(nonce) == 64, "private nonce")
        kernel,_ = native.source.api()
        identity = native.source.identity(kernel.GetCurrentProcess())
        print(json.dumps({"kind":"claim","nonce":nonce,"pid":os.getpid(),"ppid":os.getppid(),
            "sys_executable":sys.executable,"base_executable":sys._base_executable,"os_identity":identity}),flush=True)
        raw = sys.stdin.buffer.readline(8193)
        require(len(raw) <= 8192 and raw.endswith(b"\n"), "bounded private permission")
        envelope = owned.strict_json(raw)
        require(envelope == {"ownership":{"may_load":True,"nonce":nonce,"scope":"FAKE_WORK_ONLY"},
            "lane":lane,"absolute_deadline":deadline,"source_freeze_sha256":sha((HERE/"SOURCE_FREEZE.json").read_bytes())},
            "exact fake-only lane, source and deadline permission")
        require(time.monotonic() < deadline, "no work after outer deadline")
        write_new("owned/"+lane+"/BOOTSTRAP.json",{"actual_pid":os.getpid(),"parent_pid":os.getppid(),
            "lane":lane,"permission_received":True,"model_imports_before_permission":False,"absolute_deadline":deadline})
    finally:
        watchdog.cancel()
    if lane == "forced_stall":
        emergency = threading.Timer(9.,lambda:os._exit(124))
        emergency.daemon = True
        emergency.start()
        write_new("owned/forced_stall/STALL_READY.json",{"forced_stall_entered":True,
            "payload_wait_seconds":300,"monotonic":time.monotonic(),"expected_outer_termination":True})
        os.close(1)
        os.close(2)  # EOF must not be mistaken for process quiescence.
        threading.Event().wait(300.)
        return 124
    from worker import run
    return run(deadline)


if __name__ == "__main__":
    raise SystemExit(main())
