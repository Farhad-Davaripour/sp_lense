"""Private owned bootstrap: adapter injection or separately disabled production."""
import json
import os
import sys
import threading
import time
from support import HERE, check_freeze, ownership, require, sha, write_new


def main():
    require(len(sys.argv) == 3 and sys.argv[1] in ("adapter_worker","production_worker","production_audit"), "fixed candidate lane")
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
        envelope = owned.strict_json(raw)
        expected = {"ownership":{"may_load":True,"nonce":nonce,"scope":"FAKE_WORK_ONLY"},
            "lane":lane,"absolute_deadline":deadline,"source_freeze_sha256":sha((HERE/"SOURCE_FREEZE.json").read_bytes())}
        if lane != "adapter_worker":
            from production_admission import admit_production
            admitted = admit_production("worker" if lane == "production_worker" else "audit")
            expected["production_release_sha256"] = json.loads((HERE/"PRODUCTION_AUTHORIZATION.json").read_bytes())["root_release_sha256"]
        require(envelope == expected,"exact separate authority and retained ownership permission")
        require(time.monotonic() < deadline, "no work beyond parent cutoff")
        write_new("owned/"+lane+"/BOOTSTRAP.json",{"actual_pid":os.getpid(),"parent_pid":os.getppid(),
            "permission_received":True,"backend_imports_before_permission":False,"absolute_deadline":deadline,
            "production_authorized":False})
    finally:
        watchdog.cancel()
    if lane == "adapter_worker":
        from adapter_tests import run
        return run(deadline)
    from production_run import worker,audit
    return worker(deadline) if lane == "production_worker" else audit(deadline)


if __name__ == "__main__":
    raise SystemExit(main())
