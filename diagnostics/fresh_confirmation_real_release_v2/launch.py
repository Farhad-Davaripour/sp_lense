"""Trusted-root CLI. No real release files are provided by this preparation."""
import argparse
import json
import os
import time
from pathlib import Path
from real_boundary import MAIN,Boundary,need,encoded,write_exclusive,digest,strict


def preflight(lock_sha):
    try:
        value=Boundary().verify_chain(lock_sha)
        return {"status":"ROOT_AUTHORITY_PRESENT_NOT_LAUNCHED","intended_mode":"REAL_QWEN",
            "observed_model_work":"NONE","production_authorized":True}
    except (ValueError,OSError):
        return {"status":"DISABLED_NO_APPROVED_REAL_RELEASE","intended_mode":"REAL_QWEN",
            "observed_model_work":"NONE","production_authorized":False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--authority-lock",required=True)
    parser.add_argument("--approved-lock-sha256",required=True)
    parser.add_argument("--preflight",action="store_true")
    args=parser.parse_args()
    expected=MAIN/"root_release"/"AUTHORITY_LOCK.json"
    need(Path(args.authority_lock).absolute()==expected,"D_FIXED_AUTHORITY_PATH")
    if args.preflight:
        result=preflight(args.approved_lock_sha256)
        print(json.dumps(result,sort_keys=True))
        return 0 if result["production_authorized"] else 2
    boundary=Boundary()
    pin,identity=boundary.admit_once(args.approved_lock_sha256)
    os.environ["SP_CONFIRMATION_AUTHORITY_SHA"]=args.approved_lock_sha256
    os.environ["SP_CONFIRMATION_ADMISSION_SHA"]=pin
    result=None
    try:
        from production_run import controller
        result=controller(time.monotonic()+1995.)
        return 0 if result["audit_completed"] else 1
    finally:
        parent_sha=None
        try:
            terminal_path=boundary.control_path("LOADER_TERMINAL.json")
            observed=strict(boundary.read_control("LOADER_TERMINAL.json"))["observed_model_work"] if terminal_path.exists() else (
                "UNKNOWN_MISSING_LOADER_TERMINAL" if boundary.control_path("LOADER_STARTED.json").exists() else "NONE_LOADER_NOT_DISPATCHED")
            if boundary.control_path("PARENT_FINAL.json").exists(): parent_sha=digest(boundary.read_control("PARENT_FINAL.json"))
        except (OSError,ValueError,KeyError): observed="UNKNOWN_TERMINAL_READ_FAILURE"
        write_exclusive(boundary.control/"ATTEMPT_TERMINAL.json",encoded({"schema":"real_attempt_terminal.v1",
            "execution":identity["execution"],"observed_model_work":observed,"parent_final_sha256":parent_sha,
            "attempt_finished":result is not None,"real_scientific_success":bool(result and result["classification"]=="PASS_STUDY")}),boundary.root)


if __name__=="__main__":raise SystemExit(main())
