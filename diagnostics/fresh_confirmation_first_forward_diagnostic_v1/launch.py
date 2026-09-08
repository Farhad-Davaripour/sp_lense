"""Disabled until separate root release; one unedited baseline, no scientific study."""
import argparse
import json
import os
import time
from pathlib import Path
from real_boundary import MAIN,Boundary,need,encoded,write_exclusive,digest,strict

def preflight(lock_sha):
    try:
        Boundary().verify_chain(lock_sha)
        return {"status":"ROOT_AUTHORITY_PRESENT_NOT_LAUNCHED","scope":"ONE_LOCKED_BASELINE_1LOAD_MAX1F_0D",
            "observed_model_work":"NONE","production_authorized":True,"scientific_execution_allowed":False}
    except (ValueError,OSError):
        return {"status":"DISABLED_NO_APPROVED_REAL_RELEASE","scope":"ONE_LOCKED_BASELINE_1LOAD_MAX1F_0D",
            "observed_model_work":"NONE","production_authorized":False,"scientific_execution_allowed":False}

def terminal(boundary,identity,result):
    parent_sha=None
    try:
        observed=strict(boundary.read_control("LOADER_TERMINAL.json"))["observed_model_work"] if boundary.control_path("LOADER_TERMINAL.json").exists() else (
            "UNKNOWN_MISSING_LOADER_TERMINAL" if boundary.control_path("LOADER_STARTED.json").exists() else "NONE_LOADER_NOT_DISPATCHED")
        if boundary.control_path("PARENT_FINAL.json").exists(): parent_sha=digest(boundary.read_control("PARENT_FINAL.json"))
    except (OSError,ValueError,KeyError): observed="UNKNOWN_TERMINAL_READ_FAILURE"
    return write_exclusive(boundary.control/"ATTEMPT_TERMINAL.json",encoded({"schema":"first_forward_attempt_terminal.v1",
        "execution":identity["execution"],"observed_model_work":observed,"parent_final_sha256":parent_sha,
        "attempt_finished":result is not None,"real_scientific_success":False,"scientific_execution_allowed":False}),boundary.root)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--authority-lock",required=True)
    parser.add_argument("--approved-lock-sha256",required=True);parser.add_argument("--preflight",action="store_true")
    args=parser.parse_args()
    need(Path(args.authority_lock).absolute()==MAIN/"root_release/AUTHORITY_LOCK.json","D_FIXED_AUTHORITY_PATH")
    need(not any(os.environ.get(key) for key in ("SP_SETUP_FIXTURE","SP_SETUP_BATCH_LOCK","SP_FIRST_FORWARD_FIXTURE")),"D_NO_REAL_FIXTURE_OVERRIDE")
    if args.preflight:
        result=preflight(args.approved_lock_sha256);print(json.dumps(result,sort_keys=True));return 0 if result["production_authorized"] else 2
    boundary=Boundary();pin,identity=boundary.admit_once(args.approved_lock_sha256)
    os.environ["SP_CONFIRMATION_AUTHORITY_SHA"]=args.approved_lock_sha256;os.environ["SP_CONFIRMATION_ADMISSION_SHA"]=pin
    result=None
    try:
        from setup_budget import Budget
        from production_run import controller
        result=controller(Budget(time.monotonic()),"real_first_forward_diagnostic")
        return 0 if result["audit_completed"] else 1
    finally: terminal(boundary,identity,result)

if __name__=="__main__": raise SystemExit(main())
