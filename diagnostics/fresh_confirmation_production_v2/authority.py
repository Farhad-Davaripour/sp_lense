"""Acyclic source manifest -> release -> authorization -> caller-pinned lock.

Mode is a required authenticated value, never a directory-name inference.
The injection authorization can never authorize the real loader.
"""
import json
import os
import time
from support import HERE, SOURCES, require, sha, check_freeze, fixture
from admission import admit


def check_chain(manifest_raw,release_raw,authorization_raw,lock,expected_lock_sha):
    require(sha((json.dumps(lock,sort_keys=True,indent=2)+"\n").encode()) == expected_lock_sha,"caller-pinned authority lock")
    require(sha(manifest_raw)==lock["source_sha256"] and sha(release_raw)==lock["release_sha256"] and
        sha(authorization_raw)==lock["authorization_sha256"],"exact acyclic chain bytes")
    release,auth=json.loads(release_raw),json.loads(authorization_raw)
    require(release["schema"]=="confirmation_release.v2" and auth["schema"]=="confirmation_authorization.v2","explicit authority schemas")
    require(release["source_sha256"]==sha(manifest_raw) and auth["release_sha256"]==sha(release_raw),"one-way dependency links")
    require(auth["execution_mode"] in ("INJECTION_MODULE","REAL_QWEN") and auth["execution_mode"]==release["execution_mode"],"exact authorized mode")
    require(auth["allow_execute"] is True,"execution permission required")
    real=auth["execution_mode"]=="REAL_QWEN"
    require(auth["production_authorized"] is real and release["production_authorized"] is real,"no injection relabeling")
    require(release["bounded_schemas_admitted"] is True and release["hook_contract_reviewed"] is True,"reviewed component conditions")
    require(release["production_seconds"]=={"worker":1800,"cleanup":15,"audit":180} and release["production_ceiling"]=={"forwards":180,"derivatives":48,"loads":1,"tokens":160},"unchanged production settings")
    return release,auth


def authenticate():
    check_freeze()
    lock=json.loads((HERE/"AUTHORITY_LOCK.json").read_bytes())
    expected=os.environ.get("SP_CONFIRMATION_AUTHORITY_SHA","")
    release,auth=check_chain((HERE/"SOURCE_FREEZE.json").read_bytes(),(HERE/"RELEASE.json").read_bytes(),
        (HERE/"AUTHORIZATION.json").read_bytes(),lock,expected)
    bound=admit()
    require(release["input_lock_sha256"]==bound["binding"]["input_lock_sha256"],"exact 24/48 complete input binding")
    execution={"schema":"confirmation_execution.v2","mode":auth["execution_mode"],
        "production_authorized":auth["production_authorized"],"real_model_work":auth["execution_mode"]=="REAL_QWEN",
        "source_sha256":lock["source_sha256"],"release_sha256":lock["release_sha256"],
        "authorization_sha256":lock["authorization_sha256"],"authority_lock_sha256":expected,
        "input_lock_sha256":bound["binding"]["input_lock_sha256"],"fixture_id":fixture()}
    require(auth["execution_mode"]=="INJECTION_MODULE","this prepared successor does not release REAL_QWEN")
    require(fixture() in auth["fixtures"],"one predeclared injection fixture")
    usage=json.loads((HERE/"USAGE_RECEIPT.json").read_bytes())
    require(sha((HERE/"USAGE_RECEIPT.json").read_bytes())==release["usage_sha256"],"usage receipt bytes")
    require(0<=usage["derived"]["used_percent"]<100,"available standard usage")
    # The one batch start, not each child/audit, admits the fresh receipt.
    start=json.loads((HERE/"BATCH_STARTED.json").read_bytes())
    require(start["usage_sha256"]==release["usage_sha256"] and 0<=start["observed_unix_seconds"]-usage["observed_unix_seconds"]<=120,"usage admitted before single batch")
    return {"execution":execution,"release":release,"binding":bound["binding"]}


def independent_execution(saved):
    """Reader-side reauthentication; does not import a writer/backend/torch."""
    current=authenticate()["execution"]
    require(saved==current,"saved execution mode and authority independently authenticated")
    return current
