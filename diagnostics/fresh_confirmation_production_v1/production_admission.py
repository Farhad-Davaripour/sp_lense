"""Fail-closed production authority and usage checks, before research imports."""
import ast
import json
import math
import time
from admission import admit
from support import HERE, SOURCES, SCIENCE_COMMIT, SCIENCE, check_freeze, require, sha


def usage_value(receipt):
    source = SOURCES.read(SCIENCE_COMMIT,SCIENCE+"usage_receipt.py").decode()
    namespace = {"json":json,"math":math,"require":require,
        "BOOLEAN_FLAGS":("spendControlReached","rateLimitReached","isRateLimited","limitReached","exhausted")}
    for node in ast.parse(source).body:
        if isinstance(node,ast.FunctionDef):
            exec(compile(ast.get_source_segment(source,node),"pinned_usage:"+node.name,"exec"),namespace)
    return namespace["derive_standard_usage"](receipt)


def admit_production(lane):
    require(lane in ("worker","audit"), "fixed production lane")
    check_freeze()
    lock = json.loads((HERE/"PRODUCTION_AUTHORIZATION.json").read_bytes())
    require(lock["production_authorized"] is True and lock["root_release_sha256"] is not None,
            "production disabled; no real model release exists")
    # The current committed false authorization cannot satisfy this predicate.
    # A reviewed successor release must independently pin every receipt below.
    release_raw = (HERE/"ROOT_RELEASE.json").read_bytes()
    require(sha(release_raw) == lock["root_release_sha256"], "exact independently reviewed root release")
    release = json.loads(release_raw)
    require(release["production_authorized"] is True and release["source_freeze_sha256"] == sha((HERE/"SOURCE_FREEZE.json").read_bytes()), "root admits exact production sources")
    bound = admit()
    require(release["input_lock_sha256"] == bound["binding"]["input_lock_sha256"], "root admits exact complete inputs")
    usage_raw = (HERE/"PRODUCTION_USAGE.json").read_bytes()
    require(sha(usage_raw) == release["usage_sha256"], "fresh production usage receipt pin")
    usage = json.loads(usage_raw)
    derived = usage_value(usage["tool_result"])
    require(derived["used_percent"] < 100, "available unexhausted standard Codex usage")
    if lane == "worker":
        require(0 <= time.time()-usage["observed_unix_seconds"] <= 60, "fresh usage before model access")
    else:
        start = json.loads((HERE/"PRODUCTION_START.json").read_bytes())
        require(start["usage_sha256"] == release["usage_sha256"] and
            0 <= start["observed_unix_seconds"]-usage["observed_unix_seconds"] <= 60,
            "audit binds usage actually admitted before the fixed worker, not a new model attempt")
    require(release["bounded_schemas_admitted"] is True and release["hook_contract_reviewed"] is True,
            "complete pre-load source/input/schema admission")
    return {"binding":bound["binding"],"release":release,"usage":derived}
