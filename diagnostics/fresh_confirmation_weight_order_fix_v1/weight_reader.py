"""Saved-only native binding joins; no weight access or scientific PASS."""
import collections
import json
import math
from support import sha

ALGORITHM_SHA="bee50e24d5fe1e4774b9f4b43c6755a9cfe75f7c8b8897ad986831726d21be3a"

def interpret(native_raw,terminal,execution,source_sha,expected):
    result={"interpretation":"BINDING_INCOMPLETE","binding_verified":False,"scientific_pass":False}
    try:
        assert len(native_raw)<=2*1024**2
        native=json.loads(native_raw)
        assert terminal["complete_diagnostic_evidence"] and not terminal["diagnostic_io_failed"]
        assert terminal["receipt_sha256"]==sha(native_raw) and native["execution"]==terminal["execution"]==execution
        p=native["legacy_weight_binding"]
        assert p==terminal["legacy_weight_binding"] and p["source_sha256"]==source_sha and p["execution"]==execution
        assert p["algorithm_source_sha256"]==ALGORITHM_SHA and p["expected_frozen_sha256"]==expected
        assert p["additional_hash_attempts"]==p["additional_hash_completed"]==1
        assert p["content_order"]=="COMPLETE_PRE_HOOK_LEGACY_STRONG_TUPLE" and p["registry_order"]=="UNCHANGED_CURRENT_NAMED_TRAVERSAL"
        if p["failure"]=="INITIAL_FINGERPRINT_MISMATCH":
            assert p["phases"]==["PRE_HOOK_LEGACY_HASH"] and p["pre_setup_sha256"]!=expected
            return dict(result,interpretation="INITIAL_FINGERPRINT_MISMATCH")
        assert p["phases"]==["PRE_HOOK_LEGACY_HASH","POST_HOOK_REFERENCE_BIJECTION","CONSTRUCTOR_RETAINED_LEGACY_HASH"]
        pre,ret,cur=p["before"],p["retained_after"],p["current_after"]
        assert 0<len(pre)==len(ret)==len(cur)<=1024
        for rows in (pre,ret,cur):
            seen=collections.Counter()
            for i,x in enumerate(rows):
                seen[x["identity"]]+=1
                assert x["ordinal"]==i and x["occurrence"]==seen[x["identity"]]
                assert x["element_size"]==4 and x["numel"]==math.prod(x["shape"]) and x["byte_length"]==4*x["numel"]
                assert x["device_type"]=="cpu" and x["dtype"] in ("torch.float32","float32")
        assert collections.Counter(x["identity"] for x in pre)==collections.Counter(x["identity"] for x in cur)
        fields=("identity","shape","dtype","device_type","numel","element_size","byte_length","requires_grad","version")
        by={(x["identity"],x["occurrence"]):x for x in pre}
        assert all(all(a[k]==b[k] for k in fields) for a,b in zip(pre,ret,strict=True))
        assert all(all(by[(x["identity"],x["occurrence"])][k]==x[k] for k in fields) for x in cur)
        for label,rows in (("BEFORE_HOOK_SETUP",pre),("ADAPTER_CAPTURE",cur)):
            original=native["parameter_enumerations"][label]
            original_raw=(json.dumps(original["records"],sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
            assert original["complete"] and sha(original_raw)==original["metadata_sha256"]
            assert len(original["records"])==len(rows)
            assert all(all(x[k]==y[k] for k in x) for x,y in zip(original["records"],rows,strict=True))
        assert p["pre_setup_sha256"]==expected
        assert p["constructor_sha256"]==native["already_computed_digests"]["actual_ordered_sha256"]
        assert native["already_computed_digests"]["expected_frozen_sha256"]==expected
        if p["constructor_sha256"]!=expected:
            assert p["failure"]=="CONSTRUCTOR_FINGERPRINT_MISMATCH"
            return dict(result,interpretation="CONSTRUCTOR_FINGERPRINT_MISMATCH")
        assert p["failure"] is None and native["first_failure"] is None
        return dict(result,interpretation="RETAINED_ORDER_BINDING_VERIFIED",binding_verified=True,parameter_occurrences=len(pre),
            fresh_initial_hash=p["pre_setup_sha256"],fresh_constructor_hash=p["constructor_sha256"],additional_hash_calls=1)
    except (AssertionError,ValueError,KeyError,TypeError,OverflowError):return result
