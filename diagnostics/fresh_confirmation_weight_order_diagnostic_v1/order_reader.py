"""Independent finite A/B/C evidence interpretation; never a scientific verdict."""
import collections
import json
from support import HERE,sha
from real_boundary import strict

PHASES=["A_PRE_SETUP","B_RETAINED_POST_SETUP","C_CURRENT_POST_SETUP"]
ALGORITHM_SHA="bee50e24d5fe1e4774b9f4b43c6755a9cfe75f7c8b8897ad986831726d21be3a"

def need(value):
    if not value:raise ValueError("INCOMPLETE_ORDER_PROOF")
def hexsha(value):return type(value) is str and len(value)==64 and all(c in "0123456789abcdef" for c in value)
def rows(value):
    need(type(value) is list and 0<len(value)<=1024)
    seen=collections.Counter()
    for ordinal,p in enumerate(value):
        need(p["ordinal"]==ordinal and type(p["identity"]) is int and type(p["name"]) is str)
        seen[p["identity"]]+=1;need(p["occurrence"]==seen[p["identity"]])
        product=1
        for n in p["shape"]:need(type(n) is int and n>=0);product*=n
        need(product==p["numel"] and p["element_size"]==4 and p["byte_length"]==product*4
            and p["dtype"] in ("float32","torch.float32") and p["device_type"]=="cpu")
    return seen

def interpret(native_bytes,terminal,execution,source_sha256,expected_sha256):
    result={"interpretation":"PROOF_INCOMPLETE","permits_runtime_admission":False,"scientific_pass":False,"complete":False}
    try:
        need(type(native_bytes) is bytes and len(native_bytes)<=2*1024**2)
        need(terminal["receipt_sha256"]==sha(native_bytes) and terminal["complete_diagnostic_evidence"] is True
            and terminal["diagnostic_io_failed"] is False and terminal["execution"]==execution)
        native=strict(native_bytes);proof=native["order_fingerprint"]
        need(proof==terminal["order_fingerprint"] and native["execution"]==execution
            and proof["execution"]==execution and proof["source_sha256"]==source_sha256
            and proof["algorithm_source_sha256"]==ALGORITHM_SHA and proof["parameter_contents_recorded"] is False
            and proof["scientific_authorization"] is False)
        need(proof["phases"]==PHASES and proof["extra_hash_attempts"]==proof["extra_hash_completed"]==2
            and proof["incomplete_reason"] is None and proof["B_C_interval"]=="SOURCE_LOCKED_ADJACENT_B_THEN_UNCHANGED_C_NO_MODEL_MUTATOR")
        need(all(hexsha(proof[k]) for k in ("A","B","C")) and proof["expected_frozen_sha256"]==expected_sha256)
        need(native["already_computed_digests"]["actual_ordered_sha256"]==proof["C"]
            and native["already_computed_digests"]["expected_frozen_sha256"]==expected_sha256)
        before,retained,current=(proof[k] for k in ("before","retained_after","current_after"))
        counts=rows(before);need(rows(retained)==counts==rows(current))
        fields=("identity","occurrence","shape","dtype","device_type","requires_grad","version","numel","element_size","byte_length")
        need(len(before)==len(retained) and all(all(a[k]==b[k] for k in fields) for a,b in zip(before,retained,strict=True)))
        by_occurrence={(p["identity"],p["occurrence"]):p for p in before}
        for p in current:
            original=by_occurrence[p["identity"],p["occurrence"]]
            need(all(p[k]==original[k] for k in fields))
        for label,extended in (("BEFORE_HOOK_SETUP",before),("ADAPTER_CAPTURE",current)):
            ordinary=native["parameter_enumerations"][label]
            data=(json.dumps(ordinary["records"],sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
            need(ordinary["complete"] is True and sha(data)==ordinary["metadata_sha256"] and len(ordinary["records"])==len(extended))
            for a,b in zip(ordinary["records"],extended,strict=True):need(all(a[k]==b[k] for k in a))
        need(proof["bijection"] is True and proof["metadata_stable"] is True)
        permutation=[p["identity"] for p in before]!=[p["identity"] for p in current]
        result.update(complete=True,traversal_permutation=permutation,parameter_occurrences=len(before),unique_objects=len(counts),
            A=proof["A"],B=proof["B"],C=proof["C"],additional_hash_calls=2)
        if proof["A"]!=expected_sha256: verdict="INITIAL_FINGERPRINT_MISMATCH"
        elif proof["A"]!=proof["B"]: verdict="RETAINED_ORDER_CONTENT_CHANGE_OR_MEASUREMENT_INSTABILITY"
        elif proof["C"]!=proof["B"] and permutation: verdict="ORDER_EXPLANATION_SUPPORTED_FOR_THIS_CAPTURE"
        elif proof["C"]==proof["B"]: verdict="NO_CURRENT_FINGERPRINT_MISMATCH"
        else: verdict="INCONSISTENT_MEASUREMENT_NO_ORDER_REMEDY"
        result["interpretation"]=verdict
    except (KeyError,ValueError,TypeError,IndexError,OverflowError):pass
    return result
