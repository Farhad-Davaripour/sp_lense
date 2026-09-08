"""Independent saved-byte diagnostic check; never supplies scientific PASS."""
import hashlib
import json


def verify(root,expected_execution,source_sha256):
    terminal=json.loads((root/"WORKER_TERMINAL.json").read_bytes())
    result={"classification":"INCONCLUSIVE_STUDY","permits_scientific_pass":False,"diagnostic_verified":False}
    try:
        assert terminal["state"]["status"]=="INCONCLUSIVE_STUDY" and terminal["complete_evidence"] is False
        assert terminal["state"]["attempts"]=={"load":1,"forward":0,"derivative":0}
        assert len(terminal["state"]["cell_status"])==180 and set(terminal["state"]["cell_status"].values())=={"UNRUN"}
        assert len(terminal["state"]["request_status"])==48 and set(terminal["state"]["request_status"].values())=={"UNRUN"}
        assert terminal["execution"]==expected_execution
        diagnostic=terminal["loader_diagnostics"]
        assert diagnostic["execution"]==expected_execution and diagnostic["source_sha256"]==source_sha256
        assert diagnostic["first_failure"] is not None and diagnostic["retry_allowed"] is False
        result["first_failure"]=diagnostic["first_failure"]
        if diagnostic["diagnostic_io_failed"] or not diagnostic["complete_diagnostic_evidence"]:
            result["diagnostic_status"]="INCOMPLETE_DIAGNOSTIC_BYTES"
            return result
        raw=(root/"LOADER_DIAGNOSTICS.json").read_bytes()
        assert len(raw)<=2*1024**2 and hashlib.sha256(raw).hexdigest()==diagnostic["receipt_sha256"]
        saved=json.loads(raw)
        for key in ("execution","source_sha256","stage","first_failure","parameter_enumerations","already_computed_digests","recorder_status"):
            assert saved[key]==diagnostic[key]
        assert terminal["recorder_status"]==diagnostic["recorder_status"]
        assert not saved["exception_text_serialized"] and not saved["parameter_bytes_serialized"]
        result.update(diagnostic_verified=True,diagnostic_status="AUTHENTICATED_FINITE_FAILURE")
    except (AssertionError,ValueError,KeyError,OSError,TypeError):
        result["diagnostic_status"]="INCOMPLETE_OR_INVALID_DIAGNOSTIC_BYTES"
    return result
