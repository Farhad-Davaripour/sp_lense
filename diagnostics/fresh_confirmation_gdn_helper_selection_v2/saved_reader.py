"""Saved-only independent joins. Never imports component, loader or handoff."""
import json
from support import need,Rejected,sha,encoded
from selection import LABELS
from selection_reader import verify_graph

def judge(reader,*,execution,source_lock,controller_status,expected_setup):
    try:
        live=dict(controller_status)
        ack=live.pop("terminal_ack")
        need(type(ack) is dict and ack.get("name")=="HELPER_TERMINAL.json","TERMINAL_ACK")
        raw=reader(ack["name"]);need(len(raw)<=65536 and len(raw)==ack["bytes"] and sha(raw)==ack["sha256"],"TERMINAL_BYTES")
        terminal=json.loads(raw)
        need(terminal==live,"AUTHORITATIVE_TERMINAL_STATUS")
        need(terminal["execution"]==execution and terminal["source_lock"]==source_lock,"TERMINAL_BINDING")
        need(terminal["state"]=="CLOSED" and terminal["closed"] and terminal["primary"] is None and
             terminal["secondary"]==[] and not terminal["publication_failed"],"INCOMPLETE_HANDOFF")
        setup_ack=terminal["setup_ack"]
        need(type(setup_ack) is dict and setup_ack["name"]=="HELPER_SETUP.json","SETUP_ACK")
        need(type(expected_setup) is dict and expected_setup["execution"]==execution and
             expected_setup["source_lock"]==source_lock and expected_setup["setup_ack"]==setup_ack,
             "IMMUTABLE_SETUP_ADMISSION")
        setup_raw=reader(setup_ack["name"])
        need(len(setup_raw)<=65536 and len(setup_raw)==setup_ack["bytes"] and sha(setup_raw)==setup_ack["sha256"],"SETUP_BYTES")
        setup=json.loads(setup_raw)
        need(setup["schema"]=="gdn_helper_setup.v1" and setup["source_lock"]==source_lock and
             setup["execution"]==execution,"SETUP_BINDING")
        selection=setup["selection"]
        need([x["label"] for x in selection]==list(LABELS) and len({x["identity"] for x in selection})==18 and
             all(type(x["identity"]) is int and x["identity"]>0 for x in selection),"SELECTION_PROOF")
        need(terminal["selection"]==selection,"SELECTION_TERMINAL_JOIN")
        need(expected_setup["selection"]==selection,"ADMITTED_INSTANCE_SET")
        verify_graph(setup["selection_graph"],selection,execution["mode"],source_lock,
                     expected_setup["selection_graph_sha256"])
        graph_digest=sha(encoded(setup["selection_graph"]))
        need(terminal["selection_graph_sha256"]==graph_digest and
             all(x["selection_graph_sha256"]==graph_digest for x in terminal["checks"]),"SELECTION_GRAPH_TERMINAL")
        fingerprint=setup["fingerprint"]
        need(fingerprint["schema"]=="immutable_callable_fingerprint.v1" and
             1<=len(fingerprint["nodes"])<=12 and len(fingerprint["anchors"])==4,"FINGERPRINT_SCHEMA")
        digest=sha(encoded(fingerprint))
        need(digest==setup["fingerprint_sha256"]==terminal["reference_fingerprint_sha256"],"REFERENCE_FINGERPRINT_JOIN")
        need(digest==expected_setup["fingerprint_sha256"],"ADMITTED_FUNCTION_REFERENCE")
        checks=terminal["checks"]
        need([x["label"] for x in checks]==["POST_SETUP","PRE_FORWARD","DIAGNOSTIC_CLOSEOUT"],"CHECKPOINT_SCHEDULE")
        need(setup["checks"]==checks[:1] and all(x["selection"]==selection and x["fingerprint_sha256"]==digest for x in checks),"IMMUTABLE_REFERENCE_CHECKS")
        need(setup["original_setup_returned"] and terminal["original_setup_returned"] and
             terminal["original_cleanup_returned"] and terminal["restore_attempted"] and terminal["restore_returned"],
             "ORIGINAL_CHECKS_CLEANUP")
        before=setup["component_status"];after=terminal["component_status"]
        need(before["state"]=="ACTIVE" and after["state"]=="CLOSED" and after["rollback_complete"] is True and
             before["instances"]==after["instances"]==selection and after["first_failure"] is None,"COMPONENT_LIFECYCLE")
        for k in ("mode","conv_adapter_identity","conv_target_identity","chunk_target_identity","source_sha256"):
            need(before[k]==after[k],"HELPER_IDENTITY_JOIN")
        need(before["mode"]==("INERT_FIXTURE" if execution["mode"]=="INERT_FIXTURE" else "LIVE_EXISTING_MODULE"),"MODE_JOIN")
        need(terminal["scientific_pass"] is False and setup["scientific_pass"] is False and
             terminal["normal_hook_checks_unrun"]==109 and terminal["full_scientific_finalizer_called"] is False,"NO_SCIENTIFIC_UPGRADE")
        events=terminal["events"]
        expected=[("SELECTION","ENTER"),("SELECTION","RETURN"),("INSTALL","ENTER"),("INSTALL","RETURN"),
            ("IMMUTABLE_FINGERPRINT","ENTER"),("IMMUTABLE_FINGERPRINT","RETURN"),("ORIGINAL_SETUP","ENTER"),("ORIGINAL_SETUP","RETURN"),
            ("SETUP_PUBLICATION","ENTER"),("SETUP_PUBLICATION","RETURN"),("PRE_FORWARD","ENTER"),("PRE_FORWARD","RETURN"),
            ("ORIGINAL_CLEANUP","ENTER"),("FINAL_FINGERPRINT","ENTER"),("FINAL_FINGERPRINT","RETURN"),
            ("HELPER_ROLLBACK","ENTER"),("HELPER_ROLLBACK","RETURN"),("ORIGINAL_GUARD_RESTORE","ENTER"),
            ("ORIGINAL_GUARD_RESTORE","RETURN"),("ORIGINAL_CLEANUP","RETURN")]
        need([(x["phase"],x["edge"]) for x in events]==expected,"EVENT_ORDER")
        return {"status":"HANDOFF_DIAGNOSTIC_JOIN_COMPLETE","binding_verified":True,"scientific_pass":False,
                "real_execution_authorized":False,"mode":execution["mode"],"instance_count":18,"normal_hook_checks_unrun":109}
    except BaseException as exc:
        code=exc.code if type(exc) is Rejected else "SAVED_HANDOFF_INCOMPLETE"
        return {"status":"INCOMPLETE","binding_verified":False,"scientific_pass":False,
                "real_execution_authorized":False,"failure_code":code}
