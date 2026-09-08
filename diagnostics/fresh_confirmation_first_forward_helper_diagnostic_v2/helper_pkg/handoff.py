"""Prospective helper admission/cleanup joins; not a model controller."""
import json
import sys
from .support import need,Rejected,sha,encoded,component
from .selection import select,saved_selection,saved_graph,graph_sha
from .fingerprint import FrozenFunctions

MAX_RECEIPT=65536
PHASES=("SELECTION","INSTALL","IMMUTABLE_FINGERPRINT","ORIGINAL_SETUP","SETUP_PUBLICATION",
        "PRE_FORWARD","ORIGINAL_CLEANUP","FINAL_FINGERPRINT","HELPER_ROLLBACK",
        "ORIGINAL_GUARD_RESTORE","TERMINAL_PUBLICATION","ORIGINAL_STOP")
CODES={"SAME_OBJECT_IMPLEMENTATION_DRIFT","CALLABLE_BINDING_DRIFT","TARGET_DRIFT","HELPER_DRIFT",
    "INSTANCE_DRIFT","COMPLETE_TRAVERSAL_BIJECTION","COMPLETE_UNIQUE_SET","MAPPING_IDENTITY",
    "LAYER_LAYOUT","LAYER_COUNT","LAYER_KIND","FULL_LAYER_GDN","INSTALL_IO_OR_CALLBACK",
    "CLOSE_INCOMPLETE","NOT_ACTIVE","ORIGINAL_REJECTED","RECEIPT_ACK","RECEIPT_CAP"}
def finding(phase,exc):
    code=getattr(exc,"code",None)
    return {"phase":phase,"code":code if code in CODES else "OTHER_FINITE_FAILURE",
            "category":"IO_ERROR" if isinstance(exc,OSError) else "REJECTED" if code else "OTHER"}

class Handoff:
    def __init__(self,targets,hf_model,bridge_model,bridge_type,publisher,original_stop,original_restore,
                 *,execution,source_lock):
        self.targets=targets;self.hf=hf_model;self.bridge=bridge_model;self.bridge_type=bridge_type
        self.publisher=publisher;self.stop=original_stop;self.restore=original_restore
        need(len(encoded(execution))<=2048 and execution["mode"] in ("INERT_FIXTURE","REAL_QWEN"),"EXECUTION_SCHEMA")
        need((execution["mode"]=="INERT_FIXTURE")== (targets.mode=="INERT_FIXTURE"),"EXECUTION_MODE")
        self.execution=json.loads(encoded(execution));self.source_lock=json.loads(encoded(source_lock))
        self.state="NEW";self.phase="SELECTION";self.primary=None;self.secondary=[];self.events=[]
        self.selected=();self.installation=None;self.frozen=None;self.setup_ack=None;self.terminal_ack=None
        self.checks=[];self.original_setup_returned=False;self.original_cleanup_returned=False
        self.restore_attempted=False;self.restore_returned=False;self.closed=False;self.publication_failed=False
        self._admitted_raw=None
        self._reported_exceptions=[];self._cleanup_incoming_exception=None

    def _event(self,phase,edge):
        need(phase in PHASES and len(self.events)<48,"EVENT_BOUND")
        self.events.append({"phase":phase,"edge":edge})
    def _fail(self,phase,exc):
        if any(exc is old for old in self._reported_exceptions):return
        if len(self._reported_exceptions)<9:self._reported_exceptions.append(exc)
        entry=finding(phase,exc)
        if self.primary is None:self.primary=entry
        elif len(self.secondary)<8:self.secondary.append(entry)
        self.state="FAILED"  # Before diagnostic publication, restoration or rollback.
        try:self.stop("GDN_HELPER_HANDOFF_FAILURE")
        except BaseException as stop_exc:
            if len(self.secondary)<8:self.secondary.append(finding("ORIGINAL_STOP",stop_exc))

    def _write(self,name,value):
        raw=encoded(value);need(len(raw)<=MAX_RECEIPT,"RECEIPT_CAP")
        ack=self.publisher.write_new(name,raw)
        need(ack=={"name":name,"bytes":len(raw),"sha256":sha(raw)},"RECEIPT_ACK")
        return ack

    def _identity(self,label):
        self.targets.inspect();self.installation.inspect()
        now=select(self.hf,self.bridge,self.targets,self.bridge_type)
        need(tuple((k,id(v)) for k,v in now)==tuple((k,id(v)) for k,v in self.selected),
             "COMPLETE_TRAVERSAL_BIJECTION")
        need(graph_sha(now)==graph_sha(self.selected),"COMPLETE_ALIAS_GRAPH")
        digest=self.frozen.inspect(self.targets,self.installation)
        self.checks.append({"label":label,"fingerprint_sha256":digest,"selection":saved_selection(now),"selection_graph_sha256":graph_sha(now)})
        return digest

    def setup(self,original_setup):
        need(self.state=="NEW","NO_RETRY")
        self.state="SETTING_UP"
        try:
            self.phase="SELECTION";self._event(self.phase,"ENTER")
            self.selected=select(self.hf,self.bridge,self.targets,self.bridge_type)
            self._event(self.phase,"RETURN")
            self.phase="INSTALL";self._event(self.phase,"ENTER")
            self.installation=component().Installation(self.targets,self.selected,before_hook_reference=True)
            self.installation.install();self._event(self.phase,"RETURN")
            self.phase="IMMUTABLE_FINGERPRINT";self._event(self.phase,"ENTER")
            self.frozen=FrozenFunctions.capture(self.targets,self.installation)
            self._event(self.phase,"RETURN")
            self.phase="ORIGINAL_SETUP";self._event(self.phase,"ENTER")
            original=original_setup()  # Unchanged prospective HookRecorder/setup admission.
            self.original_setup_returned=True;self._event(self.phase,"RETURN")
            self._identity("POST_SETUP")
            payload={"schema":"gdn_helper_setup.v1","execution":self.execution,"source_lock":self.source_lock,
                "selection":saved_selection(self.selected),"selection_graph":saved_graph(self.selected),"fingerprint":json.loads(self.frozen.raw),
                "fingerprint_sha256":sha(self.frozen.raw),"checks":list(self.checks),
                "original_setup_returned":True,"component_status":self.installation.status(),
                "scientific_pass":False}
            self.phase="SETUP_PUBLICATION";self._event(self.phase,"ENTER")
            self.setup_ack=self._write("HELPER_SETUP.json",payload)
            self._admitted_raw=encoded({"setup_ack":self.setup_ack,"source_lock":self.source_lock,
                "execution":self.execution,"fingerprint_sha256":sha(self.frozen.raw),
                "selection":saved_selection(self.selected),"selection_graph_sha256":graph_sha(self.selected)})
            self._event(self.phase,"RETURN");self.state="ACTIVE"
            return original
        except BaseException as exc:
            self._event(self.phase,"RAISE");self._fail(self.phase,exc)
            self._restore()
            self.closed=True
            self._terminal()
            raise

    def checkpoint(self,original_check):
        need(self.state=="ACTIVE" and not self.closed and
             [x["label"] for x in self.checks]==["POST_SETUP"],"CHECKPOINT_SCHEDULE")
        self.phase="PRE_FORWARD";self._event(self.phase,"ENTER")
        try:
            result=original_check()  # Original latch/guard checks stay necessary and run first.
            self._identity("PRE_FORWARD")
            self._event(self.phase,"RETURN")
            return result
        except BaseException as exc:
            self._event(self.phase,"RAISE");self._fail(self.phase,exc)
            raise

    def _restore(self):
        if self.restore_attempted:return
        self.restore_attempted=True
        pending=sys.exception()
        if self.phase=="ORIGINAL_CLEANUP" and pending is not None and pending is not self._cleanup_incoming_exception:
            self._fail("ORIGINAL_CLEANUP",pending)
        try:
            if self.frozen is not None and self.installation.state=="ACTIVE":
                try:
                    self._event("FINAL_FINGERPRINT","ENTER")
                    self._identity("DIAGNOSTIC_CLOSEOUT")
                    self._event("FINAL_FINGERPRINT","RETURN")
                except BaseException as exc:
                    self._event("FINAL_FINGERPRINT","RAISE");self._fail("FINAL_FINGERPRINT",exc)
            if self.installation is not None:
                self._event("HELPER_ROLLBACK","ENTER")
                try:
                    if self.installation.state=="ACTIVE":self.installation.close()
                    need(self.installation.rollback_complete is True,"CLOSE_INCOMPLETE")
                    self._event("HELPER_ROLLBACK","RETURN")
                except BaseException as exc:
                    self._event("HELPER_ROLLBACK","RAISE");self._fail("HELPER_ROLLBACK",exc)
        finally:
            self._event("ORIGINAL_GUARD_RESTORE","ENTER")
            try:
                self.restore();self.restore_returned=True;self._event("ORIGINAL_GUARD_RESTORE","RETURN")
            except BaseException as exc:
                self._event("ORIGINAL_GUARD_RESTORE","RAISE");self._fail("ORIGINAL_GUARD_RESTORE",exc)

    def close(self,original_cleanup):
        need(not self.closed and self.state in ("ACTIVE","FAILED"),"NO_RETRY")
        self._cleanup_incoming_exception=sys.exception()
        result=None;original_error=None
        self.phase="ORIGINAL_CLEANUP";self._event(self.phase,"ENTER")
        try:
            result=original_cleanup(self._restore)  # Exact original cleanup receives a finally-restore wrapper.
            self.original_cleanup_returned=True;self._event(self.phase,"RETURN")
        except BaseException as exc:
            original_error=exc
            self._event(self.phase,"RAISE");self._fail(self.phase,exc)
        finally:
            if not self.restore_attempted:
                self._fail("ORIGINAL_CLEANUP",Rejected("ORIGINAL_REJECTED"))
                self._restore()
            self.closed=True
            if self.primary is None:self.state="CLOSED"
            self._terminal()
        if original_error is not None:raise original_error
        need(self.state=="CLOSED" and not self.publication_failed,"HANDOFF_CLOSE_INCOMPLETE")
        return result

    def _base_status(self):
        return {"schema":"gdn_helper_terminal.v1","execution":self.execution,"source_lock":self.source_lock,
            "state":self.state,"closed":self.closed,"primary":self.primary,"secondary":list(self.secondary),
            "setup_ack":self.setup_ack,"selection":saved_selection(self.selected),"checks":list(self.checks),
            "selection_graph_sha256":graph_sha(self.selected),
            "reference_fingerprint_sha256":sha(self.frozen.raw) if self.frozen else None,
            "original_setup_returned":self.original_setup_returned,
            "original_cleanup_returned":self.original_cleanup_returned,
            "restore_attempted":self.restore_attempted,"restore_returned":self.restore_returned,
            "component_status":self.installation.status() if self.installation else None,
            "events":list(self.events),"publication_failed":self.publication_failed,
            "scientific_pass":False,"full_scientific_finalizer_called":False,"normal_hook_checks_unrun":109}

    def _terminal(self):
        try:self.terminal_ack=self._write("HELPER_TERMINAL.json",self._base_status())
        except BaseException as exc:
            self.publication_failed=True;self._fail("TERMINAL_PUBLICATION",exc)

    def status(self):
        # Finite non-IO status must be copied to authoritative outer closeout even
        # if a full terminal was written physically and publisher then raised.
        result=self._base_status();result["terminal_ack"]=self.terminal_ack
        return result

    def admission(self):
        # Copy this once into the authoritative setup/outer-controller binding;
        # do not reconstruct it later from a terminal or edited saved receipt.
        return json.loads(self._admitted_raw) if self._admitted_raw else None
