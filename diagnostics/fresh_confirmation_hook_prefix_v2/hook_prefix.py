"""Typed negative hook closeout; inherited full-run predicates remain unchanged."""
import hashlib
import importlib.util
import json
from pathlib import Path

BASE_PATH=Path(__file__).resolve().parents[1]/"fresh_confirmation_hook_failclosed_v1"/"hook_evidence.py"
BASE_SHA="f86ab31d463e9be3b25f8e2be0a45a50600da0b54e61d89189c58fd068f80484"
if hashlib.sha256(BASE_PATH.read_bytes()).hexdigest()!=BASE_SHA:
    raise RuntimeError("PINNED_HOOK_COMPONENT_CHANGED")
_spec=importlib.util.spec_from_file_location("pinned_failclosed_hook_v1",BASE_PATH)
b=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b)
HookStopped=b.HookStopped
FileIO=b.FileIO
CODES=b.CODES
REASONS=frozenset(("ROUTING","FINITE_ELIGIBILITY","ENDPOINT_BEHAVIOR"))

def valid_hash(value):
    return type(value) is str and len(value)==64 and all(c in "0123456789abcdef" for c in value)

class DispatchLatch(b.DispatchLatch):
    def __init__(self,schedule_ids):
        super().__init__(schedule_ids)
        self._scientific=None
    @property
    def scientific(self):
        return None if self._scientific is None else dict(self._scientific)
    @property
    def terminal(self):
        return self._state in ("TERMINAL","SCIENTIFIC_STOP")
    def scientific_stop(self,reason,finding_sha256,checks):
        if self._scientific is not None:
            self.trip("H_REENTRY")
            raise HookStopped("H_REENTRY")
        if self._state!="ADMITTED" or reason not in REASONS or not valid_hash(finding_sha256):
            self.trip("H_SCHEMA")
            raise HookStopped("H_SCHEMA")
        self._scientific={"reason_code":reason,"finding_sha256":finding_sha256,
                          "cursor":self.cursor,"completed_checks":checks}
        self._state="SCIENTIFIC_STOP"
    def require_dispatch(self):
        if self._scientific is not None:
            self.trip("H_REENTRY")
            raise HookStopped("H_REENTRY")
        return super().require_dispatch()
    def _admit(self):
        if self._scientific is not None:
            self.trip("H_REENTRY")
            raise HookStopped("H_REENTRY")
        return super()._admit()

def cleanup_labels(labels,count):
    """Only the active request exit, then the matrix final check; no caller override."""
    prefix=labels[:count]
    if prefix and prefix[-1]=="matrix-finally":
        return ()
    active=None
    for label in prefix:
        if label.startswith("REQUEST-entry:"):
            b.need(active is None,"H_SCHEMA");active=label[len("REQUEST-entry:"):]
        elif label.startswith("REQUEST-exit:"):
            b.need(active==label[len("REQUEST-exit:"):],"H_SCHEMA");active=None
    tail=labels[count:]
    result=[]
    if active is not None:
        exit_label="REQUEST-exit:"+active
        b.need(exit_label in tail,"H_SCHEMA");result.append(exit_label)
    b.need("matrix-finally" in tail,"H_SCHEMA")
    result.append("matrix-finally")
    b.need(len(result)<=len(tail),"H_SCHEMA")
    return tuple(result)

class Recorder(b.Recorder):
    def __init__(self,output_root,latch,expected_check_labels,io=None):
        b.need(isinstance(latch,DispatchLatch))
        super().__init__(output_root,latch,expected_check_labels,io)
        self.terminal_labels=()
        self.terminal_checks=0
        self.terminal_attempts=0
        self.prefix_index_sha=None
    def scientific_stop(self,reason_code,finding_sha256):
        # No diagnostic codec or IO occurs before the independent terminal state.
        try:
            self.latch.scientific_stop(reason_code,finding_sha256,self.checks)
            self.terminal_labels=cleanup_labels(self.labels,self.checks)
            event={"schema":"fresh_hook_scientific_stop_v2","scientific_stop":self.latch.scientific,
                "schedule_sha256":b.sha(b.encode(self.latch.schedule)),
                "full_check_labels_sha256":b.sha(b.encode(self.labels)),
                "remaining_ids":list(self.latch.remaining),
                "normal_checks_unrun":list(self.labels[self.checks:]),
                "terminal_cleanup_labels":list(self.terminal_labels),
                "full_run_hook_pass":False,"planned_checks":b.MAX_CHECKS}
            raw=b.encode(event);b.need(len(raw)<=b.JSON_CAP,"H_CAPACITY")
            self._write([("hook_evidence/scientific_stop.json",raw)],0,keep_index=False)
            return self.status()
        except BaseException as error:
            self._fault(error.code if type(error) is HookStopped else "H_CLOSEOUT",original=error)
            raise HookStopped(self.latch.primary_code) from None
    def terminal_cleanup(self,provider,label):
        try:
            b.need(self.latch._state=="SCIENTIFIC_STOP" and self.latch.primary_code is None and
                   self.prefix_index_sha is None,"H_REENTRY")
            self.terminal_attempts+=1
            b.need(self.terminal_attempts<=len(self.terminal_labels) and
                   label==self.terminal_labels[self.terminal_attempts-1],"H_SCHEMA")
            inspection=provider()
            changes=b.differences(self.reference,inspection["current"])
            if changes:
                self._fault("H_IDENTITY",changes)
                raise HookStopped("H_IDENTITY")
            b.need(inspection["matches"] is True and inspection["changes"]==[],"H_IDENTITY")
            event={"terminal_check":self.terminal_attempts,"label":label,"normal_scheduled_check":False,
                "matches":True,"complete_evidence":True,"changes":[],
                "reference_sha256":self.refs["reference"]["raw_sha256"]}
            raw=b.encode(event);b.need(len(raw)<=b.CHECK_BYTES,"H_CAPACITY")
            self._write([("hook_evidence/terminal_cleanup.jsonl",raw)],0,keep_index=False,append=True)
            self.terminal_checks+=1
            return True
        except BaseException as error:
            self._fault(error.code if type(error) is HookStopped else "H_CAPTURE",original=error)
            raise HookStopped(self.latch.primary_code) from None
    def finish_scientific_stop(self):
        try:
            b.need(self.latch._state=="SCIENTIFIC_STOP" and self.latch.primary_code is None and
                   self.prefix_index_sha is None,"H_CLOSEOUT")
            stop=self.latch.scientific
            b.need(stop is not None and stop["cursor"]==self.latch.cursor and
                   stop["completed_checks"]==self.checks and self.check_attempts==self.checks,"H_CLOSEOUT")
            b.need(self.terminal_checks==self.terminal_attempts==len(self.terminal_labels),"H_CLOSEOUT")
            b.need(set(self.io.hook_files())==set(self.files),"H_AUTH")
            for path,record in self.files.items():
                raw=self.io.read(path)
                b.need(len(raw)==record["bytes"] and b.sha(raw)==record["sha256"],"H_AUTH")
            index={"schema":"fresh_hook_scientific_prefix_v2","attempt_complete":False,"full_run_hook_pass":False,
                "scientific_stop":stop,"schedule":list(self.latch.schedule),"planned_checks":b.MAX_CHECKS,
                "check_labels":list(self.labels),"normal_checks":self.checks,
                "remaining_ids":list(self.latch.remaining),"normal_checks_unrun":list(self.labels[self.checks:]),
                "terminal_cleanup_labels":list(self.terminal_labels),"terminal_cleanup_checks":self.terminal_checks,
                "files":list(self.files.values()),"setup_values":self.refs}
            raw=b.encode(index);b.need(len(raw)<=b.INDEX_RESERVE,"H_CAPACITY")
            self._write([("hook_evidence/prefix_index.json",raw)],0,keep_index=False)
            self.prefix_index_sha=b.sha(raw)
            return {"hook_status":"SCIENTIFIC_STOP_COMPLETE_PREFIX","permits_pass":False,
                    "prefix_index_sha256":self.prefix_index_sha,"scientific_stop":stop}
        except BaseException as error:
            self._fault(error.code if type(error) is HookStopped else "H_CLOSEOUT",original=error)
            raise HookStopped(self.latch.primary_code) from None
    def status(self):
        value=super().status()
        if self.latch.scientific is not None:
            value.update(scientific_stop=self.latch.scientific,prefix_index_sha256=self.prefix_index_sha,
                normal_checks=self.checks,terminal_cleanup_checks=self.terminal_checks,
                terminal_cleanup_attempts=self.terminal_attempts,terminal_cleanup_labels=list(self.terminal_labels),
                permits_pass=False)
        return value

# Normal Recorder.finish/inspect/admit, byte serializer and full judge are inherited, not rewritten.
judge_full=b.judge

def judge_prefix(root,expected_labels,expected_schedule,expected_source_lock,controller_status,
                 expected_scientific_stop,io=None):
    """Authenticate a negative prefix, never supply hook/scientific PASS."""
    try:
        b.need(b.ids_ok(expected_labels,b.MAX_CHECKS) and len(expected_labels)==b.MAX_CHECKS and
               b.ids_ok(expected_schedule,180),"H_SCHEMA")
        stop=expected_scientific_stop
        b.need(type(stop) is dict and set(stop)=={"reason_code","finding_sha256","cursor","completed_checks"} and
               stop["reason_code"] in REASONS and valid_hash(stop["finding_sha256"]) and
               type(stop["cursor"]) is int and 0<=stop["cursor"]<=len(expected_schedule) and
               type(stop["completed_checks"]) is int and 0<=stop["completed_checks"]<=b.MAX_CHECKS,"H_AUTH")
        count=stop["completed_checks"]
        terminal=cleanup_labels(expected_labels,count)
        io=io if io is not None else FileIO(root,create=False)
        files=io.hook_files()
        b.need("hook_evidence/fault.json" not in files and "hook_evidence/index.json" not in files and
               "hook_evidence/prefix_index.json" in files,"H_CLOSEOUT")
        b.need(all(0<=n<=b.FILE_CAP for n in files.values()) and sum(files.values())<=b.HOOK_CAP and
               io.total_bytes()<=b.AGGREGATE_CAP,"H_CAPACITY")
        raw=io.read("hook_evidence/prefix_index.json");b.need(len(raw)<=b.INDEX_RESERVE,"H_AUTH")
        expected_status={"latch_state":"SCIENTIFIC_STOP","index_sha256":None,"terminal":True,"primary_code":None,
            "secondary_codes":[],"receipt_failed":False,"diagnostic_complete":False,
            "remaining_ids":list(expected_schedule[stop["cursor"]:]),"exception_text_serialized":False,
            "permits_pass":False,"scientific_stop":stop,"prefix_index_sha256":b.sha(raw),
            "normal_checks":count,"terminal_cleanup_checks":len(terminal),"terminal_cleanup_attempts":len(terminal),
            "terminal_cleanup_labels":list(terminal)}
        b.need(controller_status==expected_status,"H_CLOSEOUT")
        index=json.loads(raw)
        b.need(index["schema"]=="fresh_hook_scientific_prefix_v2" and index["attempt_complete"] is False and
               index["full_run_hook_pass"] is False and index["scientific_stop"]==stop and
               index["schedule"]==list(expected_schedule) and index["planned_checks"]==b.MAX_CHECKS and
               index["check_labels"]==list(expected_labels) and index["normal_checks"]==count and
               index["remaining_ids"]==list(expected_schedule[stop["cursor"]:]) and
               index["normal_checks_unrun"]==list(expected_labels[count:]) and
               index["terminal_cleanup_labels"]==list(terminal) and index["terminal_cleanup_checks"]==len(terminal),"H_AUTH")
        records=index["files"];mapping={r["path"]:r for r in records}
        b.need(len(records)==len(mapping) and set(files)==set(mapping)|{"hook_evidence/prefix_index.json"},"H_AUTH")
        for path,record in mapping.items():
            data=io.read(path);b.need(len(data)==record["bytes"] and b.sha(data)==record["sha256"],"H_AUTH")
        source_bytes=io.read("hook_evidence/source_lock.json")
        setup_bytes=io.read("hook_evidence/setup_receipt.json")
        b.need(len(source_bytes)<=b.JSON_CAP and len(setup_bytes)<=b.JSON_CAP,"H_AUTH")
        b.need(json.loads(source_bytes)==expected_source_lock,"H_AUTH")
        setup=json.loads(setup_bytes)
        b.need(setup["schema"]=="fresh_hook_setup_v1" and setup["complete"] is True and setup["forward_calls"]==0 and
               setup["source_lock_sha256"]==b.sha(source_bytes) and setup["values"]==index["setup_values"] and
               setup["reserved_checks"]==b.MAX_CHECKS and setup["check_record_bytes"]==b.CHECK_BYTES and
               setup["fault_reserve_bytes"]==b.FAULT_RESERVE and setup["index_reserve_bytes"]==b.INDEX_RESERVE,"H_AUTH")
        values={};used={"hook_evidence/source_lock.json","hook_evidence/setup_receipt.json","hook_evidence/scientific_stop.json"}
        for name,ref in setup["values"].items():
            value,paths=b._read_value(io,ref,mapping,name);values[name]=value;used.update(paths)
        b.need(set(values)=={"setup_before","reference","setup_changes"} and
               b.differences(values["setup_before"],values["reference"])==values["setup_changes"]["changes"],"H_AUTH")
        evidence_bytes=io.read("hook_evidence/scientific_stop.json")
        b.need(len(evidence_bytes)<=b.JSON_CAP,"H_AUTH");evidence=json.loads(evidence_bytes)
        b.need(evidence=={"schema":"fresh_hook_scientific_stop_v2","scientific_stop":stop,
            "schedule_sha256":b.sha(b.encode(tuple(expected_schedule))),
            "full_check_labels_sha256":b.sha(b.encode(tuple(expected_labels))),
            "remaining_ids":list(expected_schedule[stop["cursor"]:]),"normal_checks_unrun":list(expected_labels[count:]),
            "terminal_cleanup_labels":list(terminal),"full_run_hook_pass":False,"planned_checks":b.MAX_CHECKS},"H_AUTH")
        for path,n,labels,is_terminal in (("hook_evidence/checks.jsonl",count,expected_labels[:count],False),
                 ("hook_evidence/terminal_cleanup.jsonl",len(terminal),terminal,True)):
            if n==0:
                b.need(path not in mapping,"H_AUTH");continue
            used.add(path);lines=io.read(path).splitlines()
            b.need(len(lines)==n and all(len(line)+1<=b.CHECK_BYTES for line in lines),"H_AUTH")
            for i,line in enumerate(lines):
                event=json.loads(line)
                expected={"label":labels[i],"matches":True,"complete_evidence":True,"changes":[],
                    "reference_sha256":setup["values"]["reference"]["raw_sha256"]}
                expected.update({"terminal_check":i+1,"normal_scheduled_check":False} if is_terminal else {"check":i+1})
                b.need(event==expected,"H_AUTH")
        b.need(used==set(mapping),"H_AUTH")
        return {"hook_status":"SCIENTIFIC_STOP_COMPLETE_PREFIX","prefix_verified":True,"permits_pass":False,
            "scientific_stop":stop,"normal_checks":count,"terminal_cleanup_checks":len(terminal),
            "planned_checks":b.MAX_CHECKS,"planned_schedule_slots":len(expected_schedule),
            "remaining_ids":list(expected_schedule[stop["cursor"]:])}
    except BaseException as error:
        return {"hook_status":"INCOMPLETE","execution_status":"INCONCLUSIVE","prefix_verified":False,
            "permits_pass":False,"code":error.code if type(error) is HookStopped else "H_AUTH",
            "scientific_stop":expected_scientific_stop,"exception_text_serialized":False}
