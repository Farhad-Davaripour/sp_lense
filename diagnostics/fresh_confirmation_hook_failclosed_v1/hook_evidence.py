"""Pure fail-closed hook evidence boundary; no model or tokenizer imports."""
import hashlib
import json
import os
from pathlib import Path
import zlib

MIB=1024**2
HOOK_CAP=16*MIB
AGGREGATE_CAP=288*MIB
FILE_CAP=5*MIB
RAW_CAP=64*MIB
CHUNK=MIB
MAX_CHECKS=109
CHECK_BYTES=4096
FAULT_RESERVE=65536
INDEX_RESERVE=65536
JSON_CAP=65536
CODES=frozenset(("H_SCHEMA","H_ADMISSION","H_IDENTITY","H_CAPTURE","H_SERIALIZE","H_RAW_CAP",
    "H_COMPRESSED_CAP","H_CAPACITY","H_IO","H_CHECK_COUNT","H_AUTH","H_CLOSEOUT","H_REENTRY","H_RECEIPT_IO"))
class HookStopped(RuntimeError):
    def __init__(self,code):
        self.code=code if type(code) is str and code in CODES else "H_SCHEMA"
        super().__init__(self.code)
def need(ok,code="H_SCHEMA"):
    if not ok: raise HookStopped(code)
def sha(raw): return hashlib.sha256(raw).hexdigest()
def encode(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def compact(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def ids_ok(values,maximum):
    return type(values) in (tuple,list) and len(values)<=maximum and len(set(values))==len(values) and all(
        type(x) is str and 0<len(x)<=256 and all(32<=ord(c)<127 for c in x) for x in values)
def differences(before,after,prefix=""):
    result=[]
    if isinstance(before,dict) and isinstance(after,dict):
        for key in sorted(set(before)|set(after)):
            path=prefix+"/"+key
            if key not in before: result.append({"path":path,"change":"added","after":after[key]})
            elif key not in after: result.append({"path":path,"change":"removed","before":before[key]})
            else: result.extend(differences(before[key],after[key],path))
    elif before!=after: result.append({"path":prefix,"change":"changed","before":before,"after":after})
    return result

class DispatchLatch:
    """Shared by controller and every forward/derivative gate; no reset API."""
    def __init__(self,schedule_ids):
        need(ids_ok(schedule_ids,180))
        self.schedule=tuple(schedule_ids); self.cursor=0; self._state="PENDING"
        self.primary_code=None; self.secondary_codes=[]
    @property
    def terminal(self): return self._state=="TERMINAL"
    @property
    def remaining(self): return self.schedule[self.cursor:]
    def require_dispatch(self):
        if self._state!="ADMITTED" or self.primary_code is not None:
            self.trip("H_REENTRY")
            raise HookStopped(self.primary_code)
    def consume(self,cell_id):
        self.require_dispatch()
        if self.cursor>=len(self.schedule) or cell_id!=self.schedule[self.cursor]:
            self.trip("H_SCHEMA");raise HookStopped("H_SCHEMA")
        self.cursor+=1
    def trip(self,code):
        code=code if type(code) is str and code in CODES else "H_SCHEMA"
        self._state="TERMINAL"
        if self.primary_code is None: self.primary_code=code
        elif code!=self.primary_code and code not in self.secondary_codes and len(self.secondary_codes)<8:
            self.secondary_codes.append(code)
    def _admit(self):
        need(self._state=="PENDING" and self.primary_code is None,"H_REENTRY")
        self._state="ADMITTED"
    def _finish(self):
        self.require_dispatch();need(not self.remaining,"H_CLOSEOUT");self._state="COMPLETE"

class FileIO:
    """Default fixture IO. Production supplies the same API through its EvidenceWriter."""
    def __init__(self,root,create=True):
        self.root=Path(root).resolve()
        if create:
            self.root.mkdir(parents=True,exist_ok=False)
            (self.root/"hook_evidence").mkdir()
    def _path(self,name):
        need(type(name) is str and name.startswith("hook_evidence/") and "\\" not in name)
        p=(self.root/name).resolve()
        need(p.parent==self.root/"hook_evidence" and p.name not in ("",".",".."))
        return p
    def write(self,name,data,append=False,fault=False):
        path=self._path(name)
        with path.open("ab" if append else "xb") as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
    def read(self,name): return self._path(name).read_bytes()
    def hook_files(self):
        directory=self.root/"hook_evidence"
        return {p.relative_to(self.root).as_posix():p.stat().st_size for p in directory.rglob("*") if p.is_file()}
    def total_bytes(self): return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

def allocation(used_hook,used_total,added,checks_remaining,keep_index=True,fault=False):
    need(all(type(x) is int and x>=0 for x in (used_hook,used_total,added,checks_remaining)))
    reserve=0 if fault else FAULT_RESERVE+checks_remaining*CHECK_BYTES+(INDEX_RESERVE if keep_index else 0)
    need(used_hook+added+reserve<=HOOK_CAP and used_total+added+reserve<=AGGREGATE_CAP,"H_CAPACITY")
    return used_hook+added+reserve

class Recorder:
    """Admit complete normal data; permanently stop before serializing any fault details."""
    def __init__(self,output_root,latch,expected_check_labels,io=None):
        need(isinstance(latch,DispatchLatch) and ids_ok(expected_check_labels,MAX_CHECKS) and len(expected_check_labels)==MAX_CHECKS)
        self.latch=latch; self.labels=tuple(expected_check_labels)
        self.io=io if io is not None else FileIO(output_root)
        self.files={};self.checks=0;self.check_attempts=0;self.reference=None
        self.primary_exception=None;self.fault_attempted=False;self.receipt_failed=False
        self.diagnostic_complete=False;self.refs=None
        self.normal_index_bound_bytes=None
        self.completed_index_sha=None
    def _index_value(self,records,refs,cursor):
        return {"schema":"fresh_hook_index_v1","complete":True,"fault":False,"schedule":list(self.latch.schedule),
            "consumed_prefix":cursor,"check_labels":list(self.labels),"checks":MAX_CHECKS,
            "files":records,"setup_values":refs,"scientific_verdict":"NOT_SUPPLIED"}
    def _packet(self,name,value):
        try: raw=compact(value)
        except BaseException as error:
            if self.primary_exception is None: self.primary_exception=error
            raise HookStopped("H_SERIALIZE") from None
        need(len(raw)<=RAW_CAP,"H_RAW_CAP")
        chunks=[raw[i:i+CHUNK] for i in range(0,len(raw),CHUNK)]
        records=[];files=[]
        for i,piece in enumerate(chunks):
            data=zlib.compress(piece);need(len(data)<=FILE_CAP,"H_COMPRESSED_CAP")
            path="hook_evidence/%s_%03d.zlib"%(name,i)
            files.append((path,data))
            records.append({"path":path,"raw_bytes":len(piece),"raw_sha256":sha(piece),
                "compressed_bytes":len(data),"compressed_sha256":sha(data)})
        manifest={"complete":True,"raw_bytes":len(raw),"raw_sha256":sha(raw),"chunks":records,
            "encoding":"UTF8 compact JSON; concatenated zlib-compressed raw chunks"}
        path="hook_evidence/"+name+".json";data=encode(manifest);need(len(data)<=JSON_CAP,"H_CAPACITY")
        files.append((path,data))
        return files,{"manifest":path,"raw_bytes":len(raw),"raw_sha256":sha(raw)}
    def _write(self,files,checks_remaining,keep_index=True,diagnostic=False,fault=False,append=False):
        known=self.io.hook_files()
        for path,data in files:
            need(type(data) is bytes and len(data)+(known.get(path,0) if append else 0)<=FILE_CAP,"H_COMPRESSED_CAP")
            need(append or path not in known,"H_IO")
        allocation(sum(known.values()),self.io.total_bytes(),sum(len(data) for _,data in files),
            0 if diagnostic else checks_remaining,False if diagnostic else keep_index,fault)
        for path,data in files:
            try:
                previous=b""
                if append and path in known:
                    previous=self.io.read(path)
                    record=self.files.get(path)
                    need(record is not None and len(previous)==record["bytes"] and sha(previous)==record["sha256"],"H_AUTH")
                self.io.write(path,data,append=append,fault=fault)
                actual=self.io.read(path)
            except BaseException as error:
                if self.primary_exception is None: self.primary_exception=error
                raise HookStopped("H_IO") from None
            if not append: need(actual==data,"H_IO")
            else: need(actual==previous+data,"H_IO")
            self.files[path]={"path":path,"bytes":len(actual),"sha256":sha(actual)}
    def _fault(self,code,details=None,original=None):
        # The latch is set before JSON, codec, filesystem scans or any user IO callback.
        self.latch.trip(code)
        if self.primary_exception is None: self.primary_exception=original
        if self.fault_attempted: return
        self.fault_attempted=True;reference=None
        if details is not None:
            try:
                files,reference=self._packet("failure_%03d"%self.check_attempts,details)
                self._write(files,0,diagnostic=True)
                self.diagnostic_complete=True
            except BaseException as error:
                self.latch.trip(error.code if type(error) is HookStopped else "H_SERIALIZE")
                reference=None
        receipt={"schema":"fresh_hook_terminal_v1","status":"INCONCLUSIVE","attempt_complete":False,
            "primary_code":self.latch.primary_code,"secondary_codes":self.latch.secondary_codes,
            "diagnostic_complete":self.diagnostic_complete,"full_details_available":self.diagnostic_complete,
            "details_reference":reference,"cursor":self.latch.cursor,"remaining_ids":list(self.latch.remaining),
            "schedule_sha256":sha(encode(self.latch.schedule)),"check_attempts":self.check_attempts,
            "completed_checks":self.checks,"existing_and_partial_bytes_preserved":True,
            "exception_text_serialized":False,"fault_receipt_retry_authorized":False}
        try:
            data=encode(receipt);need(len(data)<=FAULT_RESERVE,"H_CAPACITY")
            self._write([("hook_evidence/fault.json",data)],0,keep_index=False,fault=True)
        except BaseException:
            self.receipt_failed=True;self.latch.trip("H_RECEIPT_IO")
    def fail(self,code,original=None):
        self._fault(code,original=original)
        raise HookStopped(self.latch.primary_code) from None
    def status(self):
        return {"latch_state":self.latch._state,"index_sha256":self.completed_index_sha,
            "terminal":self.latch.terminal,"primary_code":self.latch.primary_code,
            "secondary_codes":list(self.latch.secondary_codes),"receipt_failed":self.receipt_failed,
            "diagnostic_complete":self.diagnostic_complete,"remaining_ids":list(self.latch.remaining),
            "exception_text_serialized":False,"permits_pass":self.latch._state=="COMPLETE"}
    def admit(self,before,reference,setup_changes,source_lock,setup_receipt):
        try:
            need(self.latch._state=="PENDING" and not self.io.hook_files(),"H_REENTRY")
            need(setup_receipt.get("forward_calls")==0,"H_ADMISSION")
            need(source_lock.get("passed") is True and source_lock.get("before_materialization") is True,"H_ADMISSION")
            need(setup_changes["changes"]==differences(before,reference),"H_ADMISSION")
            files=[];refs={}
            for name,value in (("setup_before",before),("reference",reference),("setup_changes",setup_changes)):
                packet,refs[name]=self._packet(name,value);files.extend(packet)
            source_bytes=encode(source_lock);need(len(source_bytes)<=JSON_CAP,"H_CAPACITY")
            receipt={"schema":"fresh_hook_setup_v1","forward_calls":0,"complete":True,"values":refs,
                "source_lock_sha256":sha(source_bytes),"declared_setup_receipt":setup_receipt,
                "reserved_checks":MAX_CHECKS,"check_record_bytes":CHECK_BYTES,
                "fault_reserve_bytes":FAULT_RESERVE,"index_reserve_bytes":INDEX_RESERVE}
            receipt_bytes=encode(receipt);need(len(receipt_bytes)<=JSON_CAP,"H_CAPACITY")
            files.extend((("hook_evidence/source_lock.json",source_bytes),("hook_evidence/setup_receipt.json",receipt_bytes)))
            preview=[{"path":path,"bytes":len(data),"sha256":sha(data)} for path,data in files]
            preview.append({"path":"hook_evidence/checks.jsonl","bytes":MAX_CHECKS*CHECK_BYTES,"sha256":"0"*64})
            self.normal_index_bound_bytes=len(encode(self._index_value(preview,refs,len(self.latch.schedule))))
            need(self.normal_index_bound_bytes<=INDEX_RESERVE,"H_CAPACITY")
            self._write(files,MAX_CHECKS)
            self.reference=reference;self.refs=refs
            self.latch._admit()
            return receipt
        except BaseException as error:
            code=error.code if type(error) is HookStopped else "H_ADMISSION"
            self._fault(code,original=error)
            raise HookStopped(self.latch.primary_code) from None
    def inspect(self,provider,label):
        if self.latch.terminal: raise HookStopped(self.latch.primary_code) from None
        try:
            self.latch.require_dispatch()
            self.check_attempts+=1
            need(self.check_attempts<=MAX_CHECKS,"H_CHECK_COUNT")
            need(label==self.labels[self.check_attempts-1],"H_SCHEMA")
            inspection=provider()
            changes=differences(self.reference,inspection["current"])
            if changes:
                self._fault("H_IDENTITY",changes)
                raise HookStopped("H_IDENTITY")
            need(inspection["matches"] is True and inspection["changes"]==[],"H_IDENTITY")
            event={"check":self.check_attempts,"label":label,"matches":True,"complete_evidence":True,
                "reference_sha256":self.refs["reference"]["raw_sha256"],"changes":[]}
            raw=encode(event);need(len(raw)<=CHECK_BYTES,"H_CAPACITY")
            self._write([("hook_evidence/checks.jsonl",raw)],MAX_CHECKS-self.check_attempts,append=True)
            self.checks+=1
            return True
        except BaseException as error:
            code=error.code if type(error) is HookStopped else "H_CAPTURE"
            self._fault(code,original=error)
            raise HookStopped(self.latch.primary_code) from None
    def finish(self):
        if self.latch.terminal: raise HookStopped(self.latch.primary_code) from None
        try:
            self.latch.require_dispatch()
            need(self.checks==MAX_CHECKS and not self.latch.remaining,"H_CLOSEOUT")
            actual=self.io.hook_files()
            need(set(actual)==set(self.files),"H_AUTH")
            for path,record in self.files.items():
                raw=self.io.read(path)
                need(len(raw)==record["bytes"] and sha(raw)==record["sha256"],"H_AUTH")
            index=self._index_value(list(self.files.values()),self.refs,self.latch.cursor)
            raw=encode(index);need(len(raw)<=INDEX_RESERVE,"H_CAPACITY")
            self._write([("hook_evidence/index.json",raw)],0,keep_index=False)
            self.latch._finish()
            self.completed_index_sha=sha(raw)
            return {"hook_status":"COMPLETE","permits_pass":True,"index_sha256":sha(raw)}
        except BaseException as error:
            code=error.code if type(error) is HookStopped else "H_CLOSEOUT"
            self._fault(code,original=error)
            raise HookStopped(self.latch.primary_code) from None

def _read_value(io,ref,index,name):
    need(ref["manifest"]=="hook_evidence/"+name+".json","H_AUTH")
    need(ref["manifest"] in index,"H_AUTH")
    raw_manifest=io.read(ref["manifest"]);need(len(raw_manifest)<=JSON_CAP,"H_AUTH")
    manifest=json.loads(raw_manifest)
    need(manifest["complete"] is True and manifest["raw_sha256"]==ref["raw_sha256"] and
         manifest["raw_bytes"]==ref["raw_bytes"] and 0<manifest["raw_bytes"]<=RAW_CAP,"H_AUTH")
    entries=manifest["chunks"];need(0<len(entries)<=64,"H_AUTH")
    chunks=[]
    for i,entry in enumerate(entries):
        need(entry["path"]=="hook_evidence/%s_%03d.zlib"%(name,i),"H_AUTH")
        need(entry["path"] in index and 0<entry["raw_bytes"]<=CHUNK,"H_AUTH")
        need(i==len(entries)-1 or entry["raw_bytes"]==CHUNK,"H_AUTH")
        data=io.read(entry["path"])
        need(len(data)==entry["compressed_bytes"] and sha(data)==entry["compressed_sha256"],"H_AUTH")
        decoder=zlib.decompressobj()
        raw=decoder.decompress(data,entry["raw_bytes"]+1)
        need(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,"H_AUTH")
        need(len(raw)==entry["raw_bytes"] and sha(raw)==entry["raw_sha256"],"H_AUTH")
        chunks.append(raw)
    raw=b"".join(chunks)
    need(len(raw)==manifest["raw_bytes"] and sha(raw)==manifest["raw_sha256"],"H_AUTH")
    value=json.loads(raw);need(compact(value)==raw,"H_AUTH")
    return value,{ref["manifest"]}|{entry["path"] for entry in entries}

def judge(root,expected_labels,expected_schedule,expected_source_lock,controller_status,io=None):
    """Reconstruct bytes AND require externally bound controller closeout; not a scientific PASS."""
    try:
        need(ids_ok(expected_labels,MAX_CHECKS) and len(expected_labels)==MAX_CHECKS and ids_ok(expected_schedule,180))
        io=io if io is not None else FileIO(root,create=False)
        files=io.hook_files()
        need("hook_evidence/fault.json" not in files and "hook_evidence/index.json" in files,"H_CLOSEOUT")
        need(all(0<=n<=FILE_CAP for n in files.values()) and sum(files.values())<=HOOK_CAP and io.total_bytes()<=AGGREGATE_CAP,"H_CAPACITY")
        raw=io.read("hook_evidence/index.json");need(len(raw)<=INDEX_RESERVE,"H_AUTH")
        need(controller_status=={"latch_state":"COMPLETE","index_sha256":sha(raw),
            "terminal":False,"primary_code":None,"secondary_codes":[],"receipt_failed":False,
            "diagnostic_complete":False,"remaining_ids":[],"exception_text_serialized":False,
            "permits_pass":True},"H_CLOSEOUT")
        index=json.loads(raw)
        need(index["schema"]=="fresh_hook_index_v1" and index["complete"] is True and index["fault"] is False,"H_AUTH")
        need(index["schedule"]==list(expected_schedule) and index["consumed_prefix"]==len(expected_schedule) and
             index["check_labels"]==list(expected_labels) and index["checks"]==MAX_CHECKS,"H_AUTH")
        records=index["files"];mapping={r["path"]:r for r in records}
        need(len(mapping)==len(records) and set(files)==set(mapping)|{"hook_evidence/index.json"},"H_AUTH")
        for path,record in mapping.items():
            data=io.read(path);need(len(data)==record["bytes"] and sha(data)==record["sha256"],"H_AUTH")
        source_bytes=io.read("hook_evidence/source_lock.json");need(len(source_bytes)<=JSON_CAP,"H_AUTH")
        source=json.loads(source_bytes)
        need(source==expected_source_lock,"H_AUTH")
        setup_bytes=io.read("hook_evidence/setup_receipt.json");need(len(setup_bytes)<=JSON_CAP,"H_AUTH")
        setup=json.loads(setup_bytes)
        need(setup["schema"]=="fresh_hook_setup_v1" and setup["complete"] is True and setup["forward_calls"]==0 and
             setup["source_lock_sha256"]==sha(io.read("hook_evidence/source_lock.json")) and
             setup["values"]==index["setup_values"] and setup["reserved_checks"]==MAX_CHECKS and
             setup["check_record_bytes"]==CHECK_BYTES and setup["fault_reserve_bytes"]==FAULT_RESERVE and
             setup["index_reserve_bytes"]==INDEX_RESERVE,"H_AUTH")
        values={};used={"hook_evidence/source_lock.json","hook_evidence/setup_receipt.json","hook_evidence/checks.jsonl"}
        for name,ref in setup["values"].items():
            value,paths=_read_value(io,ref,mapping,name);values[name]=value;used.update(paths)
        need(used==set(mapping),"H_AUTH")
        need(set(values)=={"setup_before","reference","setup_changes"} and
             differences(values["setup_before"],values["reference"])==values["setup_changes"]["changes"],"H_AUTH")
        lines=io.read("hook_evidence/checks.jsonl").splitlines()
        need(all(len(line)+1<=CHECK_BYTES for line in lines),"H_AUTH")
        checks=[json.loads(line) for line in lines]
        need(len(checks)==MAX_CHECKS,"H_AUTH")
        for i,record in enumerate(checks):
            need(record["check"]==i+1 and record["label"]==expected_labels[i] and record["matches"] is True and
                 record["complete_evidence"] is True and record["changes"]==[] and
                 record["reference_sha256"]==setup["values"]["reference"]["raw_sha256"],"H_AUTH")
        return {"hook_status":"COMPLETE","permits_pass":True,"checks":MAX_CHECKS,"scientific_verdict":"NOT_SUPPLIED"}
    except BaseException as error:
        return {"hook_status":"INCONCLUSIVE","permits_pass":False,
            "code":error.code if type(error) is HookStopped else "H_AUTH","exception_text_serialized":False,
            "scientific_verdict":"NOT_SUPPLIED"}
