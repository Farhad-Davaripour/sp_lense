"""Pure root authority, exclusive attempt and saved-role boundary. No backend imports."""
import ast
import hashlib
import json
import math
import os
import stat
import time
from pathlib import Path

MAIN=Path(__file__).resolve().parent
ATTEMPT="fresh_confirmation_real_attempt_002"
OUTPUT="real_evidence/"+ATTEMPT
MIB=1024**2


class Denied(ValueError): pass


def need(value,code):
    if not value: raise Denied(code)


def digest(raw): return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()


def strict(raw):
    def pairs(items):
        result={}
        for key,value in items:
            need(key not in result,"D_DUPLICATE_JSON_KEY");result[key]=value
        return result
    def constant(value): raise Denied("D_NONFINITE_JSON")
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)


def checked_path(root,relative):
    root=Path(root).absolute()
    need(type(relative) is str and relative and "\\" not in relative,"D_PATH")
    parts=relative.split("/")
    blocked={"con","prn","aux","nul"}|{prefix+str(n) for prefix in ("com","lpt") for n in range(1,10)}
    need(all(p not in ("",".","..") and not p.endswith((" ",".")) and
        not any(c in p for c in '\x00<>:"|?*') and p.split(".")[0].casefold() not in blocked for p in parts),"D_PATH")
    path=root.joinpath(*parts)
    need(path.resolve(strict=False).is_relative_to(root.resolve(strict=False)),"D_PATH_ESCAPE")
    for parent in (root,*[root.joinpath(*parts[:i]) for i in range(1,len(parts)+1)]):
        if parent.exists():
            info=parent.lstat()
            need(not stat.S_ISLNK(info.st_mode) and not (getattr(info,"st_file_attributes",0)&0x400),"D_LINK")
            if parent.is_file(): need(info.st_nlink==1,"D_HARDLINK")
    return path


def write_exclusive(path,raw,root):
    need(type(raw) is bytes and len(raw)<=5*MIB,"D_FILE_CAP")
    from support import bounds
    usage=bounds()
    path=Path(path).absolute()
    if path.is_relative_to((MAIN/"test_evidence").absolute()):
        need(usage["test_evidence_bytes"]+len(raw)<=8*MIB,"D_TEST_CAP")
    elif path.is_relative_to((MAIN/"real_evidence").absolute()):
        need(usage["evidence_bytes"]+len(raw)<=288*MIB,"D_REAL_CAP")
    else:
        need(usage["preparation_bytes"]+len(raw)<=32*MIB,"D_PREPARATION_CAP")
    target=checked_path(root,Path(path).relative_to(root).as_posix())
    target.parent.mkdir(parents=True,exist_ok=True)
    checked_path(root,target.relative_to(root).as_posix())
    with target.open("xb") as stream:
        need(stream.write(raw)==len(raw),"D_PARTIAL_RECEIPT")
        stream.flush();os.fsync(stream.fileno())
    need(target.read_bytes()==raw,"D_RECEIPT_ACK")
    return digest(raw)


def usage_value(tool_result):
    from support import SOURCES,SCIENCE,SCIENCE_COMMIT
    source=SOURCES.read(SCIENCE_COMMIT,SCIENCE+"usage_receipt.py").decode()
    namespace={"json":json,"math":math,"require":need,
        "BOOLEAN_FLAGS":("spendControlReached","rateLimitReached","isRateLimited","limitReached","exhausted")}
    for node in ast.parse(source).body:
        if isinstance(node,ast.FunctionDef):
            exec(compile(ast.get_source_segment(source,node),"pinned_usage:"+node.name,"exec"),namespace)
    return namespace["derive_standard_usage"](tool_result)


class Boundary:
    def __init__(self,root=MAIN,*,mock=False):
        self.root=Path(root).absolute();self.mock=mock
        if mock: need(self.root.resolve().is_relative_to((MAIN/"test_evidence").resolve()),"D_MOCK_SCOPE")
        else: need(self.root==MAIN,"D_PRODUCTION_ROOT")
        self.bindings=strict((MAIN/"BINDINGS.json").read_bytes())
        self.output=checked_path(self.root,OUTPUT)
        self.control=self.output/"control"

    def read(self,name):
        path=checked_path(self.root,name)
        need(path.stat().st_size<=5*MIB,"D_READ_FILE_CAP")
        return path.read_bytes()

    def control_path(self,name): return checked_path(self.root,OUTPUT+"/control/"+name)

    def read_control(self,name): return self.read(OUTPUT+"/control/"+name)

    def verify_chain(self,approved_lock_sha256):
        from support import check_freeze
        check_freeze()
        raw_lock=self.read("root_release/AUTHORITY_LOCK.json")
        need(type(approved_lock_sha256) is str and len(approved_lock_sha256)==64 and digest(raw_lock)==approved_lock_sha256,"D_CALLER_LOCK")
        lock=strict(raw_lock)
        need(set(lock)=={"schema","source_sha256","release_sha256","authorization_sha256"} and lock["schema"]=="real_root_authority_lock.v1","D_LOCK_SCHEMA")
        manifest=(MAIN/"SOURCE_FREEZE.json").read_bytes()
        release_raw=self.read("root_release/ROOT_RELEASE.json")
        auth_raw=self.read("root_release/AUTHORIZATION.json")
        need(lock["source_sha256"]==digest(manifest) and lock["release_sha256"]==digest(release_raw) and
            lock["authorization_sha256"]==digest(auth_raw),"D_CHAIN_HASH")
        release,auth=strict(release_raw),strict(auth_raw)
        required={"schema","source_sha256","binding_sha256","input_lock_sha256","runtime_spec_sha256","resource_contract_sha256",
            "execution_mode","permission_scope","attempt_id","output_relative","usage_sha256","production_authorized","source_input_hook_reviewed"}
        need(set(release)==required and release["schema"]=="real_root_release.v1","D_RELEASE_SCHEMA")
        need(set(auth)=={"schema","release_sha256","allow_execute","execution_mode","permission_scope","attempt_id","output_relative"}
            and auth["schema"]=="real_root_authorization.v1","D_AUTH_SCHEMA")
        scope="MODEL_FREE_SENTINEL_ONLY" if self.mock else "ROOT_REAL_SINGLE_ATTEMPT"
        need(auth["allow_execute"] is True and auth["release_sha256"]==digest(release_raw),"D_DISABLED_AUTHORITY")
        need(release["source_sha256"]==digest(manifest) and release["binding_sha256"]==digest((MAIN/"BINDINGS.json").read_bytes()),"D_SOURCE_BINDING")
        for field in ("input_lock_sha256","runtime_spec_sha256","resource_contract_sha256"):
            need(release[field]==self.bindings[field],"D_"+field.upper())
        need(release["runtime_spec_sha256"]==digest((MAIN/"RUNTIME_SPEC.json").read_bytes()),"D_RUNTIME_BYTES")
        from support import SOURCES
        reference=self.bindings["frozen_runtime_reference"]
        old_raw=SOURCES.read(reference["commit"],reference["path"])
        need(digest(old_raw)==reference["sha256"] and strict(old_raw)==strict((MAIN/"RUNTIME_SPEC.json").read_bytes()),"D_FROZEN_RUNTIME_SEMANTICS")
        need(release["source_input_hook_reviewed"] is True,"D_REVIEW_REQUIRED")
        need(release["production_authorized"] is (not self.mock),"D_ACTUAL_AUTHORITY_SCOPE")
        for document in (release,auth):
            need(document["execution_mode"]=="REAL_QWEN" and document["permission_scope"]==scope and
                document["attempt_id"]==ATTEMPT and document["output_relative"]==OUTPUT,"D_MODE_OUTPUT_BINDING")
        usage_raw=self.read("root_release/USAGE_RECEIPT.json")
        need(digest(usage_raw)==release["usage_sha256"],"D_USAGE_BYTES")
        usage=strict(usage_raw)
        need(set(usage)=={"observed_unix_seconds","tool_result"} and type(usage["observed_unix_seconds"]) in (float,int)
            and math.isfinite(usage["observed_unix_seconds"]),"D_USAGE_SCHEMA")
        derived=usage_value(usage["tool_result"])
        need(derived["used_percent"]<100,"D_USAGE_EXHAUSTED")
        if not self.mock:
            from admission import admit
            admitted=admit()
            need(admitted["binding"]["input_lock_sha256"]==release["input_lock_sha256"],"D_EXACT_INPUTS")
        return {"lock":lock,"release":release,"usage":usage,"derived_usage":derived,"approved_lock_sha256":approved_lock_sha256}

    def admit_once(self,approved_lock_sha256,*,now=None):
        need(now is None or self.mock,"D_REAL_CLOCK_OVERRIDE")
        chain=self.verify_chain(approved_lock_sha256)
        # A failed/partial write leaves this exclusive directory occupied forever.
        self.output.parent.mkdir(parents=True,exist_ok=True)
        checked_path(self.root,OUTPUT)
        # Freshness is measured after all potentially slow validation, at reservation.
        observed=time.time() if now is None else now
        need(0<=observed-chain["usage"]["observed_unix_seconds"]<=120,"D_STALE_USAGE")
        self.output.mkdir(exist_ok=False)
        self.control.mkdir(exist_ok=False)
        record={"schema":"exclusive_real_attempt_admission.v1","attempt_id":ATTEMPT,"output_relative":OUTPUT,
            "approved_lock_sha256":approved_lock_sha256,"source_sha256":chain["lock"]["source_sha256"],
            "release_sha256":chain["lock"]["release_sha256"],"authorization_sha256":chain["lock"]["authorization_sha256"],
            "usage_sha256":chain["release"]["usage_sha256"],"admitted_unix_seconds":observed,
            "derived_usage":chain["derived_usage"],"permission_scope":chain["release"]["permission_scope"],
            "observed_model_work":"NONE_AT_ADMISSION","maximum_attempts":1,"resume_allowed":False}
        raw=encoded(record)
        pin=write_exclusive(self.control/"ATTEMPT_ADMISSION.json",raw,self.root)
        return pin,self.reauthenticate(approved_lock_sha256,pin)

    def reauthenticate(self,approved_lock_sha256,admission_sha256):
        chain=self.verify_chain(approved_lock_sha256)
        raw=self.read_control("ATTEMPT_ADMISSION.json")
        need(digest(raw)==admission_sha256,"D_ADMISSION_BYTES")
        admission=strict(raw)
        need(admission["approved_lock_sha256"]==approved_lock_sha256 and admission["source_sha256"]==chain["lock"]["source_sha256"]
            and admission["release_sha256"]==chain["lock"]["release_sha256"] and admission["authorization_sha256"]==chain["lock"]["authorization_sha256"]
            and admission["usage_sha256"]==chain["release"]["usage_sha256"] and admission["derived_usage"]==chain["derived_usage"]
            and admission["output_relative"]==OUTPUT and admission["attempt_id"]==ATTEMPT
            and admission["permission_scope"]==chain["release"]["permission_scope"],"D_ADMITTED_IDENTITY")
        # Audit validates the age at admission, never demands a new mid-run receipt.
        need(0<=admission["admitted_unix_seconds"]-chain["usage"]["observed_unix_seconds"]<=120,"D_ADMITTED_USAGE_TIME")
        execution={"schema":"root_real_execution_identity.v1","mode":"REAL_QWEN","intended_backend":"PINNED_QWEN35_08B_CPU_FLOAT32",
            "production_authorized":not self.mock,"permission_scope":admission["permission_scope"],"attempt_id":ATTEMPT,"fixture_id":None,
            "output_relative":OUTPUT,"source_sha256":admission["source_sha256"],"release_sha256":admission["release_sha256"],
            "authorization_sha256":admission["authorization_sha256"],"authority_lock_sha256":approved_lock_sha256,
            "admission_sha256":admission_sha256,"input_lock_sha256":self.bindings["input_lock_sha256"],
            "real_model_work":None,"model_work_observation":"SEPARATE_MEASURED_LOADER_AND_TERMINAL_RECEIPTS"}
        return {"execution":execution,"release":chain["release"],"binding":self.bindings["input_binding"]}

    def role(self,role,approved,admission,*,pid=None):
        need(role in ("worker","writer","audit"),"D_ROLE")
        identity=self.reauthenticate(approved,admission)
        if role=="audit":
            need(self.control_path("WORKER_RESULT.json").is_file(),"D_MISSING_WORKER_TERMINAL")
            if self.control_path("LOADER_STARTED.json").exists():
                need(self.control_path("LOADER_TERMINAL.json").is_file(),"D_MISSING_LOADER_TERMINAL")
                terminal=strict(self.read_control("LOADER_TERMINAL.json"))
                need(terminal["execution"]==identity["execution"],"D_LOADER_TERMINAL_IDENTITY")
            worker=strict(self.read_control("WORKER_RESULT.json"))
            need(worker["terminal"]["execution"]==identity["execution"],"D_WORKER_TERMINAL_IDENTITY")
        if role=="worker":
            bootstrap=strict(self.read_control("owned/production_worker/BOOTSTRAP.json"))
            need(bootstrap["permission_received"] is True and bootstrap["execution"]==identity["execution"] and
                bootstrap["actual_pid"]==(os.getpid() if pid is None else pid),"D_OWNED_PERMISSION")
            need(pid is None or self.mock,"D_REAL_PID_OVERRIDE")
            write_exclusive(self.control/"WORKER_ENTRY.json",encoded({"execution":identity["execution"],
                "bootstrap_sha256":digest(self.read_control("owned/production_worker/BOOTSTRAP.json")),"observed_model_work":"NONE_BEFORE_IMPORTS"}),self.root)
        return identity

    def dispatch(self,approved,admission,callback,*,sentinel=False):
        identity=self.reauthenticate(approved,admission)
        need(sentinel is self.mock,"D_SENTINEL_REAL_SEPARATION")
        entry=strict(self.read_control("WORKER_ENTRY.json"))
        need(entry["execution"]==identity["execution"],"D_WORKER_ENTRY_IDENTITY")
        write_exclusive(self.control/"LOADER_STARTED.json",encoded({"execution":identity["execution"],
            "observed_model_work":"NONE_BEFORE_LOADER_DISPATCH","target":self.bindings["loader"]}),self.root)
        outcome={"execution":identity["execution"],"loader_returned":False,
            "observed_model_work":"SENTINEL_ONLY" if self.mock else "UNKNOWN_IF_PARTIAL_REAL_LOAD", "failure_code":None}
        try:
            result=callback(identity,self.bindings["loader"])
            outcome.update(loader_returned=True,observed_model_work="SENTINEL_ONLY" if self.mock else "PINNED_REAL_ADAPTER_RETURNED")
            return result
        except BaseException:
            outcome["failure_code"]="LOADER_BOUNDARY_FAILURE"
            raise
        finally: write_exclusive(self.control/"LOADER_TERMINAL.json",encoded(outcome),self.root)
