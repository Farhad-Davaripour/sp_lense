"""Finite pre-Trace failure evidence; cannot permit continuation or success."""
import json
from support import HERE,sha,require
STAGES=("CORE_IMPORT","AUTHORITY","PUBLIC_INPUT","NATIVE_RESERVE","WRITER_SOURCE_ADMISSION","ALLOWLIST","TRACE")
CODES=("SOURCE_PATH_CONFLICT","SOURCE_PIN_MISMATCH","UNDECLARED_SOURCE","STARTUP_EXCEPTION")
MESSAGES={"same path cannot name distinct compiled source":"SOURCE_PATH_CONFLICT",
          "locked committed source bytes":"SOURCE_PIN_MISMATCH","undeclared source access":"UNDECLARED_SOURCE"}
MAX_BYTES=8192
def capture(error,stage,execution,publisher):
    argument=error.args[0] if len(getattr(error,"args",()))==1 and type(error.args[0]) is str else None
    code=MESSAGES.get(argument,"STARTUP_EXCEPTION")
    category="VALUE_ERROR" if isinstance(error,ValueError) else "OS_ERROR" if isinstance(error,OSError) else "OTHER_EXCEPTION"
    value={"schema":"finite_pretrace_startup_failure.v1","execution":execution,
        "source_sha256":sha((HERE/"SOURCE_FREEZE.json").read_bytes()),"stage":stage if stage in STAGES else "CORE_IMPORT",
        "code":code,"category":category,"before_trace":True,"loader_dispatched":False,
        "scientific_pass":False,"classification":"INCONCLUSIVE_DIAGNOSTIC"}
    status={"finding":value,"pointer":None,"publication_failed":False}
    try:
        raw=(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
        require(len(raw)<=MAX_BYTES,"finite startup primary cap inside existing native closeout")
        status["pointer"]=publisher("STARTUP_FAILURE.json",raw,raw=True,critical=True)
        require(status["pointer"]["bytes"]==len(raw) and status["pointer"]["sha256"]==sha(raw),"startup primary native acknowledgement")
    except BaseException:status["publication_failed"]=True
    return status
def verify_saved(control,execution,status,inner_status):
    if status is None:
        require(inner_status is None,"same absent closed startup status")
        return None
    require(status==inner_status,"closed worker startup primary copies")
    finding=status["finding"]
    require(finding["schema"]=="finite_pretrace_startup_failure.v1" and finding["execution"]==execution and
        finding["source_sha256"]==sha((HERE/"SOURCE_FREEZE.json").read_bytes()) and finding["stage"] in STAGES and
        finding["code"] in CODES and finding["category"] in ("VALUE_ERROR","OS_ERROR","OTHER_EXCEPTION") and
        finding["before_trace"] is True and finding["loader_dispatched"] is False and finding["scientific_pass"] is False and
        finding["classification"]=="INCONCLUSIVE_DIAGNOSTIC","finite exact startup finding")
    result={"finding":finding,"native_verified":False,"scientific_pass":False}
    if status["publication_failed"] or status["pointer"] is None:return result
    path=control/"STARTUP_FAILURE.json";raw=path.read_bytes();pointer=status["pointer"]
    require(len(raw)==pointer["bytes"]<=MAX_BYTES and sha(raw)==pointer["sha256"] and
        pointer["path"]==path.relative_to(HERE).as_posix() and json.loads(raw)==finding,"authenticated startup native primary")
    result["native_verified"]=True
    return result
