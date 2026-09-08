"""Finite observational code metadata, never parameters or exception text."""
import json
import sys
import types
from pathlib import Path
from support import HERE,ROOT,sha,require
FILE_CAPS={"CONSTRUCTOR_RESERVATION.json":4096,"CONSTRUCTOR_BEFORE.json":16384,
           "CONSTRUCTOR_AFTER.json":16384,"CONSTRUCTOR_TERMINAL.json":28672}
PREDICATES=("CALLABLE_FUNCTION_TYPE","CALLABLE_CODE_IDENTITY","CALLABLE_SOURCE_PATH")

def source_identity():
    return {"source_sha256":sha((HERE/"SOURCE_FREEZE.json").read_bytes()),
            "public_input_sha256":sha((HERE/"PUBLIC_INPUT.json").read_bytes())}

def loaded_snapshot():
    # Already imported by the unchanged owned loader; no new model/module import.
    from helper_binding import ensure_package
    package=ensure_package();source=package["source_contract"];admission=package["admission"]
    source.authenticate(ROOT)
    module=sys.modules.get(package["compat"].HF_MODULE)
    require(type(module) is types.ModuleType,"constructor operand existing HF module")
    original=getattr(module,"Qwen3_5GatedDeltaNet")
    path=(ROOT/source.PINS["hf"][0]).resolve()
    expected=source.code_catalog(ROOT,"hf")["Qwen3_5GatedDeltaNet.__init__"]
    fn=original.__init__;conditions=admission.code_conditions(fn,expected,path)
    code=fn.__code__ if type(fn) is types.FunctionType else None
    return {"conditions":conditions,"class_identity":id(original),"function_identity":id(fn),
        "code_identity":id(code) if code else None,"globals_identity_matches":bool(code and fn.__globals__ is vars(module)),
        "defaults_none":bool(code and fn.__defaults__ is None),"kwdefaults_none":bool(code and fn.__kwdefaults__ is None),
        "closure_none":bool(code and fn.__closure__ is None),"optimize":sys.flags.optimize,
        "bytecode_sha256":sha(code.co_code) if code else None,"expected_bytecode_sha256":sha(expected.co_code),
        "expected_source_sha256":source.PINS["hf"][1],"expected_line":expected.co_firstlineno,
        "observed_line":code.co_firstlineno if code else None,"observational_only":True}

class Operands:
    def __init__(self,execution,publish,stopper,source=None):
        self.execution=json.loads(json.dumps(execution));self.source=source or source_identity()
        self.publish=publish;self.stopper=stopper;self.pointers={};self.primary=None;self.io_failed=False;self.finished=False
        self._write("CONSTRUCTOR_RESERVATION.json",{"reserved_bytes":65536,"file_caps":FILE_CAPS,"before_load":True})
    def _write(self,name,value):
        packet={"schema":"constructor_operands.v1","execution":self.execution,"source":self.source,**value}
        raw=(json.dumps(packet,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
        require(len(raw)<=FILE_CAPS[name],"constructor operand receipt cap")
        pointer=self.publish(name,raw,raw=True,critical=True)
        require(pointer["bytes"]==len(raw) and pointer["sha256"]==sha(raw),"constructor operand native acknowledgement")
        self.pointers[name]=pointer
    def fail(self,stage,error=None):
        if self.primary is None:
            code=getattr(error,"code",None)
            self.primary={"stage":stage if stage in ("BEFORE","AFTER","LOAD_OR_ADMISSION","PUBLICATION","OUTER_CLOSEOUT") else "UNKNOWN",
                "predicate":code if code in PREDICATES else "OTHER_OR_UNKNOWN",
                "category":"OS_ERROR" if isinstance(error,OSError) else "OTHER_EXCEPTION"}
        try:self.stopper("CONSTRUCTOR_OPERANDS_FAILURE")
        except BaseException:pass
    def snapshot(self,phase,provider):
        require(not self.finished and phase in ("BEFORE","AFTER"),"constructor phase")
        require((phase=="BEFORE" and len(self.pointers)==1) or (phase=="AFTER" and "CONSTRUCTOR_BEFORE.json" in self.pointers),"constructor phase order")
        try:self._write("CONSTRUCTOR_"+phase+".json",{"phase":phase,"operands":provider()})
        except BaseException as error:
            self.io_failed=True;self.fail(phase,error);raise
    def finish(self):
        if self.finished:return self.status()
        self.finished=True
        try:self._write("CONSTRUCTOR_TERMINAL.json",{"phase":"TERMINAL","primary":self.primary,"io_failed":self.io_failed,"prior_pointers":dict(self.pointers)})
        except BaseException as error:self.io_failed=True;self.fail("PUBLICATION",error)
        return self.status()
    def status(self):
        return {"execution":self.execution,"source":self.source,"primary":self.primary,"io_failed":self.io_failed,
            "complete":not self.io_failed and set(self.pointers)==set(FILE_CAPS),"finished":self.finished,
            "pointers":dict(self.pointers),"scientific_pass":False}

def reserve_operands(writer,admitted,publish):
    require(not hasattr(writer,"constructor_operands"),"one constructor operand reservation")
    def stop(code):
        context=getattr(writer,"loader_diagnostics",None)
        if context is not None and context.latch is not None:context.latch.stop(code)
    writer.constructor_operands=Operands(admitted["execution"],publish,stop)

def terminal_status(writer):
    recorder=getattr(writer,"constructor_operands",None)
    if recorder is None:return {"complete":False,"scientific_pass":False,"primary":{"stage":"UNKNOWN"}}
    recorder.finish()
    return recorder.status()
