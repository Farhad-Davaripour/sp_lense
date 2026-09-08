"""Only private helper joins; existing scientific/numeric callbacks are unchanged."""
import importlib
import json
import sys
from pathlib import Path
from support import HERE,ROOT,require,sha,write_new
from helper_limits import FILE_CAPS,TOTAL_RESERVED
_PRIVATE={}
_PACKAGE=None

def source_lock():
    return {"schema":"first_forward_helper_source.v2",
        "source_sha256":sha((HERE/"SOURCE_FREEZE.json").read_bytes()),
        "package_binding_sha256":sha((HERE/"HELPER_PACKAGE_BINDING.json").read_bytes()),
        "pins_sha256":sha((HERE/"helper_pkg/SOURCE_PINS.json").read_bytes()),
        "selection_commit":"1081df521dd1cdda5be5d2daa4bee1feb74fbbeb",
        "compat_commit":"964f76340d3530e67a640e1e821dc5a84c541b23"}

def ensure_package():
    global _PACKAGE
    binding=json.loads((HERE/"HELPER_PACKAGE_BINDING.json").read_bytes())
    for name,record in binding["files"].items():
        raw=(HERE/name).read_bytes()
        require(len(raw)==record["bytes"] and sha(raw)==record["sha256"],"helper private source binding")
    names=("helper_pkg","helper_pkg.support","helper_pkg.source_contract","helper_pkg.admission","helper_pkg.compat",
           "helper_pkg.selection","helper_pkg.selection_reader","helper_pkg.fingerprint","helper_pkg.handoff","helper_pkg.saved_reader")
    if _PACKAGE is None:
        require(not any(name in sys.modules for name in names),"helper private namespace initially unoccupied")
        for name in names:
            module=importlib.import_module(name)
            suffix="__init__.py" if name=="helper_pkg" else name.split(".")[-1]+".py"
            require(Path(module.__file__).resolve()==(HERE/"helper_pkg"/suffix).resolve(),"helper private resolved module")
            _PRIVATE[name]=module
        _PACKAGE={name.split(".")[-1]:_PRIVATE[name] for name in names[1:]}
    require(all(sys.modules.get(name) is module for name,module in _PRIVATE.items()),"helper private module identity remains fixed")
    return _PACKAGE

def reservation_value(execution):
    return {"schema":"first_forward_helper_reservation.v2","execution":execution,"source_lock":source_lock(),
        "before_load":True,"file_caps":FILE_CAPS,"reserved_bytes":TOTAL_RESERVED,
        "inside_existing_other_closeout_bytes":3*1024**2//4,"original_total_closeout_bytes":8*1024**2,
        "real_authorization_granted":False}

def reserve_helpers(writer,admitted):
    require(not hasattr(writer,"helper_reservation"),"one helper reservation")
    value=reservation_value(admitted["execution"])
    pointer=write_new("HELPER_RESERVATION.json",value,critical=True)
    writer.helper_reservation={"value":value,"pointer":pointer}
    writer.helper_join=None
    writer.helper_outer_status=None
    return pointer

def _resolve_loaded(model,admitted,package):
    require(admitted["execution"]["mode"]=="REAL_QWEN" and admitted["execution"]["production_authorized"] is True,
            "mandatory live helper authority")
    selector=package["selection"];loaded={}
    for role,(module_name,class_name) in selector.TYPE_SOURCES.items():
        module=sys.modules.get(module_name)
        require(module is not None,"helper requires already loaded module")
        loaded[role]=getattr(module,class_name)
    hf_module=sys.modules[package["compat"].HF_MODULE]
    targets=package["compat"].Targets(hf_module,loaded["gdn"],hf_module.causal_conv1d_fn,
        hf_module.torch_chunk_gated_delta_rule,ROOT,mode="LIVE_EXISTING_MODULE")
    bindings=selector.TypeBindings(**loaded,mode="LIVE_EXISTING_MODULE")
    bindings.verify(targets)
    require(type(model) is loaded["bridge"],"actual transformer bridge instance")
    return targets,model.original_model,model,bindings

class Publisher:
    def write_new(self,name,raw):
        require(name in ("HELPER_SETUP.json","HELPER_TERMINAL.json") and type(raw) is bytes and len(raw)<=FILE_CAPS[name],
                "reserved native helper publication")
        pointer=write_new(name,raw,raw=True,critical=True)
        return {"name":name,"bytes":pointer["bytes"],"sha256":pointer["sha256"]}

class Join:
    def __init__(self,writer,admitted):
        self.writer=writer;self.execution=json.loads(json.dumps(admitted["execution"]))
        self.source=source_lock();self.session=None;self.admitted_raw=None;self.admission_pointer=None
        self.outer_pointer=None;self.outer_write_failed=False;self.first_code=None;self.closed=False
    def fail(self,code):
        if self.first_code is None:self.first_code=code
        context=getattr(self.writer,"loader_diagnostics",None)
        if context is not None and context.latch is not None:
            try:context.latch.stop("GDN_HELPER_INTEGRATION_FAILURE")
            except BaseException:pass # Original primary remains; incomplete finite status survives.
    def status(self):
        value={"schema":"first_forward_helper_outer.v2","execution":self.execution,"source_lock":self.source,
            "reservation_pointer":self.writer.helper_reservation["pointer"],"admission_pointer":self.admission_pointer,
            "session":self.session.status() if self.session else None,"first_code":self.first_code,
            "outer_write_failed":self.outer_write_failed,"scientific_pass":False}
        raw=json.dumps(value,sort_keys=True,separators=(",",":")).encode()
        if len(raw)>FILE_CAPS["HELPER_OUTER_STATUS.json"]:
            self.fail("HJ_STATUS_CAPACITY")
            value={"schema":"first_forward_helper_outer.v2","execution":self.execution,"source_lock":self.source,
                "first_code":self.first_code,"complete_evidence":False,"scientific_pass":False,"outer_write_failed":True}
        value["outer_pointer"]=self.outer_pointer
        return value

def begin(writer,model,guard,admitted):
    require(writer.helper_reservation["value"]==reservation_value(admitted["execution"]),"helper reservation before setup")
    require(writer.helper_join is None,"one helper integration")
    join=Join(writer,admitted);writer.helper_join=join
    try:
        package=ensure_package()
        targets,hf,bridge,bindings=_resolve_loaded(model,admitted,package)
        context=writer.loader_diagnostics
        require(context.legacy_parameters is not None and context.latch is not None,"legacy capture and original latch precede helpers")
        join.session=package["handoff"].Handoff(targets,hf,bridge,bindings,Publisher(),context.latch.stop,guard.restore,
            execution=admitted["execution"],source_lock=join.source)
        return join
    except BaseException:
        join.fail("HJ_LIVE_ADMISSION");raise

def setup(join,original_setup):
    try:
        result=join.session.setup(original_setup)
        packet=join.session.admission()
        require(join.admitted_raw is None and packet is not None,"one immutable helper setup packet")
        raw=json.dumps(packet,sort_keys=True,separators=(",",":")).encode()
        require(len(raw)<=FILE_CAPS["HELPER_ADMISSION.json"],"bounded immutable setup packet")
        join.admitted_raw=raw # Preserve once, before the native write can fail.
        join.admission_pointer=write_new("HELPER_ADMISSION.json",raw,raw=True,critical=True)
        return result
    except BaseException:
        join.fail("HJ_SETUP_OR_ADMISSION_WRITE");raise

def checkpoint(writer,original_check):
    join=writer.helper_join
    require(join is not None and join.first_code is None and join.admission_pointer is not None,"complete helper admission before first forward")
    try:return join.session.checkpoint(original_check)
    except BaseException:
        join.fail("HJ_PRE_FORWARD");raise

def close(writer,original_cleanup):
    join=writer.helper_join
    require(join is not None and join.session is not None,"created helper session for cleanup")
    try:return join.session.close(original_cleanup)
    except BaseException:
        join.fail("HJ_CLEANUP");raise

def abort_load(writer,error):
    """Best-effort owned helper closeout; never replace the original exception."""
    join=getattr(writer,"helper_join",None)
    if join is None:return
    join.fail("HJ_LOAD_OR_CONSTRUCTOR")
    session=join.session
    if session is None or session.closed:return
    try:
        session._fail("ORIGINAL_SETUP",error)
        def restore_only(restore):
            restore()
        session.close(restore_only)
    except BaseException:
        pass # Recorded session/first_code remain incomplete; original caller restores too.

def record_outer(writer):
    """Called once in unconditional core finally, before its own terminal write."""
    join=getattr(writer,"helper_join",None)
    if join is None:
        value={"schema":"first_forward_helper_outer.v2","complete_evidence":False,
               "first_code":"HJ_NOT_CREATED","scientific_pass":False}
        writer.helper_outer_status=value
        return value
    try:
        require(join.outer_pointer is None and not join.outer_write_failed,"one helper outer publication")
        value=join.status();value.pop("outer_pointer",None)
        raw=json.dumps(value,sort_keys=True,separators=(",",":")).encode()
        require(len(raw)<=FILE_CAPS["HELPER_OUTER_STATUS.json"],"bounded helper outer copy")
        join.outer_pointer=write_new("HELPER_OUTER_STATUS.json",raw,raw=True,critical=True)
    except BaseException:
        join.outer_write_failed=True;join.fail("HJ_OUTER_WRITE")
    writer.helper_outer_status=join.status()
    return writer.helper_outer_status
