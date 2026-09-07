"""New recorder/writer bridge; binds only the committed checked hook component."""
import json
from pathlib import Path
from support import HERE, SOURCES, SCIENCE, SCIENCE_COMMIT, require, sha

_COMPONENT = None

def component():
    global _COMPONENT
    lock = json.loads((HERE/"HOOK_BINDING.json").read_bytes())
    require(lock["status"] == "COMPLETE", "committed checked hook successor required")
    SOURCES.read(lock["commit"],lock["path"])
    tests = json.loads(SOURCES.read(lock["commit"],lock["test_receipt_path"]))
    closeout = json.loads(SOURCES.read(lock["commit"],lock["closeout_path"]))
    require(tests["status"] == "PASS" and len(tests["groups"]) == 6 and all(g["status"] == "PASS" for g in tests["groups"])
        and tests["elapsed_seconds"] <= 60 and tests["model_calls"] == tests["tokenizer_calls"] == 0
        and closeout["status"] == "PASS_MODEL_FREE_COMPONENT" and closeout["batch_process_exit_code"] == 0
        and closeout["source_freeze_sha256"] == lock["source_freeze_sha256"], "complete committed checked hook dependency")
    if _COMPONENT is None:
        _COMPONENT = SOURCES.load("production_bound_hook_evidence",lock["commit"],lock["path"])
    return _COMPONENT


class LatchView:
    def __init__(self, raw):
        self.raw = raw
        self.adapter_primary_code = None
        self.recorder = None

    @property
    def failed(self):
        return self.raw.terminal

    @property
    def remaining(self):
        return list(self.raw.remaining)

    def admit(self):
        return self.raw.require_dispatch()

    def consume(self, cell):
        return self.raw.consume(cell)

    def stop(self, reason):
        if self.adapter_primary_code is None:
            self.adapter_primary_code = reason
        self.raw.trip("H_CAPTURE")
        if self.recorder is not None:
            try:
                self.recorder.fail("H_CAPTURE")
            except BaseException:
                # The caller's original failure remains primary; status records
                # fault-receipt failure and the latch is never cleared.
                pass


def create_latch(schedule):
    return LatchView(component().DispatchLatch(schedule))


class WriterIO:
    def __init__(self, writer):
        self.writer = writer

    def write(self, relative_path, data, append=False, fault=False):
        require(relative_path.startswith("hook_evidence/"), "hook-only evidence path")
        return self.writer.write_hook(relative_path,data,append=append,fault=fault)

    def read(self, relative_path):
        key,target = self.writer._target(relative_path)
        require(key.startswith("hook_evidence/"), "hook-only read")
        return target.read_bytes()

    def hook_files(self):
        observed = self.writer._scan()
        return {name:item["actual_bytes"] for name,item in observed.items() if name.startswith("hook_evidence/")}

    def total_bytes(self):
        return self.writer._reconciliation()["actual_total_bytes"]


class OverflowAccountingIO:
    """Adapter test injection only: report +1 aggregate byte after normal setup."""
    def __init__(self, io):
        self.base,self.armed = io,False

    def __getattr__(self,name):
        return getattr(self.base,name)

    def arm(self):
        self.armed = True

    def total_bytes(self):
        return 288*1024**2+1 if self.armed else self.base.total_bytes()


class RecorderView:
    def __init__(self, raw, guard, io):
        self.raw,self.guard,self.io = raw,guard,io

    @property
    def checks(self):
        return self.raw.checks

    def inspect(self, model, label):
        self.raw.inspect(lambda:self.guard.inspect(model),label)
        return not self.raw.latch.terminal

    def finish(self):
        return self.raw.finish()

    def status(self):
        return self.raw.status()


def injected_spec():
    # The tiny module has no TransformerBridge setup, but the exact unchanged
    # HookGuard still snapshots its torch hook/module identities in full.
    return {"installed_sources_sha256":json.loads((HERE/"RUNTIME_SPEC.json").read_bytes())["installed_sources_sha256"],
            "injected_tensor_module_only":True}


def create_recorder(model, writer, latch, labels, spec, *, io_wrapper=None):
    for path,digest in spec["installed_sources_sha256"].items():
        require(sha(Path(path).read_bytes()) == digest, "installed exact hook sources before materialization")
    guard_module = SOURCES.load("production_exact_hook_guard",SCIENCE_COMMIT,SCIENCE+"guard_candidate.py")
    io = WriterIO(writer)
    if io_wrapper:
        io = io_wrapper(io)
    raw = component().Recorder(writer.root,latch.raw,labels,io=io)
    latch.recorder = raw
    try:
        guard = guard_module.HookGuard(model)
        source_lock = {"installed_sources_sha256":spec["installed_sources_sha256"],"passed":True,"before_materialization":True}
        setup = {k:v for k,v in guard.setup.items() if k not in ("before","after")}
        setup_receipt = {"status":"PASS","forward_calls":0,"already_wired_interpretation":"preserved inherited loaded state; skipped setup callbacks not newly authenticated",
            "injected_tensor_module_only":spec.get("injected_tensor_module_only",False)}
        raw.admit(guard.setup["before"],guard.baseline,setup,source_lock,setup_receipt)
        return RecorderView(raw,guard,io)
    except BaseException:
        latch.stop("HOOK_SETUP_FAILURE")
        raise


def bind_writer(area_writer):
    """Unchanged exclusive I/O mechanics; finite faults and reserved hook writes.

    No sticky latch is cleared. A private hook-fault write may bypass the normal
    sticky/reconciliation rejection, but retains every prior/partial byte and
    reservation. It still obeys all path, category, total and per-file bounds.
    """
    from bind_production import compiled,once
    source = SOURCES.read("33f9f7b3b85325334da22abdc82d9db10f53a01e","diagnostics/semantic_confirmation_io_v1/writer.py").decode()
    source = once(source,'self.failures.append({"kind": kind, "type": type(error).__name__, "message": str(error)})',
        'self.failures.append({"code":"EVIDENCE_IO_FAILURE", "full_metadata_recorded":False})')
    source = once(source,'if self.sticky_failure or self.sealed:',
        'if (self.sticky_failure and not getattr(self,"_critical_hook_fault",False)) or self.sealed:')
    source = once(source,'if reconciliation["issues"]:\n                    raise EvidenceIOError("; ".join(reconciliation["issues"]))',
        'if reconciliation["issues"] and not getattr(self,"_critical_hook_fault",False):\n                    raise EvidenceIOError("EVIDENCE_RECONCILIATION_FAILURE")')
    source = once(source,'count = self.write_function(descriptor, data[offset:])',
        'count = (os.write if getattr(self,"_critical_hook_fault",False) else self.write_function)(descriptor, data[offset:])')
    source = source.replace('raise EvidenceIOError(str(error)) from error','raise EvidenceIOError("EVIDENCE_IO_FAILURE") from error')
    source = once(source,'"real_run_authorized": False, "model_work": False}',
        '"real_run_authorized": self.root.parent.name == "real_evidence", "model_work": self.root.parent.name == "real_evidence"}')
    bounded = compiled("production_finite_evidence_writer",source)
    # Rebind the inherited area class to the new bounded base without changing
    # its shared-area admission, raw codec, reconciliation or native closeout.
    class Writer(area_writer,bounded.EvidenceWriter):
        _fail = bounded.EvidenceWriter._fail
        _reconciliation_base = bounded.EvidenceWriter._reconciliation
        def _write(self,*args,**kwargs):
            with self.lock:
                self._owner()
                from area import area_bounds
                total = area_bounds()["evidence_bytes"]
                require(total+len(args[2]) <= self.contract["storage"]["total_bytes"]-self.contract["storage"]["closeout_reserve_bytes"], "shared complete failure closeout reserve")
                return bounded.EvidenceWriter._write(self,*args,**kwargs)
        def closeout(self,name="closeout/index.json"):
            from area import area_bounds
            require(area_bounds()["evidence_bytes"]+self.contract["storage"]["per_file_bytes"] <= self.contract["storage"]["total_bytes"], "shared closeout headroom")
            value = bounded.EvidenceWriter.closeout(self,name)
            area_bounds()
            return value
        def write_hook(self,name,data,*,append=False,fault=False):
            require(type(data) is bytes and name.startswith("hook_evidence/"), "complete hook bytes")
            if fault:
                require(not append and name == "hook_evidence/fault.json" and len(data) <= 65536, "single finite reserved hook fault schema")
                self._fail("hook_fault",None)
                self._critical_hook_fault = True
            try:
                def reserve(ledger,key,n):
                    if key == "hook_evidence/checks.jsonl":
                        require(append and not fault, "typed complete check append")
                        ledger.add_hook_check(n)
                    elif key.endswith(".json"):
                        require(not append, "exclusive full hook JSON")
                        ledger.add_hook_json(key[len("hook_evidence/"):],n,fault=fault)
                    else:
                        require(not append and not fault, "exclusive admitted raw-chunk artifact")
                        ledger._reserve("hooks",key,n)
                return self._write("hooks",name,data,reserve,append=append)
            finally:
                self._critical_hook_fault = False
    return Writer
