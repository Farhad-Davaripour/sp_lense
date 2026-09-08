"""Finite loader-proof handoff; final-study counters and science are untouched."""
import json
import time
from support import HERE,SOURCES,require,sha,write_new,bounds
from weight_reader import interpret

MAX_DIAGNOSTIC_BYTES=2*1024**2

def target_loader(target):
    bindings=json.loads((HERE/"BINDINGS.json").read_bytes())
    require(target==bindings["loader"],"exact reviewed loader target")
    raw=SOURCES.read(target["commit"],target["path"])
    require(sha(raw)==target["sha256"] and raw==(HERE/"candidate_loader.py").read_bytes(),"committed candidate loader identity")
    module=SOURCES.load("final_study_reviewed_candidate_loader",target["commit"],target["path"])
    return module.load_adapter

def reservation(writer,execution):
    require(not hasattr(writer,"loader_diagnostic_publisher"),"one native loader proof publisher")
    require(bounds()["evidence_bytes"]+MAX_DIAGNOSTIC_BYTES<=280*1024**2,"initial proof fits ordinary area without spending 8MiB closeout reserve")
    binding=json.loads((HERE/"BINDINGS.json").read_bytes())
    pointer=write_new("LOADER_PROOF_RESERVATION.json",{"schema":"final_loader_proof_reservation.v1","execution":execution,
        "maximum_diagnostic_bytes":MAX_DIAGNOSTIC_BYTES,"native_receipt_file":"LOADER_DIAGNOSTICS.json",
        "native_closeout_copy_inside_existing_file_cap":True,"existing_closeout_reserve_bytes":8*1024**2,
        "resource_contract_sha256":binding["resource_contract_sha256"],"production_ceiling":binding["production_ceiling"],
        "production_seconds":binding["production_seconds"]})
    def publish(data):
        require(type(data) is bytes and len(data)<=MAX_DIAGNOSTIC_BYTES,"bounded complete native loader proof")
        writer.loader_diagnostic_pointer=write_new("LOADER_DIAGNOSTICS.json",data,raw=True)
    writer.loader_diagnostic_publisher=publish
    writer.loader_proof_reservation=pointer
    return pointer

def verify_loader_binding(root,execution,terminal):
    result={"interpretation":"BINDING_INCOMPLETE","binding_verified":False,"scientific_pass":False}
    try:
        root=root.resolve();control=root.parent.parent/"control"
        raw=(control/"LOADER_DIAGNOSTICS.json").read_bytes()
        reservation=json.loads((control/"LOADER_PROOF_RESERVATION.json").read_bytes())
        binding=json.loads((HERE/"BINDINGS.json").read_bytes())
        require(reservation=={"schema":"final_loader_proof_reservation.v1","execution":execution,
            "maximum_diagnostic_bytes":MAX_DIAGNOSTIC_BYTES,"native_receipt_file":"LOADER_DIAGNOSTICS.json",
            "native_closeout_copy_inside_existing_file_cap":True,"existing_closeout_reserve_bytes":8*1024**2,
            "resource_contract_sha256":binding["resource_contract_sha256"],"production_ceiling":binding["production_ceiling"],
            "production_seconds":binding["production_seconds"]},"same admitted final-study proof reservation")
        require(len(raw)<=MAX_DIAGNOSTIC_BYTES,"native diagnostic cap")
        expected=json.loads((HERE/"RUNTIME_SPEC.json").read_bytes())["runtime_compatibility"]["weight_sha256"]
        result=interpret(raw,terminal,execution,sha((HERE/"SOURCE_FREEZE.json").read_bytes()),expected)
    except (OSError,ValueError,KeyError,TypeError):pass
    return result

def run_candidate(writer,counters,deadline,admitted,checked_load):
    # Production supplies only target_loader's exact committed function. The two
    # pure fixtures inject a source-declared callback, never an executable real authority.
    reservation(writer,admitted["execution"])
    adapter=None
    try:
        require(time.monotonic()<deadline,"same final-study worker deadline before loader")
        adapter=checked_load(writer,counters,deadline,admitted)
        context=writer.loader_diagnostics
        require(context.latch is adapter.latch and context.latch.recorder is adapter.recorder.raw,"same created recorder/context/adapter")
        proof=verify_loader_binding(writer.root,admitted["execution"],context.terminal())
        require(proof["binding_verified"] is True and proof["fresh_constructor_hash"]==adapter.initial_digest,"complete authenticated retained-order loader handoff")
        require(time.monotonic()<deadline,"same final-study deadline after proof")
        return adapter
    except BaseException:
        context=getattr(writer,"loader_diagnostics",None)
        if context is not None:context.fail("LD_EXCEPTION")
        if adapter is not None:
            adapter.latch.stop("LOADER_BINDING_FAILURE")
            try:adapter.end_edit()
            finally:adapter.guard.restore()
        raise
