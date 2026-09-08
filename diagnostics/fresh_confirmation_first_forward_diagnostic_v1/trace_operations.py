"""Pure callback order shared by the actual diagnostic and inert recorder cases."""
from forward_trace import span

def first_call(adapter_call,serialize,publish,validate):
    with span("FORWARD_ADAPTER"):
        result=adapter_call()
    with span("LOGITS_SERIALIZATION"):
        raw=serialize(result)
    with span("LOGITS_PUBLICATION"):
        publication=publish(raw)
    with span("INPUT_CAPTURE_IDENTITY"):
        validate(result)
    return publication

def cleanup_step(stage,callback):
    with span(stage):return callback()

def diagnostic_cleanup(end_edit,parameter_state,latch_admit,inspect,identity,restore):
    """Approved diagnostic-only sequence; refusal never reaches the provider."""
    try:
        cleanup_step("CLEANUP_END_EDIT",end_edit)
        parameter=cleanup_step("CLEANUP_PARAMETER_STATE",parameter_state)
        cleanup_step("CLEANUP_LATCH_INSPECTION",latch_admit)
        inspection=cleanup_step("CLEANUP_HOOK_INSPECTION",inspect)
        return cleanup_step("CLEANUP_IDENTITY",lambda:identity(parameter,inspection))
    finally:
        cleanup_step("CLEANUP_GUARD_RESTORE",restore)
