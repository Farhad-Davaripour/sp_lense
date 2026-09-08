"""Existing independent scientific identity judge plus reviewed loader proof."""
from pinned import install
install(globals(),"saved_runtime")
_original_verify_runtime_identity=verify_runtime_identity

def verify_runtime_identity(root,state,io,capture):
    # Existing scientific/fault/prefix/full-hook conditions still execute unchanged.
    result=_original_verify_runtime_identity(root,state,io,capture)
    from loader_handoff import verify_loader_binding
    diagnostic=capture["terminal"].get("loader_diagnostics",{})
    proof=verify_loader_binding(root,state["execution"],diagnostic)
    identity=(capture["terminal"].get("runtime_record") or {}).get("final_identity") or {}
    valid=proof["binding_verified"] is True and identity.get("legacy_weight_reference_metadata_unchanged") is True
    return {**result,"evidence_valid":bool(result["evidence_valid"] and valid),
        "permits_pass":bool(result.get("permits_pass",False) and valid),"loader_binding":proof}
