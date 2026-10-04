"""Measure the historical wrapped-name PIN before and after PEFT unload.

No forward/optimizer operation. Canonical per-parameter tensor fingerprints also
prove that the observed unload changes names, not packed base tensor contents.
Quantization metadata/buffers are outside the historical parameter-hash scope.
"""
import hashlib
import json

PREFIX = "base_model.model."


def fingerprint(model, unloaded=False):
    import torch
    legacy=hashlib.sha256();parameters={};raw_names=[];total=0
    for name,value in model.named_parameters():
        if "lora_" in name:continue
        if value.requires_grad:raise RuntimeError("Base parameter trainable")
        raw_names.append(name)
        historical_name=(PREFIX+name) if unloaded else name
        normalized=historical_name.replace(".base_layer","")
        if not normalized.startswith(PREFIX):raise RuntimeError("Unexpected PEFT wrapped parameter naming")
        canonical_name=normalized[len(PREFIX):]
        if canonical_name in parameters:raise RuntimeError("Canonical base parameter name collision")
        data=value.data.detach().cpu().contiguous().view(torch.uint8).numpy()
        raw=memoryview(data);total+=raw.nbytes
        legacy.update(normalized.encode());legacy.update(raw)
        parameters[canonical_name]={"shape":list(value.shape),"dtype":str(value.dtype),
            "bytes":raw.nbytes,"sha256":hashlib.sha256(raw).hexdigest()}
    stable=hashlib.sha256(json.dumps(parameters,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return {"historical_wrapped_name_hash":legacy.hexdigest(),"stable_parameter_manifest_sha256":stable,
        "parameters":parameters,"parameter_count":len(parameters),"packed_bytes":total,
        "raw_name_examples":raw_names[:5],"historical_name_prefix_restored":unloaded,
        "scope":"Non-LoRA named parameter packed contents, shape and dtype; no quantization metadata or buffers"}


def verify_and_unload(model, expected_pin, receipt_path, save):
    before=fingerprint(model)
    receipt={"before_unload":before,"expected_historical_pin":expected_pin,
             "post_generation_wrapped_pin_passed":before["historical_wrapped_name_hash"]==expected_pin,
             "completed":False,"extra_forwards":0,"optimizer_updates":0}
    save(receipt_path,receipt)
    if not receipt["post_generation_wrapped_pin_passed"]:
        raise RuntimeError("Actual post-generation wrapped base differs from frozen PIN")
    base=model.unload()
    after=fingerprint(base,unloaded=True)
    receipt.update(after_unload=after,
        restored_unloaded_pin_passed=after["historical_wrapped_name_hash"]==expected_pin,
        same_canonical_parameter_names_shapes_dtypes_and_contents=before["parameters"]==after["parameters"],
        stable_manifest_passed=before["stable_parameter_manifest_sha256"]==after["stable_parameter_manifest_sha256"])
    receipt["completed"]=all(receipt[k] for k in ("post_generation_wrapped_pin_passed",
        "restored_unloaded_pin_passed","same_canonical_parameter_names_shapes_dtypes_and_contents","stable_manifest_passed"))
    save(receipt_path,receipt)
    if not receipt["completed"]:raise RuntimeError("Measured unloaded base integrity differs from wrapped frozen PIN")
    return base,receipt
