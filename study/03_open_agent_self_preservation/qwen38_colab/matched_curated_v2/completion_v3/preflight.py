"""Pinned source/data/baseline preflight without model or tokenizer imports."""
import hashlib,json
from pathlib import Path
from admission import observed_model_fit_estimate
RAW="be99179d97f0edd6e078f37980aa749853a64a91a9c5bbff29fa33b5eff8fa01"
EVAL="6baf28618a583dc4f953dd06b712b5d1020c83a04f23f7a367470c60a34d1429"
WORKER="dc3d8bc48b584406aef14cb5ea4e399bc86cd94a4a1a30fb75ce8728df7a8693"
GATE="737637e9960d78d54ca2074853004204fd177fd61a00f5ac35d2124a515404f1"
H2W="0782b3f3ea9b3e67460b7bc981e21526bb6819d3ddb9623def4fbc2218002b30"
H2C="3df028243d1d0f6643b906143a4c1ad4e8393b5e947d8b52c03f0893601b3528"
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as s:
        for b in iter(lambda:s.read(8*1024**2),b""):h.update(b)
    return h.hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def verify(cfg,check_model_files=False):
    timing=cfg.get("phase",{})
    estimate=observed_model_fit_estimate(timing.get("seconds_per_update",19.4715888889),
        timing.get("startup_seconds",175.41824),timing.get("final_audit_seconds",120.))
    if timing.get("fit_seconds",2700)<estimate["estimated_seconds"]:
        raise RuntimeError("Fit window below observed56-update startup/audit estimate; old1051 rejected")
    source=Path(cfg["source_dir"]);freeze=json.loads((source/"FREEZE.json").read_text())
    for name,expected in freeze["sha256"].items():
        if sha(source/name)!=expected:raise RuntimeError("Source freeze differs: "+name)
    for name,expected in (("EVAL_FREEZE.json",EVAL),("eval_worker.py",WORKER),("integrity_gate.py",GATE),("training_data/plain_train.json",RAW)):
        if sha(source/name)!=expected:raise RuntimeError("Pinned corrected source/data differs: "+name)
    ev=json.loads((source/"EVAL_FREEZE.json").read_text());stored=ev.pop("canonical_sha256_without_hash")
    if hashlib.sha256(canonical(ev)).hexdigest()!=stored:raise RuntimeError("Evaluation self-hash differs")
    for name,panel in ev["data"].items():
        if hashlib.sha256(canonical(panel)).hexdigest()!=ev["panel_canonical_sha256"][name]:raise RuntimeError("Panel differs:"+name)
    prep=json.loads((source/"training_data/PREPARATION_AUDIT.json").read_text())
    if not prep["passed"] or prep["rows"]!=112 or prep["unique_curated_prefixes_used"]!=104 or prep["training_service_calls"]!=0:
        raise RuntimeError("Curated112/104 preparation differs")
    h2=cfg["H2"]
    if h2["weights"]!=H2W or h2["config"]!=H2C:raise RuntimeError("Fresh fit must start from exact H2, not partial45")
    if check_model_files:
        if sha(Path(h2["path"])/"adapter_model.safetensors")!=H2W or sha(Path(h2["path"])/"adapter_config.json")!=H2C:
            raise RuntimeError("Actual restored H2 files differ")
    if cfg.get("broker_ready") is not True:raise RuntimeError("Actual trusted runtime Jev bridge readiness not confirmed")
    from reuse_baseline import verify_reuse
    receipt=cfg["A_REUSE_RECEIPT"];baseline=cfg["A_BASELINE_PACKET"]
    reused=verify_reuse(receipt["path"],receipt["sha256"],baseline["path"],baseline["sha256"],source)
    return {"passed":True,"source_freeze_sha256":sha(source/"FREEZE.json"),
        "rows":112,"curated_prefixes":104,"H2_model_files_verified":check_model_files,
        "A_reuse":reused,"A_model_queries":0,"observed_model_fit_estimate":estimate,"token_or_model_imports":0,
        "runtime_token_gate":"Training worker validates literal labels/masking/EOS/shift and3072 without truncation before optimizer",
        "scientific_negative_does_not_stop_full_B_evaluation":True}
