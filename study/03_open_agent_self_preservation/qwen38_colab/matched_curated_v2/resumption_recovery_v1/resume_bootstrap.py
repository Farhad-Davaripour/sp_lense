"""Trusted source-closure repair and bounded native A continuation; no credentials."""
import json,gzip,hashlib,shutil,subprocess,sys,time
from pathlib import Path
def prepare(root,packet,original_deadline,minimum_B_training_seconds=1200):
    root=Path(root);source=root/"source";new=root/"source_resume"
    if new.exists():raise RuntimeError("Resume source already exists")
    manifest=packet["manifest"]
    for name,expected in manifest["expected_original_sha256"].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError("Original source identity differs: "+name)
    original_freeze=(source/"FREEZE.json").read_bytes()
    frozen=json.loads(original_freeze)
    for name,expected in frozen["sha256"].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected:raise RuntimeError("Original frozen source differs")
    eval_freeze=json.loads((source/"EVAL_FREEZE.json").read_text())
    expected=eval_freeze.pop("canonical_sha256_without_hash")
    canonical=lambda v:json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
    if hashlib.sha256(canonical(eval_freeze)).hexdigest()!=expected:raise RuntimeError("Original evaluation criteria self-hash differs")
    prior=root/"eval_A"
    if not (prior/"reports/FAILURE.json").exists():raise RuntimeError("Prior A failure receipt absent")
    prior_cfg=json.loads((prior/"config.json").read_text());cfg=json.loads((root/"config.json").read_text())
    if prior_cfg["arms"]!=["A"] or prior_cfg["models"]["A"]!=cfg["models"]["A"]:raise RuntimeError("Prior A model identity differs")
    runtime=json.loads((prior/"A/training/receipts/runtime_A.json").read_text())
    pins=eval_freeze["source_pins"]
    if runtime["packed_base_parameter_hash"]!=pins["packed_base_sha256"] or runtime["template_sha256"]!=pins["template_sha256"]:
        raise RuntimeError("Prior A runtime identity differs")
    receipts=list((prior/"A/generation_calls").rglob("receipt.json"))
    if len(receipts)!=47:raise RuntimeError("Expected exactly47 completed native A generation calls")
    for p in receipts:
        if not json.loads(p.read_text())["input_integrity_passed"]:raise RuntimeError("Prior input integrity failed")
    prior_files=[prior/"config.json",prior/"reports/FAILURE.json",prior/"A/training/receipts/runtime_A.json",*receipts]
    for label in ("original_H2","known_development"):
        prior_files.append(prior/"A/evaluation/results"/(label+".json"))
        prior_files.extend((prior/"A/evaluation/trajectories"/label).glob("*.json"))
    prior_hashes={str(p.relative_to(prior)):hashlib.sha256(p.read_bytes()).hexdigest() for p in prior_files}
    evidence={"prior_root":str(prior),"sha256":prior_hashes,"generation_calls":47,
              "original_source_freeze_sha256":hashlib.sha256(original_freeze).hexdigest(),
              "A_model":cfg["models"]["A"],"parameter_updates":0,"old_and_new_not_requeried":True}
    evidence_path=root/"RESUME_A_NATIVE_MANIFEST.json"
    if evidence_path.exists():raise RuntimeError("Native resume manifest already exists")
    evidence_path.write_bytes((json.dumps(evidence,indent=2)+"\n").encode())
    # Pure validation before any model load.
    namespace={};exec(packet["files"]["resume_completed_A.py"],namespace)
    namespace["load_completed_A"](prior,eval_freeze["data"],evidence_path)
    shutil.copytree(source,new)
    for name,text in packet["files"].items():
        if "/" in name or "\\" in name:raise RuntimeError("Unexpected patch path")
        (new/name).write_bytes(text.encode())
    frozen["sha256"].update({name:hashlib.sha256(text.encode()).hexdigest() for name,text in packet["files"].items()})
    frozen["closure_repair"]={"original_source_freeze_sha256":hashlib.sha256(original_freeze).hexdigest(),
        "patch_sha256":manifest["patch_sha256"],"criteria_and_data_changed":False,
        "native_failure_preserved":True,"omitted_completed_panels_are_reused":True}
    (new/"FREEZE.json").write_bytes((json.dumps(frozen,indent=2)+"\n").encode())
    cfg.update(source_dir=str(new),resume_A_from=str(prior),resume_A_manifest=str(evidence_path),
               original_main_deadline_monotonic=float(original_deadline),
               minimum_B_training_seconds=float(minimum_B_training_seconds),B_eval_reserve_seconds=1500)
    if time.monotonic()>=original_deadline:raise TimeoutError("Original deadline expired")
    (root/"RESUME_CONFIG.json").write_bytes((json.dumps(cfg,indent=2)+"\n").encode())
    return {"source":str(new),"deadline_monotonic":original_deadline,"remaining_seconds":original_deadline-time.monotonic(),
            "A_reused_calls":47,"old_and_new_not_requeried":True,"minimum_B_seconds_heuristic":minimum_B_training_seconds,
            "B_eval_reserve_seconds":1500}
def launch(root,model_path):
    root=Path(root);namespace={}
    exec(compile((root/"source_resume/driver_resume_v1.py").read_text(),"driver_resume_v1.py","exec"),namespace)
    deadline=json.loads((root/"RESUME_CONFIG.json").read_text())["original_main_deadline_monotonic"]
    return namespace["launch"](root,model_path,max(1,deadline-time.monotonic()))
