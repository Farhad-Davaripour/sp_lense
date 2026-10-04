"""Trusted new-source/new-primary-A launch; preserves all earlier failed runs."""
import gzip,hashlib,importlib.util,json,os,shutil,subprocess,sys
from pathlib import Path


def install_and_launch(root,model_path,max_seconds,packet_path):
    root=Path(root);old=root/"source_resume";new=root/"source_integrity_v2"
    if new.exists():raise RuntimeError("New integrity source already exists; preserve it")
    packet=json.loads(gzip.decompress(Path(packet_path).read_bytes()))
    for name,text in packet["files"].items():
        if hashlib.sha256(text.encode()).hexdigest()!=packet["manifest"]["sha256"][name]:raise RuntimeError("Integrity packet bytes differ")
    old_freeze=(old/"FREEZE.json").read_bytes()
    eval_bytes=(old/"EVAL_FREEZE.json").read_bytes()
    shutil.copytree(old,new)
    for name in ("integrity_gate.py","eval_worker.py"):
        (new/name).write_bytes(packet["files"][name].encode())
    freeze=json.loads(old_freeze)
    for name in ("integrity_gate.py","eval_worker.py"):
        freeze["sha256"][name]=hashlib.sha256((new/name).read_bytes()).hexdigest()
    freeze["integrity_gate_revision"]={"old_source_freeze_sha256":hashlib.sha256(old_freeze).hexdigest(),
        "EVAL_FREEZE_unchanged_sha256":hashlib.sha256(eval_bytes).hexdigest(),
        "packed_parameter_pin_unchanged":"ec43e239f6ffad79d854149f9d7115aeda5292beddd411dd763cac8f9ddcd755",
        "post_generation_same_wrapper_plus_measured_unload_canonical_contents":True}
    (new/"FREEZE.json").write_text(json.dumps(freeze,indent=2)+"\n")
    if (new/"EVAL_FREEZE.json").read_bytes()!=eval_bytes:raise RuntimeError("Evaluation criteria changed")
    for relative,expected in freeze["sha256"].items():
        if hashlib.sha256((new/relative).read_bytes()).hexdigest()!=expected:raise RuntimeError("Copied integrity source freeze differs:"+relative)
    primary=root/"eval_A_integrity_v2";primary.mkdir(exist_ok=False)
    cfg=json.loads((root/"config.json").read_bytes());cfg.update(arms=["A"],source_dir=str(new))
    (primary/"config.json").write_text(json.dumps(cfg,indent=2)+"\n")
    selection={"primary_A_path":str(primary),"source_dir":str(new),"full_fresh_frozen_panel":True,
        "prior_failed_integrity_A_runs_preserved_not_accepted":True,"no_checkpoint_or_panel_selection":True,
        "new_source_freeze_sha256":hashlib.sha256((new/"FREEZE.json").read_bytes()).hexdigest(),
        "unchanged_EVAL_FREEZE_sha256":hashlib.sha256(eval_bytes).hexdigest(),"max_seconds":max_seconds}
    (root/"reports/A_INTEGRITY_PRIMARY_SELECTION.json").write_text(json.dumps(selection,indent=2)+"\n")
    spec=importlib.util.spec_from_file_location("integrity_launch_driver",new/"driver.py")
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    log=root/"training/logs/A_integrity_primary.log";log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("wb") as output:
        process=subprocess.Popen([sys.executable,str(new/"eval_worker.py"),"--root",str(primary),
            "--model",str(model_path),"--config",str(primary/"config.json"),"--max-seconds",str(max_seconds)],
            stdout=output,stderr=subprocess.STDOUT,env=driver.model_env(),start_new_session=True)
    receipt={**selection,"pid":process.pid,"log":str(log),"launch_returned_immediately":True}
    (root/"reports/A_INTEGRITY_PRIMARY_PID.json").write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt
