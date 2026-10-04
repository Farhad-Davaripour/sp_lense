"""Build a clean-start corrected private source packet; no model/service calls."""
import argparse,ast,gzip,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PRIVATE_SHA="8e154829dea6a02c9af1838ad10d958de6465d17c546c59746e3a60f3adefd18"
LEGACY_SHA="bbffc9668ce7c251dcd7de5aad2255d1c02aed2089a9fe66f8689c1614191498"
INTEGRITY_SHA="481b6795a4eb5a600b20da9b21e427272af2d23fbd160a5592e5e0abbbb0861e"
EVAL_SHA="6baf28618a583dc4f953dd06b712b5d1020c83a04f23f7a367470c60a34d1429"
EVAL_CANONICAL="3afe0825345afe750997a254f9807a583a2cc318a8992499f747b43c7b0d1922"
RAW_SHA="be99179d97f0edd6e078f37980aa749853a64a91a9c5bbff29fa33b5eff8fa01"
DEPENDENCIES=("diagnostic_world.py","build_diagnostics.py","fresh_world.py")
def sha(value):return hashlib.sha256(value).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def pinned(path,expected):
    value=Path(path).read_bytes()
    if sha(value)!=expected:raise RuntimeError("Pinned input bytes differ: "+str(path))
    return value
def build(private_packet,output_dir):
    raw=pinned(private_packet,PRIVATE_SHA);packet=json.loads(gzip.decompress(raw));files=dict(packet["files"])
    if len(files)!=33:raise RuntimeError("Expected original33-file amended private packet")
    old_freeze=json.loads(files["FREEZE.json"])
    for name,expected in old_freeze["sha256"].items():
        if sha(files[name].encode())!=expected:raise RuntimeError("Original packet source freeze differs: "+name)
    if sha(files["EVAL_FREEZE.json"].encode())!=EVAL_SHA:raise RuntimeError("Exact amended evaluation bytes differ")
    ev=json.loads(files["EVAL_FREEZE.json"]);stored=ev.pop("canonical_sha256_without_hash")
    if stored!=EVAL_CANONICAL or sha(canonical(ev))!=stored:raise RuntimeError("Evaluation self-hash differs")
    for name,panel in ev["data"].items():
        if sha(canonical(panel))!=ev["panel_canonical_sha256"][name]:raise RuntimeError("Frozen panel differs: "+name)
    if sha(files["training_data/plain_train.json"].encode())!=RAW_SHA:raise RuntimeError("Literal raw112 bytes differ")
    plain=json.loads(files["training_data/plain_train.json"]);concept=json.loads(files["training_data/concept_train.json"])
    prep=json.loads(files["training_data/PREPARATION_AUDIT.json"])
    if not prep["passed"] or len(plain)!=112 or len(concept)!=112 or prep["training_service_calls"]!=0:
        raise RuntimeError("Curated preparation differs")
    for a,b in zip(plain,concept):
        if {k:v for k,v in a.items() if k!="messages"}!={k:v for k,v in b.items() if k!="messages"}:
            raise RuntimeError("Literal targets/categories/tools changed")
        if b["messages"][:-1]!=a["messages"] or len(b["messages"])!=len(a["messages"])+1:
            raise RuntimeError("Curated raw history changed")
    legacy_blob=pinned(HERE.parents[1]/"narrow_bridge_replay_v1/source_bundle.json.gz",LEGACY_SHA)
    legacy=json.loads(gzip.decompress(legacy_blob))["files"]
    integrity_blob=pinned(HERE/"INTEGRITY_REPAIR_V2.json.gz",INTEGRITY_SHA)
    integrity=json.loads(gzip.decompress(integrity_blob))
    for name,text in integrity["files"].items():
        if sha(text.encode())!=integrity["manifest"]["sha256"][name]:raise RuntimeError("Integrity capsule entry differs")
    for name in DEPENDENCIES:files[name]=legacy[name]
    original_eval_sha=sha(files["eval_worker.py"].encode())
    files["eval_worker.py"]=integrity["files"]["eval_worker.py"]
    files["integrity_gate.py"]=integrity["files"]["integrity_gate.py"]
    # Preserve standarddriver2031 admission default, training worker, recipe and data.
    driver_tree=ast.parse(files["driver.py"])
    minimum=[n for n in ast.walk(driver_tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
             and n.func.attr=="get" and n.args and isinstance(n.args[0],ast.Constant)
             and n.args[0].value=="minimum_B_training_seconds"]
    if len(minimum)!=1 or not isinstance(minimum[0].args[1],ast.Constant) or minimum[0].args[1].value!=2031:
        raise RuntimeError("Standard original admission guard differs")
    available=set(files);local_modules={Path(n).stem for n in legacy if n.endswith(".py")}|{Path(n).stem for n in files if n.endswith(".py")}
    missing=[]
    for name,text in files.items():
        if not name.endswith(".py"):continue
        tree=ast.parse(text,filename=name)
        for node in ast.walk(tree):
            modules=[v.name.split(".")[0] for v in node.names] if isinstance(node,ast.Import) else [(node.module or "").split(".")[0]] if isinstance(node,ast.ImportFrom) else []
            for module in modules:
                if module in local_modules and module+".py" not in available:missing.append([name,module+".py"])
    if missing:raise RuntimeError("Missing local import closure: "+json.dumps(missing))
    origins=dict(old_freeze.get("origins",{}))
    for name in DEPENDENCIES:origins[name]="sealed_narrow_bundle::"+name
    origins["eval_worker.py"]="used_integrity_repair481b::eval_worker.py"
    origins["integrity_gate.py"]="used_integrity_repair481b::integrity_gate.py"
    files.pop("FREEZE.json")
    freeze=dict(old_freeze);freeze.update(sha256={n:sha(t.encode()) for n,t in files.items()},origins=origins)
    freeze["fresh_clean_start_repair"]={"input_private_packet_sha256":PRIVATE_SHA,"original_freeze_sha256":sha(packet["files"]["FREEZE.json"].encode()),
        "legacy_bundle_sha256":LEGACY_SHA,"integrity_capsule_sha256":INTEGRITY_SHA,
        "old_eval_worker_sha256":original_eval_sha,"new_eval_worker_sha256":sha(files["eval_worker.py"].encode()),
        "EVAL_FREEZE_byte_identical":True,"raw112_targets_and_training_input_bytes_unchanged":True,
        "no_resume_history_or_prior_failure_required":True,"default_B_admission_seconds":2031,
        "historical_phase_admission_metadata_is_not_new_run_authorization":True}
    files["FREEZE.json"]=json.dumps(freeze,indent=2)+"\n"
    result=gzip.compress(canonical({"schema_version":1,"files":files}),mtime=0)
    output=Path(output_dir).resolve()
    repository=next((p for p in HERE.parents if (p/".git").exists()),HERE)
    if output.is_relative_to(repository):raise RuntimeError("Derived private packet must be outside Git")
    output.mkdir(parents=True,exist_ok=False)
    (output/"MATCHED_PRIVATE_PACKET.json.gz").write_bytes(result)
    receipt={"schema_version":1,"packet_sha256":sha(result),"packet_bytes":len(result),"files":len(files),
        "freeze_sha256":sha(files["FREEZE.json"].encode()),"input_private_packet_sha256":PRIVATE_SHA,
        "integrity_capsule_sha256":INTEGRITY_SHA,"legacy_bundle_sha256":LEGACY_SHA,
        "EVAL_FREEZE_sha256":EVAL_SHA,"EVAL_canonical_sha256":stored,"raw112_file_sha256":RAW_SHA,
        "import_closure_passed":True,"dependency_additions":list(DEPENDENCIES),
        "default_B_admission_seconds":2031,"historical_admission_reuse_prohibited":True,
        "models_tokenizers_GPU_API_upload_calls":0,"output_dir":str(output),
        "no_prior_failure_or47_calls_required":True,"new_trial_launched":False}
    (output/"DERIVED_PACKET_RECEIPT.json").write_bytes((json.dumps(receipt,indent=2)+"\n").encode())
    return receipt
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--private-packet",required=True);p.add_argument("--output-dir",required=True)
    a=p.parse_args();print(json.dumps(build(a.private_packet,a.output_dir)))
