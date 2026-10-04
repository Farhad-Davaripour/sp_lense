"""Private nonsecret source/data packet from verified legacy dependencies."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
CAMPAIGN=HERE.parent
LEGACY_BUNDLE=CAMPAIGN/"narrow_bridge_replay_v1/source_bundle.json.gz"
LEGACY_SHA="bbffc9668ce7c251dcd7de5aad2255d1c02aed2089a9fe66f8689c1614191498"
LEGACY=(
 "model_ops.py","audit.py","batching.py","paired_batching.py","fast_inference.py",
 "world.py","memory_world.py","study_worker.py","fast_worker.py","evaluate.py",
 "activation_capture.py","generation_capture.py","requirements.txt","model_pin.json",
 "base_download.py","export_private_runs.py")
NEW=(
 "driver.py","restore_inputs.py","training_worker.py","prepare_training.py",
 "eval_worker.py","service_bridge.py","concept_schema_v2.py","schema_v1.py","matched_inputs.py",
 "ANCHOR_SELECTION.json","EVAL_FREEZE.json","PROTOCOL.md","PHASE_ADMISSION.json")
DATA=("plain_train.json","concept_train.json","PREPARATION_AUDIT.json")
FORBIDDEN=("local_service.py","jev_budget.py",".env","jev_client.py","ANNOTATIONS.json")
RAW_SHA="be99179d97f0edd6e078f37980aa749853a64a91a9c5bbff29fa33b5eff8fa01"


def sha(value):return hashlib.sha256(value).hexdigest()


def build(training_data,output_dir):
    packed=LEGACY_BUNDLE.read_bytes()
    if sha(packed)!=LEGACY_SHA:raise RuntimeError("Sealed legacy bundle changed")
    old=json.loads(gzip.decompress(packed))["files"]
    files={name:old[name] for name in LEGACY}
    origins={name:"sealed_narrow_bundle::"+name for name in LEGACY}
    original_newline_bytes={}
    for name in NEW:
        path=HERE/name
        if not path.is_file():raise FileNotFoundError("Required matched source not ready:"+name)
        raw=path.read_bytes()
        if b"\r\n" in raw:
            original_newline_bytes[name]={"original_sha256":sha(raw),"packet_policy":"Normalize source/metadata CRLF to LF before runtime freeze"}
            raw=raw.replace(b"\r\n",b"\n")
        files[name]=raw.decode("utf-8");origins[name]="matched_jev_v1/"+name
    training_data=Path(training_data)
    if sha((training_data/"plain_train.json").read_bytes())!=RAW_SHA:
        raise RuntimeError("Exact raw112 file changed")
    for name in DATA:
        raw=(training_data/name).read_bytes()
        if b"\r\n" in raw and name!="plain_train.json":
            raise RuntimeError("New concept/audit input must be LF before freeze:"+name)
        files["training_data/"+name]=raw.decode("utf-8")
        origins["training_data/"+name]="actual prepared104-annotation training input"
    if any(Path(name).name in FORBIDDEN for name in files):
        raise RuntimeError("Controller/secret/provider-response artifact must not enter model packet")
    audit=json.loads(files["training_data/PREPARATION_AUDIT.json"])
    if not audit["passed"] or audit["primary_scope_register_repair"] or audit["rows"]!=112:
        raise RuntimeError("Actual primary preparation failed or scope repaired")
    hashes={name:sha(value.encode("utf-8")) for name,value in files.items()}
    freeze={"schema_version":1,"sha256":hashes,"origins":origins,"private_nonsecret_packet":True,
            "legacy_bundle_sha256":LEGACY_SHA,"maximum_resident_model_workers":1,
            "credential_or_service_transport_included":False,
            "raw_provider_responses_included":False,"teacher_future_annotations_included":False,
            "source_newline_provenance":original_newline_bytes,
            "protected_raw_data_byte_policy":"Preserve exact be991 file bytes, including outer CRLF JSON whitespace"}
    files["FREEZE.json"]=json.dumps(freeze,indent=2)+"\n"
    blob=json.dumps({"schema_version":1,"files":files},ensure_ascii=False,
                    separators=(",",":"),sort_keys=True).encode("utf-8")
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=False)
    packet=gzip.compress(blob,mtime=0)
    (output/"MATCHED_PRIVATE_PACKET.json.gz").write_bytes(packet)
    receipt={"sha256":sha(packet),"bytes":len(packet),"source_files":len(files),
             "freeze_sha256":sha(files["FREEZE.json"].encode()),
             "credential_or_service_transport_included":False,
             "legacy_bundle_sha256":LEGACY_SHA,"raw_provider_responses_included":False}
    (output/"PACKET_RECEIPT.json").write_bytes((json.dumps(receipt,indent=2)+"\n").encode())
    return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--training-data",required=True);p.add_argument("--output-dir",required=True)
    a=p.parse_args();print(json.dumps(build(a.training_data,a.output_dir)))
