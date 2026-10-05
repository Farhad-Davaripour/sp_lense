"""Trusted setup I/O only: same pinned download, eight full-file SHA workers."""
import hashlib,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
BASE_ID="Qwen/Qwen3.8-27B"
BASE_REVISION="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
ALLOW_PATTERNS=["*.safetensors","*.json","*.jinja","merges.txt","vocab.json"]
def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(16*1024**2),b""):h.update(block)
    return h.hexdigest()
def verify_pinned_files(root,pin,max_workers=8):
    root=Path(root)
    if (pin["repository"],pin["revision"])!=(BASE_ID,BASE_REVISION):raise RuntimeError("Unexpected base pin")
    weights=[v for v in pin["files"] if v["name"].endswith(".safetensors")]
    if len(weights)!=18:raise RuntimeError("Expected18 pinned base shards")
    names=[v["name"] for v in pin["files"]]
    if len(names)!=len(set(names)):raise RuntimeError("Duplicate pinned filename")
    def verify(item):
        path=root/"model"/item["name"]
        if not path.is_file():
            if item["name"].endswith(".safetensors"):raise FileNotFoundError("Missing pinned base shard:"+item["name"])
            return None
        digest=sha_file(path)
        if item.get("sha256") and digest!=item["sha256"]:raise RuntimeError("Base hash mismatch:"+item["name"])
        return item["name"],{"bytes":path.stat().st_size,"sha256":digest}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        verified=list(pool.map(verify,pin["files"]))
    manifest={name:value for name,value in sorted(v for v in verified if v is not None)}
    (root/"MODEL_HASHES.json").write_bytes((json.dumps(manifest,sort_keys=True,indent=2)+"\n").encode())
    print("PARALLEL_BASE_HASHES_VERIFIED",len(manifest),"files;18 weight shards;full SHA256;workers",max_workers,flush=True)
    return manifest
def download_base(root):
    # Trusted setup network operation only, invoked by the Colab controller.
    from huggingface_hub import snapshot_download
    root=Path(root)
    pin=json.loads((root/"source/model_pin.json").read_text(encoding="utf-8-sig"))
    if (pin["repository"],pin["revision"])!=(BASE_ID,BASE_REVISION):raise RuntimeError("Unexpected base pin")
    snapshot_download(repo_id=BASE_ID,revision=BASE_REVISION,local_dir=root/"model",
        token=False,max_workers=8,allow_patterns=ALLOW_PATTERNS)
    print("BASE_DOWNLOAD_COMPLETE",flush=True)
    return verify_pinned_files(root,pin,8)
