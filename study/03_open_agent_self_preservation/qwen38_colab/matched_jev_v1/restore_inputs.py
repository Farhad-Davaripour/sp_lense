"""Trusted CPU/mounted-Drive restore of exact A/H2; no model loading."""
import hashlib
import json
import shutil
from pathlib import Path

NARROW_RUN="qwen38_narrow_bridge_replay_20261002T211853Z_5ba7a033"
A_RELATIVE=NARROW_RUN+"/narrow_bridge_stream/bridge/checkpoints/update_56"
H2_RELATIVE=NARROW_RUN+"/inputs/models/H2"
H2_FALLBACK="qwen38_preservation_hp_20260930T180110Z_0271ecd2/H2_rank16/checkpoints/adapters/preservation"
PINS={
 "A":{"weights":"8ae28fb51107a31419bcb744d9f47d80b681781a114f8bc9491751410b4b87aa",
      "config":"06a6ad4588f4f0d7b46cf9f7f67f9b7b5028d153134cdd0829a929e1caf9fcda"},
 "H2":{"weights":"0782b3f3ea9b3e67460b7bc981e21526bb6819d3ddb9623def4fbc2218002b30",
       "config":"3df028243d1d0f6643b906143a4c1ad4e8393b5e947d8b52c03f0893601b3528"}}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024**2),b""):h.update(block)
    return h.hexdigest()


def restore(root,run_parent="/content/drive/MyDrive/sp_lense/research3/runs"):
    root=Path(root).resolve();run_parent=Path(run_parent)
    if not str(root).startswith("/content/sp_lense_work/"):
        raise RuntimeError("Explicit isolated Colab run root required")
    if not run_parent.is_dir():raise RuntimeError("Mounted private Drive is not readable")
    sources={"A":run_parent/A_RELATIVE,"H2":run_parent/H2_RELATIVE}
    if not sources["H2"].is_dir():sources["H2"]=run_parent/H2_FALLBACK
    result={}
    for name,source in sources.items():
        for file,key in (("adapter_model.safetensors","weights"),("adapter_config.json","config")):
            if not (source/file).is_file() or sha(source/file)!=PINS[name][key]:
                raise RuntimeError("Mounted exact "+name+" artifact unavailable or hash differs")
        target=root/"inputs/models"/name
        if target.exists():raise RuntimeError("New run input target already exists")
        target.mkdir(parents=True,exist_ok=False)
        for file,key in (("adapter_model.safetensors","weights"),("adapter_config.json","config")):
            shutil.copy2(source/file,target/file)
            if sha(target/file)!=PINS[name][key]:raise RuntimeError("Restored copy differs:"+name)
        result[name]={"path":str(target),"source":str(source),**PINS[name]}
    receipt={"models":result,"CPU_restore_only":True,"model_queries":0,
             "selected_A_unchanged":True,"H2_unchanged":True,
             "must_unmount_Drive_before_model_workers":True}
    (root/"RESTORED_INPUTS.json").write_bytes((json.dumps(receipt,indent=2)+"\n").encode())
    return result
