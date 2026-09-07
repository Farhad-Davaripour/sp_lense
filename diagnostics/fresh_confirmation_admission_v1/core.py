"""New-namespace admission utilities; no model or dataset reader."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LIMIT=32*1024**2
FILE_LIMIT=5*1024**2
def require(ok,message):
    if not ok: raise ValueError(message)
def sha(raw):
    return hashlib.sha256(raw).hexdigest()
def encoded(value):
    return (json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode()
def read(path):
    return json.loads(Path(path).read_bytes())
def write(name,value,raw=False):
    data=value if raw else encoded(value)
    require(Path(name).name==name,"flat scoped output")
    require(len(data)<=FILE_LIMIT,"per-file cap")
    require(sum(p.stat().st_size for p in HERE.iterdir() if p.is_file())+len(data)<=LIMIT,"namespace cap")
    with (HERE/name).open("xb") as out:
        out.write(data);out.flush();os.fsync(out.fileno())
    return {"path":name,"bytes":len(data),"sha256":sha(data)}
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module
def authenticate(pin):
    raw=(ROOT/pin["path"]).read_bytes()
    require(sha(raw)==pin["sha256"],"source hash: "+pin["path"])
    if pin.get("commit"):
        committed=subprocess.check_output(["git","show",pin["commit"]+":"+pin["path"]],cwd=ROOT)
        require(raw==committed,"committed source bytes: "+pin["path"])
    return raw
def frozen(stage):
    lock=read(HERE/(stage+"_SOURCE_FREEZE.json"))
    for meta in lock["files"]:
        raw=(HERE/meta["path"]).read_bytes()
        require(len(raw)==meta["bytes"] and sha(raw)==meta["sha256"],"frozen stage source")
    return lock
def cohort_json(pairs):
    result={}
    for key,value in pairs:
        require(key not in result,"duplicate JSON key")
        result[key]=value
    return result
