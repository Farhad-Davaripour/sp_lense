"""Input-only utilities. No model, gate, or experiment execution entry point."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LIMIT=16*1024**2
FILE_LIMIT=5*1024**2
def require(value,message):
    if not value:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+"\n").encode()
def read(path):return json.loads(Path(path).read_bytes())
def git(*args):return subprocess.check_output(["git",*args],cwd=ROOT)
def blob(commit,path):return git("show",f"{commit}:{path}")
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module
def write(name,value,raw=False):
    data=value if raw else encoded(value)
    require(len(data)<=FILE_LIMIT,"per-file limit")
    require(sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file())+len(data)<=LIMIT,"namespace limit")
    target=HERE/name;require(target.parent==HERE,"flat bounded artifact path")
    with target.open("xb") as out:out.write(data);out.flush();os.fsync(out.fileno())
    return {"path":name,"sha256":sha(data),"bytes":len(data)}
