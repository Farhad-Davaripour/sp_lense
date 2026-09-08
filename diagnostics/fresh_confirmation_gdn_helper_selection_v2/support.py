"""Bounded source-only reuse; imports no installed model packages."""
import hashlib
import json
import pathlib
import sys
import types

HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parent.parent
class Rejected(RuntimeError):
    def __init__(self,code):self.code=code;super().__init__(code)
def need(ok,code):
    if not ok:raise Rejected(code)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,sort_keys=True,separators=(",",":"))+"\n").encode()
def authenticate():
    pins=json.loads((HERE/"SOURCE_PINS.json").read_bytes())
    for p,record in pins["files"].items():
        raw=(ROOT/p).read_bytes()
        need(len(raw)==record["bytes"] and sha(raw)==record["sha256"],"PINNED_SOURCE")
    return pins
def component():
    authenticate()
    prefix="diagnostics/fresh_confirmation_gdn_helper_compat_v1/"
    for name,file in (("source_contract","source_contract.py"),("_checked_gdn_compat","compat.py")):
        path=ROOT/(prefix+file)
        old=sys.modules.get(name)
        if old is not None:
            need(getattr(old,"__file__",None)==str(path),"MODULE_COLLISION")
            continue
        module=types.ModuleType(name);module.__file__=str(path);sys.modules[name]=module
        exec(compile(path.read_bytes(),str(path),"exec",dont_inherit=True),module.__dict__)
    return sys.modules["_checked_gdn_compat"]
