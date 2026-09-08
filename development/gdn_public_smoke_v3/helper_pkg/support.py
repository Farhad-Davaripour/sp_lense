"""Bounded source-only reuse; imports no installed model packages."""
import hashlib
import json
import pathlib
import sys
import types

HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[2]
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
    from . import compat
    return compat
