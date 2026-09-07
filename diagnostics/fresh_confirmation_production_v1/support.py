"""Candidate limits and checked source provider; execution requires complete v2 input lock."""
import hashlib
import json
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EVIDENCE = HERE / "fake_evidence"
REAL_EVIDENCE = HERE / "real_evidence"
MIB = 1024**2


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def bounds():
    files = [p for p in HERE.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    evidence = sum(p.stat().st_size for p in files if p.is_relative_to(EVIDENCE))
    largest = max((p.stat().st_size for p in files), default=0)
    require(total-evidence <= 32*MIB and evidence <= 32*MIB and total <= 64*MIB and largest <= 5*MIB,
            "frozen 32 MiB preparation / 32 MiB adapter evidence / 64 MiB total / 5 MiB file caps")
    return {"preparation_bytes":total-evidence,"evidence_bytes":evidence,"namespace_bytes":total,"largest_file_bytes":largest}


def production_bounds():
    """Disabled real area uses the unchanged contract; fixtures count as prep."""
    require(json.loads((HERE/"PRODUCTION_AUTHORIZATION.json").read_bytes())["production_authorized"] is True,
            "real evidence allocation needs separate production authority")
    files = [p for p in HERE.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    evidence = sum(p.stat().st_size for p in files if p.is_relative_to(REAL_EVIDENCE))
    largest = max((p.stat().st_size for p in files),default=0)
    require(total-evidence <= 32*MIB and evidence <= 288*MIB and total <= 320*MIB and largest <= 5*MIB,
            "unchanged production preparation/evidence/namespace/file ceilings")
    return {"preparation_bytes":total-evidence,"evidence_bytes":evidence,"namespace_bytes":total,"largest_file_bytes":largest}


def write_new(name, value, *, raw=False, critical=False):
    path = HERE/name
    require(path.resolve().is_relative_to(HERE) and not path.is_relative_to(EVIDENCE), "new preparation receipt path")
    data = value if raw else (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    require(isinstance(data,bytes) and len(data) <= 5*MIB, "complete bounded preparation bytes")
    before = production_bounds() if REAL_EVIDENCE.exists() else bounds()
    require(before["preparation_bytes"]+len(data) <= (32 if critical else 30)*MIB, "reserved 2 MiB preparation closeout")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    require(path.read_bytes() == data, "complete preparation receipt readback")
    return {"path":path.relative_to(HERE).as_posix(),"bytes":len(data),"sha256":sha(data)}


def check_freeze():
    lock = json.loads((HERE/"SOURCE_FREEZE.json").read_bytes())
    for name,digest in lock["source_sha256"].items():
        require(sha((HERE/name).read_bytes()) == digest, "frozen source: "+name)
    return lock


class BoundSources:
    """Exact git-show interface over preverified commit/path/hash bytes; no subprocess."""
    def __init__(self):
        self.pins = json.loads((HERE/"SOURCE_PINS.json").read_bytes())["files"]
        self.cache = {}

    def read(self, commit, path):
        key = commit+":"+path
        require(key in self.pins, "undeclared source access")
        raw = (ROOT/path).read_bytes()
        expected = self.pins[key]
        require(len(raw) == expected["bytes"] and sha(raw) == expected["sha256"], "locked committed source bytes")
        self.cache[key] = raw
        return raw

    def check_output(self, command, **kwargs):
        require(not kwargs and len(command) == 5 and command[:2] == ["git","-C"]
                and Path(command[2]).resolve() == ROOT and command[3] == "show", "only fixed source git-show interface")
        commit,path = command[4].split(":",1)
        return self.read(commit,path)

    def load(self, name, commit, path):
        result = types.ModuleType(name)
        result.__file__ = str(ROOT/path)
        sys.modules[name] = result
        exec(compile(self.read(commit,path),result.__file__,"exec"),result.__dict__)
        return result


SOURCES = BoundSources()
SCIENCE_COMMIT = "1d4cc39eba5fb7f987649a05f71247061f005d02"
SCIENCE = "diagnostics/semantic_editor_f03_v2_first_C_v2/"


def ownership():
    native = SOURCES.load("native",SCIENCE_COMMIT,SCIENCE+"native.py")
    # native.py independently authenticates the already reviewed probe source.
    SOURCES.read(SCIENCE_COMMIT,"diagnostics/windows_worker_identity_probe_v1/probe.py")
    owned = SOURCES.load("owned",SCIENCE_COMMIT,SCIENCE+"owned.py")
    return native,owned
