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
MIB = 1024**2
FIXTURES = ("normal","baseline_stop","on_stop","hook_io")
def require(ok, message):
    if not ok: raise ValueError(message)
def sha(raw):
    return hashlib.sha256(raw).hexdigest()
def fixture():
    name=os.environ.get("SP_CONFIRMATION_FIXTURE","")
    require(name in FIXTURES, "explicit frozen fixture")
    return name
def run_dir():
    return EVIDENCE/fixture()
def bounds():
    paths=[p for p in HERE.rglob("*") if p.is_file()]
    total=sum(p.stat().st_size for p in paths)
    evidence=sum(p.stat().st_size for p in paths if p.is_relative_to(EVIDENCE))
    largest=max((p.stat().st_size for p in paths),default=0)
    require(total-evidence<=32*MIB and evidence<=288*MIB and total<=320*MIB and largest<=5*MIB,"fixed shared storage ceilings")
    return {"preparation_bytes":total-evidence,"evidence_bytes":evidence,"namespace_bytes":total,"largest_file_bytes":largest}
def write_new(name,value,*,raw=False,critical=False,preparation=False):
    base=HERE if preparation else run_dir()/"control"
    path=base/name
    require(path.resolve().is_relative_to(base.resolve()),"receipt containment")
    data=value if raw else (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    require(type(data) is bytes and len(data)<=5*MIB,"complete bounded receipt")
    b=bounds()
    key="preparation_bytes" if preparation else "evidence_bytes"
    cap=(32 if critical else 30)*MIB if preparation else (288 if critical else 280)*MIB
    require(b[key]+len(data)<=cap,"reserved final receipt headroom")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as stream:
        require(stream.write(data)==len(data),"full exclusive receipt")
        stream.flush(); os.fsync(stream.fileno())
    require(path.read_bytes()==data,"receipt full readback")
    return {"path":str(path.relative_to(HERE).as_posix()),"bytes":len(data),"sha256":sha(data)}

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
    raw = SOURCES.read(SCIENCE_COMMIT,SCIENCE+"owned.py").decode()
    raw = raw.replace('"scope":"FAKE_WORK_ONLY"','"scope":"RETAINED_PROCESS_ONLY"').replace('"fake_only":True','"ownership_not_model_authority":True')
    anchor='self.stop_deadline=self.stop_started_at+3.'
    require(raw.count(anchor)==1,"one retained stop deadline binding")
    raw = raw.replace(anchor,'self.stop_deadline=min(self.stop_started_at+3.,getattr(self,"cleanup_bound",self.stop_started_at+3.))')
    raw = raw.replace('str(error)','"OWNED_NATIVE_ERROR"')
    owned = types.ModuleType("owned")
    owned.__file__=str(HERE/"owned_core_bound.py")
    sys.modules["owned"]=owned
    exec(compile(raw,owned.__file__,"exec"),owned.__dict__)
    return native,owned
