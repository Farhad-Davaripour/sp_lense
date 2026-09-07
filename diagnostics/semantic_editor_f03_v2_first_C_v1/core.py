"""Scoped recorder:32MiB preparation,96MiB separately authorized real attempt."""
import hashlib,json,math,os,subprocess,threading,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
HOOK="blocks.10.hook_out"
MAX_FORWARDS=11
FILE_CAP=5*1024**2
TOTAL_CAP=32*1024**2
WORKER_SECONDS=300
CLEANUP_SECONDS=15
FINALIZE_SECONDS=90
MARGIN=.05-1e-6
MASS=.8
def require(ok,message):
    if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def json_bytes(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
def read(path):return json.loads(Path(path).read_bytes())
def git(*args):return subprocess.check_output(["git","-C",str(ROOT),*args])
def check_freeze():
    lock=read(HERE/"freeze.json")
    for name,digest in lock["source_sha256"].items():require(sha((HERE/name).read_bytes())==digest,"source freeze changed: "+name)
    return lock

class Budget:
    """Namespace-wide32MiB preparation ceiling; reserve4MiB for fault/log closeout."""
    def __init__(self, root):
        self.root = Path(root)
        self._fault = None
        self._lock = threading.Lock()

    @property
    def fault_code(self):
        with self._lock:
            return self._fault

    def fault(self, code):
        with self._lock:
            self._fault = self._fault or code

    def write_bytes(self, name, data, mode="xb"):
        target = (self.root / name).resolve()
        require(target.is_relative_to(self.root.resolve()), "output path escape")
        require(mode in ("xb", "ab"), "exclusive or append only")
        previous = target.stat().st_size if target.exists() else 0
        require(previous + len(data) <= FILE_CAP, "per-file output cap")
        attempt_root=self.root
        for parent in self.root.parents:
            if parent==HERE:break
            if (parent/"plan.json").exists():attempt_root=parent;break
        other = sum(p.stat().st_size for p in attempt_root.rglob("*") if p.is_file())
        critical=target.name in {"fault.json","technical_faults.jsonl","cleanup_errors.jsonl","execution_receipt.json","runner_finished.json","unrun.json","judge_receipt.json","judge_process.json","FINAL_INVENTORY.json","REPORT.md","capture.json","supervisor_final.json","worker_final.json","process_identity.json","recovery_receipt.json","final_closeout.json"}
        require(other + len(data) <= (96 if critical else 92) * 1024**2, "per-attempt all-artifact/log output reserve cap")
        aggregate=sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file() and not p.is_relative_to(HERE/"real_attempt"))
        require(target.is_relative_to(HERE/"real_attempt") or aggregate+len(data)<=TOTAL_CAP-(0 if critical else 4*1024**2),"preparation namespace output reserve cap")
        with target.open(mode) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

    def write(self, name, value):
        self.write_bytes(name, json_bytes(value))

    def event(self, name, value):
        self.write_bytes(name, (json.dumps(value, allow_nan=False) + "\n").encode(), "ab")


class Counter:
    """Conditional bridge-boundary guard, installed before loading."""
    def __init__(self,budget,cells,deadline,now=time.monotonic):
        self.budget,self.cells,self.deadline,self.now=budget,cells,deadline,now
        self.cursor=self.attempted=self.completed=0
        self.failed=self.pending=False
        self.skips=[]
    @property
    def attempts(self): return self.attempted
    def call(self,cell,forward):
        require(not self.failed and not self.pending,"failed or pending attempt cannot retry")
        require(self.attempted<MAX_FORWARDS and self.cursor<len(self.cells),"12th forward blocked")
        require(cell==self.cells[self.cursor],"forward conditional order/preload")
        require(self.now()<self.deadline,"worker deadline")
        self.attempted+=1
        self.cursor+=1
        self.pending=True
        event={"attempt":self.attempted,"cell":cell}
        self.budget.event("forward_events.jsonl",{**event,"event":"attempt_started","monotonic":self.now()})
        try:
            result=forward()
        except BaseException as error:
            self.failed=True
            self.budget.event("forward_events.jsonl",{**event,"event":"attempt_failed","monotonic":self.now(),"error":str(error)[:1024]})
            raise
        finally:
            self.pending=False
        self.completed+=1
        self.budget.event("forward_events.jsonl",{**event,"event":"attempt_completed","monotonic":self.now()})
        return result
    def skip(self,cell,reason,after_cell_id):
        require(not self.failed and not self.pending and self.cursor<len(self.cells) and cell==self.cells[self.cursor] and (cell["optional"] or (reason=="no_opportunity" and cell["request_id"] is not None)),"illegal skip")
        require(reason in ("accepted","quality_failure","no_opportunity"),"invalid skip reason")
        event={"cell":cell,"reason":reason,"after_cell_id":after_cell_id,"monotonic":self.now()}
        self.budget.event("skip_events.jsonl",event)
        self.skips.append(event)
        self.cursor+=1

# Namespace-local read-only loading for authenticated selection/tokenizer helpers.
def load_module(name,path):
    import importlib.util
    spec=importlib.util.spec_from_file_location(name,ROOT/path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
