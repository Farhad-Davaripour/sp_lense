"""Bounded SYNTHETIC execution adapter; no model loader or tokenizer entry point."""
import json,time
from pathlib import Path
from core import HERE,Budget,Counter,read,require,sha,check_freeze
import editor
from learned_gate import RoutingMismatch

class ForwardGuard:
    """Count at the bridge's forward boundary, installed before the loader."""
    def __init__(self, model_class, counter):
        self.model_class, self.counter, self.cell = model_class, counter, None
        self.original = model_class.forward

    def install(self):
        def counted(model, *args, **kwargs):
            return self.counter.call(self.cell, lambda: self.original(model, *args, **kwargs))
        self.model_class.forward = counted

    def restore(self):
        self.model_class.forward = self.original



def execute(plan,backend_factory,model_class,output,seconds=180):
    require(plan["execution_mode"]=="SYNTHETIC_ONLY" and model_class.__module__=="synthetic_backend","this job has no real-model authority")
    if (HERE/"freeze.json").exists():check_freeze()
    else:require(plan["fixture_scope"].startswith("compact"),"full declared fixture requires prospective source lock")
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    budget=Budget(output);started=time.monotonic();deadline=started+seconds
    budget.write_bytes("fitted_parameters.json",(HERE/"fitted_parameters.json").read_bytes())
    budget.write("plan.json",plan)
    budget.write("release.json",{"freeze_sha256":sha((HERE/"freeze.json").read_bytes()) if (HERE/"freeze.json").exists() else None,"scope":"synthetic only; compact preparation before freeze when null"})
    ledger=Counter(budget,plan["cells"],deadline)
    derivatives=editor.DerivativeLedger(output,plan["derivative_cells"],deadline)
    guard=ForwardGuard(model_class,ledger);guard.install()
    status,error="complete",None
    try:
        backend=backend_factory()
        require(backend.metadata()["execution_mode"]=="SYNTHETIC_ONLY","no real backend allowed")
        require(ledger.attempts==0,"unexpected synthetic construction forward")
        editor.evaluate(plan,backend,ledger,derivatives,output,guard)
    except (RoutingMismatch,editor.EligibilityError) as fault:
        status,error="scientific_stop",type(fault).__name__+": "+str(fault)
    except BaseException as fault:
        status,error="technical_stop",type(fault).__name__+": "+str(fault)
        budget.event("technical_faults.jsonl",{"kind":type(fault).__name__,"error":str(fault),"monotonic":time.monotonic()})
    finally:
        guard.restore()
        budget.write("execution_receipt.json",{"execution_status":status,"error":error,"elapsed_seconds":time.monotonic()-started,
            "real_model_loads":0,"real_model_forwards":0,"tokenizer_calls":0,"gate_fit_calls":0,
            "synthetic_forward_attempts":ledger.attempts,"synthetic_forward_completed":ledger.completed,
            "synthetic_derivatives_attempted":derivatives.attempts,"synthetic_derivatives_completed":derivatives.completed,
            "cursor":ledger.cursor,"skips":len(ledger.skips),"deadline_seconds":seconds,"forward_guard_restored":model_class.forward is guard.original})
        budget.write("unrun.json",plan["cells"][ledger.cursor:])
        budget.write("runner_finished.json",{"status":"writer exited synchronously","monotonic":time.monotonic()})
    return {"status":status,"error":error,"output":str(output)}

def final_inventory(output):
    output=Path(output)
    require((output/"runner_finished.json").exists() and (output/"judge_process.json").exists() and (output/"judge_receipt.json").exists(),"inventory after runner and saved judge receipts")
    files=[{"path":p.relative_to(output).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p.read_bytes())} for p in sorted(output.rglob("*")) if p.is_file() and p.name!="FINAL_INVENTORY.json"]
    Budget(output).write("FINAL_INVENTORY.json",{"files":files,"sole_owner":"run.final_inventory after runner/judge/report/receipts","synthetic_only":True})
