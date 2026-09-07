"""Tiny synthetic process faults only; no model/tokenizer imports."""
import sys,time
from pathlib import Path
from core import Budget
mode=sys.argv[1];output=Path(sys.argv[2]);budget=Budget(output)
if mode=="complete":print("synthetic child exited normally")
elif mode=="deadline":
    print("synthetic child waiting",flush=True);time.sleep(10)
elif mode=="overflow":sys.stdout.buffer.write(b"x"*(5*1024**2));sys.stdout.buffer.flush()
elif mode=="prior_science_then_fault":
    budget.event("scientific_failures.jsonl",{"kind":"endpoint_behavior","cell_id":"synthetic_performed_endpoint","fixture_only":True})
    budget.event("cleanup_errors.jsonl",{"kind":"synthetic_cleanup_fault","original_exception":"finite_behavior_failure_retained"})
    raise RuntimeError("synthetic later technical fault")
else:raise SystemExit("fixed synthetic modes only")
