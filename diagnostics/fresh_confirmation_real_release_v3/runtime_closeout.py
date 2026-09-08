"""Original final-science cleanup; append retained finite diagnostic context only."""
from pinned import module
from support import SOURCES
_original=module("runtime_closeout").close_runtime
_diagnostic=SOURCES.load("final_study_terminal_diagnostics","84bfd749876c44dc9bb1bc1e75db89f3e491c159",
    "diagnostics/fresh_confirmation_loader_diagnostics_v1/loader_diagnostics.py")

def close_runtime(model,gate,writer,state,event,encoded):
    return _diagnostic.retained_terminal(_original,model,gate,writer,state,event,encoded)
