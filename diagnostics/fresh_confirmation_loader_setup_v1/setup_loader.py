"""Checked candidate loader and adapter are inseparable; one actual load dispatch."""
import types
from support import HERE,ROOT,SOURCES,require
from setup_counter import LoadSources
COMMIT="84bfd749876c44dc9bb1bc1e75db89f3e491c159"
PREFIX="diagnostics/fresh_confirmation_loader_diagnostics_v1/"

def load(writer,counter,deadline,admitted,mock_case=None):
    raw=SOURCES.read(COMMIT,PREFIX+"candidate_loader.py")
    if mock_case is not None:
        from mock_setup import invoke
        return invoke(raw,writer,counter,deadline,admitted,mock_case)
    module=types.ModuleType("startup_only_candidate_loader");module.__file__=str(HERE/"bound_candidate_loader.py")
    exec(compile(raw,module.__file__,"exec"),module.__dict__)
    module.SOURCES=LoadSources(SOURCES,counter)
    return module.load_adapter(writer,counter,deadline,admitted)
