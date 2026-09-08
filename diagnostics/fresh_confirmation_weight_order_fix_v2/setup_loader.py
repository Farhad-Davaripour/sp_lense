"""Same owned startup loader, with only source-locked retained legacy weight ordering."""
import json
import types
from support import HERE,SOURCES,require,sha
from setup_counter import LoadSources

def load(writer,counter,deadline,admitted,mock_case=None):
    require(mock_case is None,"this successor has no full fake-loader workflow; use fixed proof-only batch")
    raw=(HERE/"candidate_loader.py").read_bytes();binding=json.loads((HERE/"BINDINGS.json").read_bytes())["loader"]
    require(binding["namespace_file"]=="candidate_loader.py" and sha(raw)==binding["sha256"],"exact extended loader source")
    module=types.ModuleType("order_diagnostic_candidate_loader");module.__file__=str(HERE/"candidate_loader.py")
    exec(compile(raw,module.__file__,"exec"),module.__dict__)
    module.SOURCES=LoadSources(SOURCES,counter)
    return module.load_adapter(writer,counter,deadline,admitted)
