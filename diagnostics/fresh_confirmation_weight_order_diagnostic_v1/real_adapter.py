"""Only the source-locked B-before-C extension; no loader or scientific remedy."""
import json
from support import HERE,require,sha
raw=(HERE/"candidate_real_adapter.py").read_bytes()
binding=json.loads((HERE/"BINDINGS.json").read_bytes())["candidate_adapter"]
require(binding["namespace_file"]=="candidate_real_adapter.py" and sha(raw)==binding["sha256"],"exact extended adapter source")
exec(compile(raw,__file__,"exec"),globals())
_OriginalGuard=ForwardDerivativeGuard
def ForwardDerivativeGuard(model_class,counters,latch,deadline):
    guard=_OriginalGuard(model_class,counters,latch,deadline);counters.guard=guard;return guard
