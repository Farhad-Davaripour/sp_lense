"""Exact source instrumentation and pure extraction; never imports a backend."""
import ast
import hashlib
import json
from pathlib import Path
from loader_diagnostics import PREDICATES

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def source(name):
    pins=json.loads((HERE/"SOURCE_PINS.json").read_bytes())["files"]
    matches=[p for p in pins if p["path"]=="diagnostics/fresh_confirmation_production_v2/"+name]
    if len(matches)!=1: raise ValueError("SOURCE_PIN")
    pin=matches[0];raw=(ROOT/pin["path"]).read_bytes()
    if len(raw)!=pin["bytes"] or hashlib.sha256(raw).hexdigest()!=pin["sha256"]: raise ValueError("SOURCE_BYTES")
    return raw.decode()

def once(text,old,new):
    if text.count(old)!=1: raise ValueError("SOURCE_ANCHOR")
    return text.replace(old,new)

def instrumented():
    original_loader=source("loader.py")
    loader=once(original_loader,"def load_adapter(writer, counters, deadline, admitted):","def _diagnostic_load_adapter(writer, counters, deadline, admitted, diagnostics):")
    loader=loader.replace("require(","diagnostics.check(require,")
    markers=[("    from authority import authenticate","AUTHORITY"),("    for path,digest in spec[\"installed_sources_sha256\"].items():","INSTALLED_SOURCE"),
        ("    raw = SOURCES.read(SCIENCE_COMMIT,spec[\"model\"][\"config_path\"])","CONFIGURATION"),
        ("    guard.install()","GUARD_INSTALL"),("        backend = backend_module.ResearchBackend.load","LOAD"),
        ("        metadata = backend.metadata()","RUNTIME_METADATA"),("        diagnostics.check(require,sha(backend.model.tokenizer.chat_template.encode())","TEMPLATE"),
        ("        recorder = create_recorder","HOOK_SETUP")]
    for anchor,stage in markers:
        indent=anchor[:len(anchor)-len(anchor.lstrip())]
        loader=once(loader,anchor,indent+'diagnostics.enter("'+stage+'")\n'+anchor)
    loader=once(loader,"    guard = ForwardDerivativeGuard", "    diagnostics.latch = latch\n    guard = ForwardDerivativeGuard")
    loader=once(loader,"        recorder = create_recorder",'        diagnostics.capture_enumeration("BEFORE_HOOK_SETUP",list(backend.model.named_parameters()))\n        recorder = create_recorder')
    old='        return RealAdapter(backend.model,recorder,guard,expected_weight_sha256=compatibility["weight_sha256"],runtime_metadata=metadata)'
    new='        diagnostics.enter("ADAPTER_CONSTRUCTOR")\n        return diagnostics.ready(RealAdapter(backend.model,recorder,guard,expected_weight_sha256=compatibility["weight_sha256"],runtime_metadata=metadata,diagnostics=diagnostics))'
    loader=once(loader,old,new)
    loader+='''\n\ndef load_adapter(writer,counters,deadline,admitted):
    from loader_diagnostics import new_context
    diagnostics=new_context(writer,admitted)
    try:
        return _diagnostic_load_adapter(writer,counters,deadline,admitted,diagnostics)
    except BaseException:
        diagnostics.failure_closeout()
        raise
'''
    adapter=source("real_adapter.py")
    original_ast=ast.parse(adapter)
    cls=next(n for n in original_ast.body if isinstance(n,ast.ClassDef) and n.name=="RealAdapter")
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="__init__")
    old=ast.get_source_segment(adapter,init)
    new=old.replace("runtime_metadata=None):","runtime_metadata=None, diagnostics=None):").replace("require(","diagnostics.check(require,")
    new=once(new,'self.parameters = [p for _, p in self.named_parameters]',
        'self.parameters = [p for _, p in self.named_parameters]\n        diagnostics.capture_enumeration("ADAPTER_CAPTURE",self.named_parameters)')
    new=once(new,'self.initial_digest = parameter_digest(self.parameters)',
        'self.initial_digest = parameter_digest(self.parameters)\n        diagnostics.retain_digest(self.initial_digest,expected_weight_sha256)')
    adapter=once(adapter,old,new)
    return loader,adapter

def extracted(text,names,namespace,*,strip_imports=False):
    tree=ast.parse(text)
    tree.body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    if len(tree.body)!=len(names): raise ValueError("EXTRACTED_NAMES")
    if strip_imports:
        class Pure(ast.NodeTransformer):
            def visit_Import(self,node): return None
            def visit_ImportFrom(self,node): return None
        tree=Pure().visit(tree)
    ast.fix_missing_locations(tree)
    exec(compile(tree,"authenticated_pure_extraction","exec"),namespace)
    return namespace
