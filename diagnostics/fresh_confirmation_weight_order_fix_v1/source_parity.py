"""AST-bound minimal bookkeeping delta; no executable research imports."""
import ast
from support import HERE,SOURCES,require,sha
COMMIT="84bfd749876c44dc9bb1bc1e75db89f3e491c159"
PREFIX="diagnostics/fresh_confirmation_loader_diagnostics_v1/"
def code(node):return ast.dump(node,include_attributes=False)
def segment(text,name):return next(n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
def expected_sources():
    loader=SOURCES.read(COMMIT,PREFIX+"candidate_loader.py").decode()
    adapter=SOURCES.read(COMMIT,PREFIX+"candidate_real_adapter.py").decode()
    loader=loader.replace("from real_adapter import ForwardDerivativeGuard, RealAdapter","from real_adapter import ForwardDerivativeGuard, RealAdapter, parameter_digest")
    loader=loader.replace("        recorder = create_recorder(backend.model,writer,latch,labels,spec)","        diagnostics.capture_legacy_before(backend.model,parameter_digest,compatibility[\"weight_sha256\"])\n        recorder = create_recorder(backend.model,writer,latch,labels,spec)")
    loader=loader.replace("runtime_metadata=metadata,diagnostics=diagnostics))","runtime_metadata=metadata,diagnostics=diagnostics,legacy_weight_parameters=diagnostics.legacy_parameters))")
    adapter=adapter.replace("expected_weight_sha256=None, runtime_metadata=None, diagnostics=None):","expected_weight_sha256=None, runtime_metadata=None, diagnostics=None, legacy_weight_parameters=None):")
    adapter=adapter.replace("        self.initial_digest = parameter_digest(self.parameters)","        self.legacy_weights = diagnostics.bind_legacy(self.named_parameters,legacy_weight_parameters)\n        self.initial_digest = parameter_digest(self.legacy_weights.parameters)")
    adapter=adapter.replace("        if digest:\n            result.update(parameter_sha256=parameter_digest(self.parameters),initial_parameter_sha256=self.initial_digest)","        result[\"legacy_weight_reference_metadata_unchanged\"] = self.legacy_weights.matches(current)\n        if digest:\n            result.update(parameter_sha256=parameter_digest(self.legacy_weights.parameters),initial_parameter_sha256=self.initial_digest)")
    return loader,adapter
def check_candidate(loader,adapter):
    expected=expected_sources()
    require(code(ast.parse(loader))==code(ast.parse(expected[0])) and code(ast.parse(adapter))==code(ast.parse(expected[1])),"only exact retained-order interface delta")
def check():
    import json
    loader=(HERE/"candidate_loader.py").read_text();adapter=(HERE/"candidate_real_adapter.py").read_text()
    check_candidate(loader,adapter)
    old_loader=SOURCES.read(COMMIT,PREFIX+"candidate_loader.py").decode();old_adapter=SOURCES.read(COMMIT,PREFIX+"candidate_real_adapter.py").decode()
    def predicates(text):
        return [code(n) for n in ast.walk(ast.parse(text)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="check" and isinstance(n.func.value,ast.Name) and n.func.value.id=="diagnostics"]
    require(predicates(loader)==predicates(old_loader) and len(predicates(loader))==10 and predicates(adapter)==predicates(old_adapter) and len(predicates(adapter))==4,"original14 predicate expressions/messages")
    require([code(n) for n in ast.walk(segment(loader,"_diagnostic_load_adapter")) if isinstance(n,ast.ExceptHandler)]==[code(n) for n in ast.walk(segment(old_loader,"_diagnostic_load_adapter")) if isinstance(n,ast.ExceptHandler)],"original inner latch stop/guard restore")
    legacy=SOURCES.read("1d4cc39eba5fb7f987649a05f71247061f005d02","diagnostics/semantic_editor_f03_v2_first_C_v2/editor.py").decode()
    require(code(segment(adapter,"parameter_digest"))==code(segment(old_adapter,"parameter_digest"))==code(segment(legacy,"parameter_digest")),"unchanged legacy digest")
    for pin in json.loads((HERE/"INFRASTRUCTURE_PARITY.json").read_bytes())["files"]:
        require((HERE/pin["namespace_file"]).read_bytes()==SOURCES.read(pin["commit"],pin["path"]),"unchanged checked infrastructure")
    context=segment((HERE/"loader_diagnostics.py").read_text(),"Context")
    require(sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="digest" for n in ast.walk(context))==1,"one additional fresh pre-hook hash")
    # Prospective static negative: neither stale constructor nor stale closeout is allowlisted.
    for expression in ("parameter_digest(self.legacy_weights.parameters)","result.update(parameter_sha256=parameter_digest(self.legacy_weights.parameters)"):
        tampered=adapter.replace(expression,"self.initial_digest" if expression.startswith("parameter_digest") else "result.update(parameter_sha256=self.initial_digest",1)
        try:check_candidate(loader,tampered)
        except ValueError:pass
        else:raise ValueError("cached hash substitution incorrectly admitted")
    return {"status":"PARITY_VERIFIED","loader_predicates":10,"constructor_predicates":4,"original_inner_cleanup_identical":True,
        "legacy_digest_ast_identical":True,"digest_function_source_sha256":sha(ast.get_source_segment(adapter,segment(adapter,"parameter_digest")).encode()),
        "current_registry_parameter_arrays_unchanged":True,"content_hashes_all_use_legacy_tuple":True,"additional_hash_passes":1,
        "stale_hash_source_substitutions_rejected":2,"scientific_change":False}
