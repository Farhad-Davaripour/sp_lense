"""Model-free AST checks of the two added calls and their mutation-free interval."""
import ast
import hashlib
from support import HERE,SOURCES,require,sha
COMMIT="84bfd749876c44dc9bb1bc1e75db89f3e491c159"
PREFIX="diagnostics/fresh_confirmation_loader_diagnostics_v1/"

def segment(text,name):
    return next(n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
def code(node):return ast.dump(node,include_attributes=False)
def check():
    old_loader=SOURCES.read(COMMIT,PREFIX+"candidate_loader.py").decode()
    old_adapter=SOURCES.read(COMMIT,PREFIX+"candidate_real_adapter.py").decode()
    loader=(HERE/"candidate_loader.py").read_text();adapter=(HERE/"candidate_real_adapter.py").read_text()
    expected_loader=old_loader.replace("from real_adapter import ForwardDerivativeGuard, RealAdapter","from real_adapter import ForwardDerivativeGuard, RealAdapter, parameter_digest")
    anchor="        recorder = create_recorder(backend.model,writer,latch,labels,spec)"
    require(expected_loader.count(anchor)==1,"one pre-HookRecorder capture site")
    expected_loader=expected_loader.replace(anchor,"        diagnostics.capture_legacy_before(backend.model,parameter_digest)\n"+anchor)
    anchor="        self.initial_digest = parameter_digest(self.parameters)"
    require(old_adapter.count(anchor)==1,"one unchanged C assignment")
    expected_adapter=old_adapter.replace(anchor,"        diagnostics.capture_legacy_after(self.named_parameters,parameter_digest)\n"+anchor)
    require(code(ast.parse(loader))==code(ast.parse(expected_loader)) and code(ast.parse(adapter))==code(ast.parse(expected_adapter)),"only two observation calls and import")
    def predicates(text):
        return [code(n) for n in ast.walk(ast.parse(text)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="check" and isinstance(n.func.value,ast.Name) and n.func.value.id=="diagnostics"]
    require(predicates(loader)==predicates(old_loader) and len(predicates(loader))==10 and predicates(adapter)==predicates(old_adapter) and len(predicates(adapter))==4,"all14 original condition expressions and messages")
    require(code(segment(adapter,"parameter_digest"))==code(segment(old_adapter,"parameter_digest")),"unchanged full raw-byte hash algorithm")
    legacy=SOURCES.read("1d4cc39eba5fb7f987649a05f71247061f005d02","diagnostics/semantic_editor_f03_v2_first_C_v2/editor.py").decode()
    require(code(segment(adapter,"parameter_digest"))==code(segment(legacy,"parameter_digest")),"exact legacy digest AST")
    old_reader=SOURCES.read("99e51b6ef83182e534f443339dcce23c58a462ee","diagnostics/fresh_confirmation_loader_setup_v1/setup_reader.py")
    require((HERE/"base_setup_reader.py").read_bytes()==old_reader,"checked setup reader byte-identical")
    original_handlers=[code(n) for n in ast.walk(segment(old_loader,"_diagnostic_load_adapter")) if isinstance(n,ast.ExceptHandler)]
    handlers=[code(n) for n in ast.walk(segment(loader,"_diagnostic_load_adapter")) if isinstance(n,ast.ExceptHandler)]
    require(handlers==original_handlers,"original inner stop/restore cleanup")
    cls=segment(adapter,"RealAdapter");init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="__init__")
    idx=next(i for i,n in enumerate(init.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=="capture_legacy_after")
    require(ast.unparse(init.body[idx+1])=="self.initial_digest = parameter_digest(self.parameters)","B then unchanged C adjacent")
    context=segment((HERE/"loader_diagnostics.py").read_text(),"Context")
    after=next(n for n in context.body if isinstance(n,ast.FunctionDef) and n.name=="capture_legacy_after")
    branch=after.body[0].body
    bidx=next(i for i,n in enumerate(branch) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=="digest")
    require(ast.unparse(branch[bidx])=="self.order['B'] = digest(self._retained)","B uses strong retained tuple")
    require([ast.unparse(n) for n in branch[bidx+1:]]==["self.order['extra_hash_completed'] += 1", "self.order['phases'].append('B_RETAINED_POST_SETUP')"],"no parameter operation between B and return to C")
    for name in ("capture_legacy_before","capture_legacy_after"):
        method=next(n for n in context.body if isinstance(n,ast.FunctionDef) and n.name==name)
        require(sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="digest" for n in ast.walk(method))==1,"exactly one extra hash at each phase")
    return {"status":"PARITY_VERIFIED","loader_predicates":10,"constructor_predicates":4,"original_cleanup_identical":True,
        "digest_ast_identical":True,"digest_function_source_sha256":sha(ast.get_source_segment(old_adapter,segment(old_adapter,"parameter_digest" )).encode()),
        "additional_full_content_hash_calls":2,"B_C_source_checked_adjacent_interval":True,"no_parameter_mutator_between_B_C":True,
        "base_setup_reader_sha256":sha((HERE/"base_setup_reader.py").read_bytes()),"scientific_change":False}
