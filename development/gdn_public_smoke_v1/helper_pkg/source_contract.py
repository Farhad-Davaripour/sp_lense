"""Read/compile pinned source only. Never import a model package."""
import ast
import hashlib
import pathlib
import types

PINS = {
    "hf": (".venv/Lib/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", "67cf849081143a998f0e189c551e4b5f532365a055805570747385e400372abb"),
    "bridge": (".venv/Lib/site-packages/transformer_lens/model_bridge/generalized_components/gated_delta_net.py", "b3cf5b73e000b79f55ec0b286be7fccfc578dc2d7bb81025921b853d07acd0f5"),
    "decorator": (".venv/Lib/site-packages/transformers/integrations/hub_kernels.py", "50e5b5f938cdb2c5a2f7e90ae1ab3933cb2d505ab38c6c4f4d0226320df4b94a"),
    "architecture": (".venv/Lib/site-packages/transformer_lens/model_bridge/supported_architectures/qwen3_5.py", "9f7f84035ba26f034aa420129300bfd6585263fee90eaa108ed1253e352d0242"),
}

class Rejected(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)

def need(condition, code):
    if not condition:
        raise Rejected(code)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def authenticate(root, supplied=None):
    root = pathlib.Path(root).resolve()
    result = {}
    for label, (relative, expected) in PINS.items():
        path = (root / relative).resolve()
        need(path.is_relative_to(root), "SOURCE_PATH")
        raw = path.read_bytes() if supplied is None else supplied[label]
        need(sha(raw) == expected, "SOURCE_SHA")
        result[label] = {"path": relative, "bytes": len(raw), "sha256": expected}
    return result

def parse_sources(root):
    receipt = authenticate(root)
    return receipt, {k: ast.parse((pathlib.Path(root) / p).read_bytes()) for k, (p, _) in PINS.items()}

def code_catalog(root, label):
    path = (pathlib.Path(root) / PINS[label][0]).resolve()
    top = compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    result = {}
    def visit(code):
        result[code.co_qualname] = code
        for child in code.co_consts:
            if isinstance(child, types.CodeType):
                visit(child)
    visit(top)
    return result

def source_facts(root):
    receipt, trees = parse_sources(root)
    hf = trees["hf"]
    cls = next(x for x in hf.body if isinstance(x, ast.ClassDef) and x.name == "Qwen3_5GatedDeltaNet")
    init = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "__init__")
    assigned = {x.attr for x in ast.walk(init) if isinstance(x, ast.Attribute) and isinstance(x.ctx, ast.Store)
                and isinstance(x.value, ast.Name) and x.value.id == "self"}
    bridge_cls = next(x for x in trees["bridge"].body if isinstance(x, ast.ClassDef) and x.name == "GatedDeltaNetBridge")
    bridge = next(x for x in bridge_cls.body if isinstance(x, ast.FunctionDef) and x.name == "_hooked_forward")
    calls = [x for x in ast.walk(bridge) if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute)
             and isinstance(x.func.value, ast.Name) and x.func.value.id == "hf"
             and x.func.attr in ("causal_conv1d_fn", "chunk_gated_delta_rule")]
    calls.sort(key=lambda x: x.lineno)
    need(len(calls) == 2 and all(x.func.attr not in assigned for x in calls), "SOURCE_INTERFACE")
    need([k.arg for k in calls[0].keywords] == ["x", "weight", "bias", "activation", "seq_idx"], "CONV_KEYWORDS")
    need(isinstance(calls[0].keywords[-1].value, ast.Constant) and calls[0].keywords[-1].value.value is None, "SEQ_IDX_SOURCE")
    need(len(calls[1].args) == 3 and [k.arg for k in calls[1].keywords] ==
         ["g", "beta", "initial_state", "output_final_state", "use_qk_l2norm_in_kernel"], "CHUNK_ARGUMENTS")
    names = ["causal_conv1d_fn", "torch_chunk_gated_delta_rule"]
    functions = [next(x for x in hf.body if isinstance(x, ast.FunctionDef) and x.name == name) for name in names]
    need(functions[0].args.args[0].arg == "hidden_states", "HF_SIGNATURE")
    return {"schema": "gdn_source_interface.v1", "sources": receipt,
            "bridge_call_lines": [x.lineno for x in calls], "hf_function_lines": [x.lineno for x in functions],
            "hf_constructor_lines": [init.lineno, init.end_lineno],
            "hf_constructor_assigns_neither_helper": True,
            "bridge_calls_ast_sha256": [sha(ast.dump(x, include_attributes=False).encode()) for x in calls],
            "unchanged_numeric_function_ast_sha256": {x.name: sha(ast.dump(x, include_attributes=False).encode()) for x in functions},
            "mapping": {"x": "hidden_states", "seq_idx": "only None; omitted as neutral unsupported metadata"},
            "numerical_execution": False, "actual_missing_member_in_prior_trace": "INFERRED_NOT_RECORDED"}
