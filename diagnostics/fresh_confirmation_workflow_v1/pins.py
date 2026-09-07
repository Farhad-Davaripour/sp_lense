"""Load pinned source only; no historical input/receipt or dataset reads."""
import ast
import hashlib
import json
import math
import struct
import subprocess
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCIENCE_COMMIT = "1d4cc39eba5fb7f987649a05f71247061f005d02"
SCIENCE = "diagnostics/semantic_editor_f03_v2_first_C_v2/"
IO_COMMIT = "33f9f7b3b85325334da22abdc82d9db10f53a01e"
IO = "diagnostics/semantic_confirmation_io_v1/"
GATE_SHA = "972c95d4ef4bc0d9fd245dacd1ef7fc6f773e2c5e3c6736a148482de39a488db"
READS = {}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def source(commit, path):
    raw = subprocess.check_output(["git", "-C", str(ROOT), "show", commit + ":" + path])
    require((ROOT / path).read_bytes() == raw, "working source differs: " + path)
    READS[path] = {"commit": commit, "sha256": sha(raw), "bytes": len(raw)}
    return raw


def module(name, commit, path):
    raw = source(commit, path)
    result = types.ModuleType(name)
    result.__file__ = str(ROOT / path)
    sys.modules[name] = result
    exec(compile(raw, result.__file__, "exec"), result.__dict__)
    return result


def definitions(path, names, namespace):
    raw = source(SCIENCE_COMMIT, SCIENCE + path).decode()
    nodes = [node for node in ast.parse(raw).body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    require({node.name for node in nodes} == set(names), "exact named scientific definitions")
    # Original source spans, compiled without unrelated historical runner imports.
    for node in nodes:
        exec(compile(ast.get_source_segment(raw, node), path + ":" + node.name, "exec"), namespace)
    return namespace


def scientific():
    models = module("confirmation_pinned_centroid", SCIENCE_COMMIT, "src/sp_lense/conditional_gate_models.py")
    ns = {"math": math, "struct": struct, "Path": Path, "ROOT": ROOT,
          "sha": sha, "require": require, "read": lambda p: json.loads(Path(p).read_bytes()),
          "MODEL_SHA": "59ae8c47bbc96668e23769851a62e8f04fb0677e4f5ccc7faaf5ad5aaf5e7414",
          "CenteredCosineCentroidModel": models.CenteredCosineCentroidModel,
          "PARAMETERS": ("grand_mean", "positive_centroid", "negative_centroid", "direction"),
          "EPS": 1e-6, "GOAL": .05, "AIM": .10, "STEP_CAP": .05, "TOTAL_CAP": .20}
    definitions("gate_reload.py", ("parameter_record", "reload_model", "predict"), ns)
    definitions("learned_gate.py", ("feature_bytes", "FrozenGate"), ns)
    definitions("editor.py", ("norm", "EligibilityError", "valid", "accepted", "eligibility", "step_recipe"), ns)
    artifact = source(SCIENCE_COMMIT, SCIENCE + "fitted_parameters.json")
    require(sha(artifact) == GATE_SHA, "frozen learned parameters")
    ns["artifact_raw"] = artifact
    ns["word_score"] = module("confirmation_word_score", SCIENCE_COMMIT, SCIENCE + "word_scoring.py")
    ns["letter_score"] = module("confirmation_letter_score", SCIENCE_COMMIT, "src/sp_lense/future_choice_scoring.py")
    return types.SimpleNamespace(**ns)


def independent():
    return types.SimpleNamespace(
        words=module("confirmation_word_reference", SCIENCE_COMMIT, SCIENCE + "word_reference.py"),
        letters=module("confirmation_letter_reference", SCIENCE_COMMIT, "scripts/future_choice_scoring_reference.py"),
        gate=module("confirmation_gate_reference", SCIENCE_COMMIT, SCIENCE + "gate_reference.py"))


def io_components():
    module("binding", IO_COMMIT, IO + "binding.py")
    return (module("confirmation_pinned_writer", IO_COMMIT, IO + "writer.py"),
            module("confirmation_pinned_reader", IO_COMMIT, IO + "reader.py"))
