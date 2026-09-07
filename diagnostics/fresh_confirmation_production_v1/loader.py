"""Pinned CPU float32 loader binding; never imported by the adapter-only test."""
import json
import os
import sys
import types
from pathlib import Path
from support import HERE, ROOT, SOURCES, SCIENCE_COMMIT, require, sha


def load_adapter(writer, counters, deadline, admitted):
    # Re-admit locally rather than trusting a caller's boolean or fake backend.
    from production_admission import admit_production
    current = admit_production("worker")
    require(current == admitted, "unchanged production authority before loader")
    from real_adapter import ForwardDerivativeGuard, RealAdapter
    from hook_binding import create_latch, create_recorder
    spec = json.loads((HERE/"RUNTIME_SPEC.json").read_bytes())
    for path,digest in spec["installed_sources_sha256"].items():
        require(sha(Path(path).read_bytes()) == digest, "pinned installed hook implementation before construction")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformer_lens.model_bridge import TransformerBridge
    package = types.ModuleType("confirmation_pinned_backend")
    package.__path__ = []
    sys.modules[package.__name__] = package
    config = SOURCES.load(package.__name__+".config",SCIENCE_COMMIT,"src/sp_lense/config.py")
    SOURCES.load(package.__name__+".core",SCIENCE_COMMIT,"src/sp_lense/core.py")
    backend_module = SOURCES.load(package.__name__+".backend",SCIENCE_COMMIT,"src/sp_lense/backend.py")
    raw = SOURCES.read(SCIENCE_COMMIT,spec["model"]["config_path"])
    require(sha(raw) == spec["model"]["config_sha256"], "pinned model configuration")
    from plan import build_plan
    plan = build_plan()
    latch = create_latch([c["cell_id"] for c in plan["cells"]])
    guard = ForwardDerivativeGuard(TransformerBridge,counters,latch,deadline)
    guard.install()  # Before the sole ResearchBackend.load, including setup.
    try:
        require(counters.attempts["load"] == 1 and counters.attempts["forward"] == 0, "one reserved load before all forwards")
        backend = backend_module.ResearchBackend.load(config.load_config(ROOT/spec["model"]["config_path"]),with_lens=False)
        require(guard.forwards == 0 and guard.rejected == 0, "loader adds no forward")
        metadata = backend.metadata()
        compatibility = spec["runtime_compatibility"]
        require(all(metadata[k] == compatibility[k] for k in
            ("model_id","model_revision","device","dtype","d_model","model_layers","packages","lens")), "fitted gate runtime metadata equality")
        require(sha(backend.model.tokenizer.chat_template.encode()) == compatibility["prompt_format"]["chat_template_sha256"], "loaded template bytes only; no encoding")
        labels = []
        categories = {p["prompt_id"]:p["category"] for p in plan["prompts"]}
        for r in plan["requests"]:
            labels.append("REQUEST-entry:"+r["request_id"])
            if categories[r["prompt_id"]] == "self":
                labels.append("ON-entry:"+r["request_id"])
            labels.append("REQUEST-exit:"+r["request_id"])
        labels.append("matrix-finally")
        require(len(labels) == 109, "exact production hook-check denominator")
        recorder = create_recorder(backend.model,writer,latch,labels,spec)
        require(guard.forwards == guard.rejected == 0, "complete hook setup admitted before first forward")
        return RealAdapter(backend.model,recorder,guard,expected_weight_sha256=compatibility["weight_sha256"],runtime_metadata=metadata)
    except BaseException:
        latch.stop("LOAD_OR_SETUP_FAILURE")
        guard.restore()
        raise
