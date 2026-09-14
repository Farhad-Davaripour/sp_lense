"""Create a fresh prompted capture lock using reviewed, committed source bytes."""
import hashlib, json, subprocess
from pathlib import Path
import native_development_runner_v2 as native
import prompted_span_runner_v1 as prompted

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
def read(path): return json.loads(path.read_bytes())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def pin(path): return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}

def main():
    old_path = HERE / "RUN_LOCK_SPAN_CAPTURE_V1.json"
    assert sha(old_path) == "a1174a712f3208813cbdbf948f6c2af2cda3b1dd8f5febdd20e43062d2e3f16c"
    old = read(old_path)
    expected = {
        "prompted_span_runner_v1.py": "fa94c39a3705ee2dd24f376ea707ca7cc66cca9a05693c2885fb754d9fcd65e1",
        "test_prompted_span_runner_v1.py": "a5770815b027cad78d4d7ee53f6d7549452da54b4a6901f034bf2fdc7466955e",
        "prompted_input_adapter_v1.py": "3dc979f89a4fe3db9c822cf2e8a93b55c95f16b39c15eae25b95ed36f9b6419b",
    }
    for name, digest in expected.items(): assert sha(HERE / name) == digest
    reviews = ["PROMPTED_RUNNER_REVIEW_V1.md", "PROMPTED_INPUT_REVIEW_V1.md",
               "PROMPTED_COMPRESSION_PLAN_REVIEW_V1.md"]
    for name in reviews: assert "PASS_SCOPED" in (HERE / name).read_text(encoding="utf-8")
    sanity_path = HERE / "PROMPTED_TOKENIZER_SANITY_V1.json"
    sanity = read(sanity_path)
    assert sanity["source_lock_sha256"] == sha(old_path)
    assert sanity["max_full_tokens"] == 196 and sanity["max_prefix_tokens"] == 151
    assert sanity["reference_locks"] == old["reference_locks"]
    assert not (HERE / "runs/native_model_owner.json").exists()
    run_id = "prompted_capture_20260914_v1"
    assert not (HERE / "runs" / run_id).exists()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    sources = {(prompted.STUDY / name).as_posix(): sha(HERE / name) for name in prompted.SOURCE_NAMES}
    native.check_sources(ROOT, commit, sources)
    lock = dict(old)
    lock.update(schema=prompted.SCHEMA, run_id=run_id, source_commit=commit, source_files=sources,
        caps=dict(prompted.CAPS),
        inputs={**old["inputs"], "prompt_sanity": pin(sanity_path)},
        independent_review_pins={name: pin(HERE / name) for name in reviews},
        prior_unprompted_lock=pin(old_path),
        scope="User-approved fixed-query condition. One 640-forward capture, 320 tokens per view, 1800 seconds, one frozen Qwen owner. No fits, generation, logits, steering or holdout access.",
        new_fit_execution_authorized=False)
    path = HERE / "RUN_LOCK_PROMPTED_CAPTURE_V1.json"
    native.write_new(path, native.encoded(lock))
    ctx = prompted.preflight(path, sha(path), ROOT)
    print(json.dumps({"path": str(path), "sha256": sha(path), "cases": len(ctx["cases"]),
                      "status": "preflight_pass", "native_execution_performed": False}))

if __name__ == "__main__": main()
