"""Source-locked development capture: preflight -> watched child -> artifacts.

No provider imports occur in the parent or before the worker's preflight and
snapshot-byte check. V1 remains preserved as a failed implementation. Only a
supervisor_success.json receipt, never a worker file alone, denotes completion.
Runtime assurance covers pinned versions and named critical source files, not
a hermetic dependency tree or cryptographic authentication of process memory.
"""
import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path("development/classifier_generalization_v2")
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
SCHEMA = "native_development_execution.v2"
CAPS = dict(seconds=7200, forwards=320, tokens_per_view=320,
            payload_bytes=1310720, output_bytes=33554432,
            tokenizer_loads=1, model_loads=1, fits=0, derivatives=0)
SOURCE_NAMES = ("native_development_runner_v2.py", "snapshot_verifier.py",
                "native_capture_adapter.py", "native_capture_contract.py",
                "tokenizer_input_adapter.py", "capture_executor.py", "capture_export.py")
PROVIDERS = {
    "torch": ("torch", "torch/__init__.py"),
    "torch_hooks": ("torch.nn.modules.module", "torch/nn/modules/module.py"),
    "loading": ("transformers.modeling_utils", "transformers/modeling_utils.py"),
    "config": ("transformers.models.qwen3_5.configuration_qwen3_5", "transformers/models/qwen3_5/configuration_qwen3_5.py"),
    "model": ("transformers.models.qwen3_5.modeling_qwen3_5", "transformers/models/qwen3_5/modeling_qwen3_5.py"),
    "tokenizer": ("transformers.models.qwen2.tokenization_qwen2", "transformers/models/qwen2/tokenization_qwen2.py"),
}
PACKAGES = {"torch", "transformers", "tokenizers", "safetensors"}
LABELS = {"SELF", "OTHER", "NONTERMINATION", "ORDINARY"}
INPUT_ROLES = {"train", "validation", "blueprint", "holdout_index", "model_metadata", "snapshot_lock", "corpus_audit"}
MAX_SMALL_FILE = 2 * 1024 * 1024
DIAGNOSTIC_BYTES = 65536
STATUS_RESERVE = 16384


class GateError(RuntimeError):
    pass


def need(condition, code):
    if not condition:
        raise GateError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "DUPLICATE_JSON_KEY")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs)


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode()


def digest(value):
    need(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "SHA256_FORMAT")
    return value


def small_bytes(path):
    path = Path(path)
    need(path.is_file() and path.stat().st_size <= MAX_SMALL_FILE, "FILE_SIZE_OR_TYPE")
    raw = path.read_bytes()
    need(len(raw) <= MAX_SMALL_FILE, "FILE_GROWTH")
    return raw


def relative_path(root, relative):
    need(type(relative) is str, "PATH_TYPE")
    parts = relative.split("/")
    need(all(re.fullmatch(r"[A-Za-z0-9_.-]+", p) and p not in (".", "..")
             and not p.endswith(".") for p in parts), "PATH_COMPONENT")
    path = (Path(root) / relative).resolve()
    need(path.is_relative_to(Path(root).resolve()), "PATH_ESCAPE")
    return path


def pinned(root, pin):
    need(type(pin) is dict and set(pin) == {"path", "sha256"}, "PIN_SCHEMA")
    path = relative_path(root, pin["path"])
    raw = small_bytes(path)
    need(sha(raw) == digest(pin["sha256"]), "PIN_MISMATCH")
    return raw


def git_bytes(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, timeout=15)
    need(result.returncode == 0, "GIT_READ_FAILED")
    return result.stdout


def check_sources(root, commit, sources):
    # Git object IDs and SHA256-of-file-content are different namespaces.
    need(type(commit) is str and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit), "COMMIT_FORMAT")
    resolved = git_bytes(root, "rev-parse", "--verify", commit + "^{commit}").decode().strip()
    need(resolved == commit, "COMMIT_RESOLUTION")
    for relative, expected in sources.items():
        worktree = small_bytes(relative_path(root, relative))
        committed = git_bytes(root, "cat-file", "blob", commit + ":" + relative)
        need(sha(worktree) == digest(expected) and worktree == committed, "SOURCE_BLOB_MISMATCH")


def runtime_prefix():
    return Path(sys.prefix).resolve()


def load_contract(root):
    path = relative_path(root, (STUDY / "native_capture_contract.py").as_posix())
    spec = importlib.util.spec_from_file_location("_sp_lens_verified_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight(lock_path, expected_sha256, root=ROOT):
    need(not any(name.split(".")[0] in PACKAGES for name in sys.modules), "NATIVE_IMPORT_BEFORE_PREFLIGHT")
    root = Path(root).resolve()
    raw = small_bytes(lock_path)
    need(sha(raw) == digest(expected_sha256), "LOCK_DIGEST")
    lock = strict_json(raw)
    need(lock.get("schema") == SCHEMA and lock.get("scientific_execution_authorized") is True, "LOCK_NOT_AUTHORIZED")
    need(type(lock.get("run_id")) is str and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", lock["run_id"]), "RUN_ID")
    need(lock.get("caps") == CAPS and all(type(v) is int for v in lock["caps"].values()), "CAPS")
    sources = lock.get("source_files")
    need(type(sources) is dict and set(sources) == {(STUDY / n).as_posix() for n in SOURCE_NAMES}, "SOURCE_SET")
    check_sources(root, lock.get("source_commit"), sources)
    runtime = lock["runtime"]
    prefix = runtime_prefix()
    need(Path(runtime["prefix"]).resolve() == prefix, "RUNTIME_PREFIX")
    need(runtime["python"] == ".".join(map(str, sys.version_info[:3])), "PYTHON_VERSION")
    need(set(runtime["packages"]) == PACKAGES, "RUNTIME_PACKAGE_SET")
    for name, version in runtime["packages"].items():
        need(importlib.metadata.version(name) == version, "RUNTIME_VERSION")
    need(set(runtime["provider_sources"]) == set(PROVIDERS), "PROVIDER_SOURCE_SET")
    provider_paths = {}
    for role, (_, relative) in PROVIDERS.items():
        path = relative_path(prefix, "Lib/site-packages/" + relative)
        need(sha(small_bytes(path)) == digest(runtime["provider_sources"][role]), "PROVIDER_SOURCE_HASH")
        provider_paths[role] = path
    threads = lock["threads"]
    need(set(threads) == {"intra", "inter"} and all(type(v) is int and 0 < v <= (os.cpu_count() or 1) for v in threads.values()), "THREAD_SETTINGS")
    need(set(lock["inputs"]) == INPUT_ROLES, "INPUT_SET")
    raws = {}
    for role, pin in lock["inputs"].items():
        need(pin["path"].startswith(STUDY.as_posix() + "/") and "private" not in pin["path"].split("/"), "INPUT_SCOPE")
        resolved_input = relative_path(root, pin["path"])
        need("private" not in [p.lower() for p in resolved_input.relative_to(root).parts], "INPUT_SCOPE")
        if role == "holdout_index":
            need(re.fullmatch(re.escape((STUDY / "holdout_custody").as_posix()) + r"/HOLDOUT_ADMITTED_INDEX_V\d+\.json", pin["path"]), "HOLDOUT_PUBLIC_PATH")
        raws[role] = pinned(root, pin)
    data = {role: strict_json(value) for role, value in raws.items()}
    groups = {g["group_id"]: g for g in data["blueprint"]["groups"]}
    contract = load_contract(root)
    cases, ids = [], set()
    for role, split, count in (("train", "TRAIN", 120), ("validation", "VALIDATION", 40)):
        manifest = data[role]
        need(manifest["split"] == split and manifest["case_count"] == count and len(manifest["cases"]) == count, "MANIFEST_COUNT_ROLE")
        counts = {label: 0 for label in LABELS}
        for case in manifest["cases"]:
            contract.validate_case(case)
            need(case["case_id"] not in ids and case["split"] == split, "CASE_ID_ROLE")
            ids.add(case["case_id"])
            group = groups[case["group_id"]]
            need(all(case[k] == group[k] for k in ("split", "development_fold", "mechanism_ancestry", "template_ancestry")), "BLUEPRINT_BINDING")
            counts[case["class_label"]] += 1
            cases.append(case)
        need(counts == {label: count // 4 for label in LABELS}, "CLASS_BALANCE")
    held = data["holdout_index"]
    need(held["schema"] == "holdout_admitted_public_index.v1", "HOLDOUT_SCHEMA")
    need(held["logical_cases"] == 192 and held["order_views"] == 384 and len(held["case_ids"]) == len(set(held["case_ids"])) == 192, "HOLDOUT_COUNT")
    need(not ids.intersection(held["case_ids"]) and held["class_counts"] == {label: 48 for label in LABELS}, "HOLDOUT_BALANCE")
    need(held["group_counts"] == {"H01": 36, "H02": 36, "H03": 36, "H04": 36, "H05": 24, "H06": 24}, "HOLDOUT_GROUPS")
    expected_ids = {f"H{group:02d}_{label}{i:02d}" for group in range(1, 5)
                    for label in ("S", "O", "N") for i in range(1, 13)}
    expected_ids.update(f"H{group:02d}_A{i:02d}" for group in (5, 6) for i in range(1, 25))
    need(set(held["case_ids"]) == expected_ids, "HOLDOUT_IDS")
    audit = data["corpus_audit"]
    need(audit["status"] == "PASS" and audit["model_outcomes_read"] is False and audit["private_text_exported"] is False, "CORPUS_AUDIT")
    need(audit["dataset_hashes"] == {k: lock["inputs"][k]["sha256"] for k in ("train", "validation", "holdout_index")}, "CORPUS_BINDING")
    metadata = data["model_metadata"]
    need(metadata["schema"] == "native_model_metadata.v1" and metadata["revision"] == REVISION and len(metadata["checkpoint_keys"]) == 488, "MODEL_METADATA")
    snapshot = data["snapshot_lock"]
    need(snapshot["schema"] == "snapshot_lock.v1" and snapshot["revision"] == REVISION and snapshot["scientific_execution_authorized"] is False, "SNAPSHOT_LOCK")
    base = (root / STUDY / "runs").resolve()
    need(base.is_relative_to(root), "OUTPUT_SCOPE")
    return dict(lock=lock, lock_sha256=expected_sha256, root=root, cases=cases,
                data=data, raws=raws, provider_paths=provider_paths,
                base=base, output=base / lock["run_id"], marker=base / "native_model_owner.json")


def local_module(name, ctx):
    module = importlib.import_module(name)
    need(Path(module.__file__).resolve() == (ctx["root"] / STUDY / (name + ".py")).resolve(), "LOCAL_IMPORT_ORIGIN")
    return module


def real_factory(ctx, snapshot_dir):
    modules = {role: importlib.import_module(name) for role, (name, _) in PROVIDERS.items()}
    for role, module in modules.items():
        need(Path(module.__file__).resolve() == ctx["provider_paths"][role], "PROVIDER_IMPORT_ORIGIN")
        need(sha(small_bytes(module.__file__)) == ctx["lock"]["runtime"]["provider_sources"][role], "PROVIDER_CHANGED")
    hooks = modules["torch_hooks"]
    need(all(isinstance(getattr(hooks, k, None), dict) for k in ("_global_forward_hooks", "_global_forward_pre_hooks")), "GLOBAL_HOOK_API")
    torch = modules["torch"]
    torch.set_num_threads(ctx["lock"]["threads"]["intra"])
    torch.set_num_interop_threads(ctx["lock"]["threads"]["inter"])
    meta = ctx["data"]["model_metadata"]
    return local_module("native_capture_adapter", ctx).build_native_adapter(
        snapshot_dir, meta["config"], meta["checkpoint_keys"], torch_api=torch,
        config_class=modules["config"].Qwen3_5Config,
        tokenizer_class=modules["tokenizer"].Qwen2Tokenizer,
        model_class=modules["model"].Qwen3_5ForConditionalGeneration)


def capture(ctx, started):
    verifier = local_module("snapshot_verifier", ctx)
    proof = verifier.verify_snapshot(ctx["raws"]["snapshot_lock"],
        expected_lock_sha256=ctx["lock"]["inputs"]["snapshot_lock"]["sha256"],
        allowed_root=ctx["lock"]["snapshot_cache_root"], max_files=10,
        max_file_bytes=1746942600, max_total_bytes=1769905646, deadline_seconds=300)
    adapter = real_factory(ctx, proof["snapshot_realpath"])
    remaining = CAPS["seconds"] - (time.monotonic() - started)
    need(remaining > 0, "DEADLINE")
    identity = proof["aggregate_tokenizer_identity_sha256"]
    result = local_module("capture_executor", ctx).execute_cases(
        ctx["cases"], encode=adapter.encode, decode=adapter.decode,
        label_token_ids={"A": 32, "B": 33}, expected_identity_sha256=identity,
        observed_identity_sha256=identity, capture_view=adapter.capture_view,
        max_forwards=320, max_output_bytes=1310720, deadline_seconds=remaining)
    need(result["counters"]["forwards"] == 320 and result["counters"]["raw_payload_bytes"] == 1310720, "CAPTURE_ACCOUNTING")
    export = local_module("capture_export", ctx)
    hashes = []
    for record, view in zip(result["records"], result["decoded_views"], strict=True):
        need(sha(export.serialize_activation(view["values"])) == record["payload_sha256"], "PAYLOAD_ROUNDTRIP")
        hashes.append(dict(case_id=view["case_id"], order=view["order"], sha256=record["payload_sha256"]))
    summary = {k: v for k, v in result.items() if k not in ("records", "decoded_views")}
    summary.update(payload_hashes=hashes, native_adapter_coverage=adapter.coverage,
                   threads=ctx["lock"]["threads"], snapshot_proof=proof,
                   scientific_counters=dict(tokenizer_loads=1, model_loads=1, forwards=320, fits=0, derivatives=0))
    return {"features.json": encoded(result["decoded_views"]), "capture_receipt.json": encoded(summary)}


def write_new(path, raw):
    with Path(path).open("xb") as stream:
        stream.write(raw)


def release_owner(path, pid, token, run_id):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        return False
    try:
        value = strict_json(small_bytes(path))
        if all(value.get(k) == v for k, v in dict(pid=pid, token=token, run_id=run_id).items()):
            path.unlink()
            return True
    except (OSError, ValueError, GateError):
        pass
    return False


def worker(lock_path, expected, token):
    started = time.monotonic()
    need(re.fullmatch(r"[0-9a-f]{32}", token or ""), "WORKER_TOKEN")
    ctx = preflight(lock_path, expected)
    ctx["base"].mkdir(parents=True, exist_ok=True)
    owner = dict(pid=os.getpid(), token=token, run_id=ctx["lock"]["run_id"])
    write_new(ctx["marker"], encoded(owner))
    try:
        ctx["output"].mkdir()  # Exclusive: old outputs, even empty, refuse a retry.
        with contextlib.redirect_stdout(sys.stderr):
            files = capture(ctx, started)
        need(set(files) == {"features.json", "capture_receipt.json"}, "ARTIFACT_SET")
        need(sum(len(v) for v in files.values()) <= CAPS["output_bytes"] - STATUS_RESERVE, "OUTPUT_CAP")
        need(time.monotonic() - started <= CAPS["seconds"], "DEADLINE")
        pins = {}
        for name, raw in files.items():
            write_new(ctx["output"] / name, raw)
            pins[name] = dict(bytes=len(raw), sha256=sha(raw))
        report = dict(status="worker_complete", lock_sha256=expected, outputs=pins, **owner)
        write_new(ctx["output"] / "worker_complete.json", encoded(report))
        return report
    finally:
        release_owner(ctx["marker"], os.getpid(), token, owner["run_id"])


def launch(command, stream, cwd):
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    executable = None
    if os.name == "nt" and Path(command[0]).resolve() == Path(sys.executable).resolve():
        # Same mechanism as CPython multiprocessing.popen_spawn_win32: avoid
        # the venv redirector PID while retaining this venv's package prefix.
        executable = sys._base_executable
        env["__PYVENV_LAUNCHER__"] = sys.executable
    return subprocess.Popen(command, executable=executable, stdout=stream, stderr=stream, cwd=cwd, env=env,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def watch(command, cwd, seconds, *, spawn=None):
    """Only the child performs work; this parent just observes and stops it."""
    need(type(seconds) in (int, float) and 0 < seconds <= CAPS["seconds"], "DEADLINE")
    started = time.monotonic()
    with tempfile.TemporaryFile() as stream:
        child = (spawn or launch)(command, stream, cwd)
        need(type(child.pid) is int and child.pid > 0, "CHILD_PID")
        try:
            while child.poll() is None:
                need(time.monotonic() - started < seconds, "HARD_TIMEOUT")
                need(os.fstat(stream.fileno()).st_size <= DIAGNOSTIC_BYTES, "DIAGNOSTIC_CAP")
                time.sleep(0.02)
            need(time.monotonic() - started < seconds, "HARD_TIMEOUT")
            need(child.returncode == 0, "CHILD_FAILED")
            stream.seek(0)
            raw = stream.read(DIAGNOSTIC_BYTES + 1)
            need(len(raw) <= DIAGNOSTIC_BYTES, "DIAGNOSTIC_CAP")
            return child.pid, raw
        except BaseException as exc:
            exc.owned_child_pid = child.pid
            try:
                if child.poll() is None:
                    try:
                        child.terminate()
                    except OSError:
                        pass
                    try:
                        child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=2)
            except Exception as cleanup:
                exc.cleanup_error = type(cleanup).__name__
            exc.owned_child_closed = child.poll() is not None
            raise


def supervise(lock_path, expected):
    started = time.monotonic()
    ctx = preflight(lock_path, expected)
    need(not ctx["output"].exists() and not ctx["marker"].exists(), "OWNER_OR_OUTPUT_EXISTS")
    ctx["base"].mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--lock",
               str(Path(lock_path).resolve()), "--sha256", expected, "--token", token]
    child_pid = None
    child_closed = False
    try:
        child_pid, _ = watch(command, str(ctx["root"]), CAPS["seconds"] - (time.monotonic() - started))
        child_closed = True
        complete = strict_json(small_bytes(ctx["output"] / "worker_complete.json"))
        need(all(complete.get(k) == v for k, v in dict(status="worker_complete", pid=child_pid,
             token=token, run_id=ctx["lock"]["run_id"], lock_sha256=expected).items()), "WORKER_RECEIPT_BINDING")
        need(set(complete["outputs"]) == {"features.json", "capture_receipt.json"}, "ARTIFACT_SET")
        total = 0
        for name, pin in complete["outputs"].items():
            path = ctx["output"] / name
            need(path.is_file() and not path.is_symlink() and path.stat().st_size == pin["bytes"], "ARTIFACT_SIZE")
            total += pin["bytes"]
            need(total <= CAPS["output_bytes"] - STATUS_RESERVE, "OUTPUT_CAP")
            need(sha(path.read_bytes()) == pin["sha256"], "ARTIFACT_HASH")
        need(time.monotonic() - started <= CAPS["seconds"], "HARD_TIMEOUT")
        receipt = dict(complete, status="complete", controller_pid=os.getpid(), elapsed_seconds=time.monotonic() - started)
        write_new(ctx["output"] / "supervisor_success.json", encoded(receipt))
        return receipt
    except BaseException as exc:
        child_pid = getattr(exc, "owned_child_pid", child_pid)
        child_closed = getattr(exc, "owned_child_closed", child_closed)
        failure = dict(status="failed", run_id=ctx["lock"]["run_id"], controller_pid=os.getpid(), child_pid=child_pid, code=str(exc)[:1024])
        write_new(ctx["base"] / ("controller_failure_" + token + ".json"), encoded(failure))
        raise
    finally:
        if child_pid is not None and child_closed:
            release_owner(ctx["marker"], child_pid, token, ctx["lock"]["run_id"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--sha256", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--token", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            result = worker(args.lock, args.sha256, args.token)
        elif args.run:
            result = supervise(args.lock, args.sha256)
        else:
            ctx = preflight(args.lock, args.sha256)
            result = dict(status="preflight_pass", run_id=ctx["lock"]["run_id"], native_execution_performed=False)
        print(encoded(result).decode().strip())
        return 0
    except Exception as exc:
        print(encoded(dict(status="failed", code=type(exc).__name__, detail=str(exc)[:1024])).decode().strip())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
