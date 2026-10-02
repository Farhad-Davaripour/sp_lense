"""Build a short, clean Colab entrypoint from a separately committed source bundle.

1. After all diagnostic sources and FIXTURES.json are ready:
   python build_notebook.py --bundle-only
2. Commit that source freeze and bundle, then pin the committed revision:
   python build_notebook.py --source-revision <40-character commit SHA>

The second command verifies the existing bundle and never rebuilds its contents.
"""
import argparse
import gzip
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
REPO = CAMPAIGN.parents[2]
RELATIVE = HERE.relative_to(REPO).as_posix()
BUNDLE_NAME = "source_bundle.json.gz"
NOTEBOOK_NAME = "Research3_Handoff_Reliability_V1.ipynb"
FROZEN_METADATA = ("PROTOCOL.md", "FIXTURE_NOTES.md", "MODEL_FREE_CHECKS.json", "FIXTURES.json")
REQUIRED_CODE = ("diagnostic_worker.py", "generation_capture.py", "launch.py",
                 "worker.py", "diagnostic_world.py", "fixture_build.py", "restore_inputs.py")
REUSED = {
    "model_ops.py": CAMPAIGN / "simple_pilot_v1/model_ops.py",
    "fast_inference.py": CAMPAIGN / "simple_pilot_v1/fast_inference.py",
    "world.py": CAMPAIGN / "simple_pilot_v1/world.py",
    "export_private_runs.py": CAMPAIGN / "simple_pilot_v1/export_private_runs.py",
    "requirements.txt": CAMPAIGN / "simple_pilot_v1/requirements.txt",
    "memory_world.py": CAMPAIGN / "h2_deadline_followup_v1/memory_world.py",
    "activation_capture.py": CAMPAIGN / "fresh_transfer_v1/activation_capture.py",
    "audit.py": CAMPAIGN / "h2_replay_diagnostics_v1/audit.py",
    "model_pin.json": CAMPAIGN / "model_pin.json",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sources():
    for name in (*REQUIRED_CODE, *FROZEN_METADATA):
        if not (HERE / name).is_file():
            raise FileNotFoundError("Source freeze is not ready: " + name)
    paths = {p.name: p for p in sorted(HERE.glob("*.py")) if p.name != "build_notebook.py"}
    paths.update({name: HERE / name for name in FROZEN_METADATA})
    collisions = paths.keys() & REUSED.keys()
    if collisions:
        raise RuntimeError("Immutable helper alias collision: " + str(sorted(collisions)))
    paths.update(REUSED)
    content = {name: path.read_text(encoding="utf-8") for name, path in paths.items()}
    hashes = {name: digest(text.encode("utf-8")) for name, text in content.items()}
    origins = {name: path.relative_to(REPO).as_posix() for name, path in paths.items()}
    return content, hashes, origins


def bundle_only():
    content, hashes, origins = sources()
    freeze = {"schema_version": 1, "status": "Frozen before any new neural diagnostic",
              "frozen_utc": datetime.now(timezone.utc).isoformat(),
              "sha256": hashes, "repository_sources": origins,
              "scope": "Frozen H2, reference, coverage adapters; no weight updates",
              "diagnostic_only": True, "source_text_normalization": "Universal newline to LF, matching Git eol=lf; dependency QA hashes separately record original worktree bytes"}
    freeze_text = json.dumps(freeze, indent=2, sort_keys=True) + "\n"
    (HERE / "SOURCE_FREEZE.json").write_text(freeze_text, encoding="utf-8", newline="\n")
    content["SOURCE_FREEZE.json"] = freeze_text
    raw = json.dumps({"schema_version": 1, "files": content},
                     sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    packed = gzip.compress(raw, mtime=0)
    (HERE / BUNDLE_NAME).write_bytes(packed)
    (HERE / "SOURCE_BUNDLE.sha256").write_text(digest(packed) + "  " + BUNDLE_NAME + "\n",
                                               encoding="utf-8", newline="\n")
    print(json.dumps({"source_files": len(hashes), "bundle_bytes": len(packed),
                      "bundle_sha256": digest(packed), "bundle": str(HERE / BUNDLE_NAME)}, indent=2))


def verify_frozen_bundle():
    content, hashes, origins = sources()
    packed = (HERE / BUNDLE_NAME).read_bytes()
    expected = (HERE / "SOURCE_BUNDLE.sha256").read_text().split()[0]
    if digest(packed) != expected:
        raise RuntimeError("Committed source bundle checksum mismatch")
    payload = json.loads(gzip.decompress(packed))
    if payload.get("schema_version") != 1:
        raise RuntimeError("Unsupported source bundle schema")
    archived = payload["files"]
    freeze = json.loads(archived["SOURCE_FREEZE.json"])
    if hashes != freeze["sha256"] or origins != freeze["repository_sources"]:
        raise RuntimeError("Source changed after freeze; do not silently regenerate the bundle")
    if set(archived) != set(content) | {"SOURCE_FREEZE.json"}:
        raise RuntimeError("Source bundle inventory mismatch")
    if any(archived[name] != value for name, value in content.items()):
        raise RuntimeError("Source bundle content mismatch")
    if archived["SOURCE_FREEZE.json"].encode("utf-8") != (HERE / "SOURCE_FREEZE.json").read_bytes():
        raise RuntimeError("Versioned source freeze differs from bundle")
    return expected, len(packed)


def cell(kind, text, identifier):
    result = {"cell_type": kind, "metadata": {}, "id": identifier,
              "source": text.splitlines(keepends=True)}
    if kind == "code":
        result.update(outputs=[], execution_count=None)
    return result


def build_notebook(revision):
    if not re.fullmatch("[0-9a-fA-F]{40}", revision):
        raise ValueError("--source-revision must be a full 40-character commit SHA")
    expected, bundle_bytes = verify_frozen_bundle()
    revision = revision.lower()
    source_url = ("https://raw.githubusercontent.com/Farhad-Davaripour/sp_lense/" +
                  revision + "/" + RELATIVE + "/" + BUNDLE_NAME)
    heading = """# Research 3 - frozen handoff and batch reliability diagnostic

Run the five cells in order: setup, mount Drive, restore, diagnostic, export.
This stage loads frozen H2, reference, and coverage adapters sequentially; it makes
no weight updates. All model actions are simulated memory operations. The 12
handoff fixtures and exact batch anchors are development diagnostics, including
supplied histories. They do not count as new generalization evidence.

One A100 80 GB; one resident model worker. Cumulative authorization: 200 Colab
units, 79.30 previously accounted. Refresh the observed balance/rate before
running. This stage has a 10-unit cap, including a 2-unit export/release reserve.
Main work is capped at 4,200 seconds and reduced by elapsed setup time.
The export cell verifies the private Drive copy and then releases the runtime.
"""
    setup = """import gzip,hashlib,json,subprocess,sys,time,uuid
from pathlib import Path,PurePosixPath
from urllib.request import urlopen
SESSION_STARTED=time.monotonic()
BALANCE_AT_LAUNCH=122.83
RATE_AT_LAUNCH=6.77
PRIOR_SPEND=79.30
AUTHORIZED_TOTAL_UNITS=200
STAGE_CAP_UNITS=10
RESERVE_UNITS=2
MAX_MAIN_SECONDS=4200
assert STAGE_CAP_UNITS<=min(BALANCE_AT_LAUNCH,AUTHORIZED_TOTAL_UNITS-PRIOR_SPEND)
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],
                   capture_output=True,text=True,check=True).stdout
assert 'A100' in gpu and int(gpu.split(',')[1].split()[0])>=75000,gpu
ROOT=Path('/content/sp_lense_work')/('qwen38_handoff_reliability_'+
    time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])
ROOT.mkdir(parents=True,exist_ok=False)
SOURCE_REVISION=__REVISION__
SOURCE_URL=__URL__
SOURCE_BUNDLE_SHA256=__HASH__
with urlopen(SOURCE_URL,timeout=120) as response: packed=response.read()
assert len(packed)==__SIZE__ and hashlib.sha256(packed).hexdigest()==SOURCE_BUNDLE_SHA256
payload=json.loads(gzip.decompress(packed))
assert payload['schema_version']==1
FILES=payload['files']
for name,content in FILES.items():
    relative=PurePosixPath(name)
    assert not relative.is_absolute() and '..' not in relative.parts
    path=ROOT/'source'/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(content.encode('utf-8'))
for name,expected in json.loads(FILES['SOURCE_FREEZE.json'])['sha256'].items():
    assert hashlib.sha256((ROOT/'source'/name).read_bytes()).hexdigest()==expected,name
(ROOT/'source_bundle.json.gz').write_bytes(packed)
(ROOT/'SOURCE_BUNDLE_DOWNLOAD.json').write_text(json.dumps({
    'revision':SOURCE_REVISION,'url':SOURCE_URL,'sha256':SOURCE_BUNDLE_SHA256,
    'bytes':len(packed)},indent=2))
subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-input','-r',
                str(ROOT/'source/requirements.txt')],check=True,timeout=900)
sys.path.insert(0,str(ROOT/'source'))
print('FROZEN_DIAGNOSTIC_SOURCE_READY',ROOT,gpu,
      'authorized remaining',AUTHORIZED_TOTAL_UNITS-PRIOR_SPEND,flush=True)
"""
    setup = (setup.replace("__REVISION__", repr(revision)).replace("__URL__", repr(source_url))
             .replace("__HASH__", repr(expected)).replace("__SIZE__", str(bundle_bytes)))
    mount = """from google.colab import drive
drive.mount('/content/drive')
print('DRIVE_MOUNT_CONNECTED',flush=True)
"""
    restore = """from restore_inputs import restore_adapters,download_base
MODEL_INPUTS=restore_adapters(ROOT)
drive.flush_and_unmount()
print('DRIVE_UNMOUNTED_BEFORE_MODEL_WORK',flush=True)
BASE_MANIFEST=download_base(ROOT)
print('H2_REFERENCE_COVERAGE_RESTORED',flush=True)
"""
    launch = """exec(compile((ROOT/'source/launch.py').read_text(),
             'trusted_handoff_diagnostic.py','exec'))
"""
    close = (HERE / 'analysis/completion_stage_cell.py').read_text(encoding='utf-8') + '\n' + (HERE / 'analysis/post_run_cell.py').read_text(encoding='utf-8')
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"colab": {"name": NOTEBOOK_NAME},
                             "kernelspec": {"name": "python3", "display_name": "Python 3"},
                             "accelerator": "GPU",
                             "research3_source_revision": revision,
                             "research3_source_bundle_sha256": expected},
                "cells": [cell("markdown", heading, "handoff-v1-00"),
                          cell("code", setup, "handoff-v1-setup"),
                          cell("code", mount, "handoff-v1-mount"),
                          cell("code", restore, "handoff-v1-restore"),
                          cell("code", launch, "handoff-v1-diagnostic"),
                          cell("code", close, "handoff-v1-export")]}
    target = HERE / NOTEBOOK_NAME
    target.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"notebook": str(target), "source_revision": revision,
                      "bundle_sha256": expected, "all_outputs_empty": True}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bundle-only", "--freeze-source", action="store_true",
                      help="Freeze the completed sources before their source commit.")
    mode.add_argument("--source-revision",
                      help="Generate notebook pointing to the already committed frozen bundle.")
    args = parser.parse_args()
    if args.bundle_only:
        bundle_only()
    else:
        build_notebook(args.source_revision)


if __name__ == "__main__":
    main()

