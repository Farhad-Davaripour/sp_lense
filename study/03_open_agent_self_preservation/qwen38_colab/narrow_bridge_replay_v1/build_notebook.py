"""Freeze a small external source bundle, then pin an output-free Colab notebook.

Use --bundle-only before committing source; --source-revision verifies rather
than rebuilding the frozen bundle. No model/tokenizer is loaded by this builder.
"""
import argparse
import gzip
import hashlib
import json
import re
from datetime import datetime,timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
CAMPAIGN=HERE.parent
REPO=CAMPAIGN.parents[2]
RELATIVE=HERE.relative_to(REPO).as_posix()
BUNDLE_NAME='source_bundle.json.gz'
NOTEBOOK_NAME='Research3_Narrow_Bridge_Replay_V1.ipynb'
FROZEN_METADATA=('PROTOCOL.md','H2_STATE_PROVENANCE.json')
REQUIRED_CODE=('dataset_build.py','replay_worker.py','worker.py','handoff_loader.py',
               'prepare_data.py','restore_inputs.py','collect.py','launch.py','production_pin.py')
REUSED={
 'model_ops.py':CAMPAIGN/'simple_pilot_v1/model_ops.py',
 'fast_inference.py':CAMPAIGN/'simple_pilot_v1/fast_inference.py',
 'world.py':CAMPAIGN/'simple_pilot_v1/world.py',
 'batching.py':CAMPAIGN/'simple_pilot_v1/batching.py',
 'export_private_runs.py':CAMPAIGN/'simple_pilot_v1/export_private_runs.py',
 'requirements.txt':CAMPAIGN/'simple_pilot_v1/requirements.txt',
 'study_worker.py':CAMPAIGN/'simple_pilot_v1/worker.py',
 'fast_worker.py':CAMPAIGN/'simple_pilot_v1/worker_fast.py',
 'diagnostic_world.py':CAMPAIGN/'simple_pilot_v1/diagnostic_world.py',
 'build_diagnostics.py':CAMPAIGN/'simple_pilot_v1/build_diagnostics.py',
 'data/benign_competence_dev.json':CAMPAIGN/'simple_pilot_v1/data/benign_competence_dev.json',
 'data/preference_validation.json':CAMPAIGN/'simple_pilot_v1/data/preference_validation.json',
 'memory_world.py':CAMPAIGN/'h2_deadline_followup_v1/memory_world.py',
 'recipes.py':CAMPAIGN/'h2_deadline_followup_v1/recipes.py',
 'paired_batching.py':CAMPAIGN/'h2_deadline_followup_v1/paired_batching.py',
 'evaluate.py':CAMPAIGN/'h2_deadline_followup_v1/evaluate.py',
 'fresh_world.py':CAMPAIGN/'fresh_transfer_v1/fresh_world.py',
 'activation_capture.py':CAMPAIGN/'fresh_transfer_v1/activation_capture.py',
 'audit.py':CAMPAIGN/'h2_replay_diagnostics_v1/audit.py',
 'behavior.py':CAMPAIGN/'h2_replay_diagnostics_v1/behavior.py',
 'frozen_worker.py':CAMPAIGN/'h2_replay_diagnostics_v1/frozen_worker.py',
 'base_download.py':CAMPAIGN/'handoff_reliability_v1/restore_inputs.py',
 'generation_capture.py':CAMPAIGN/'handoff_reliability_v1/generation_capture.py',
 'handoff/diagnostic_world.py':CAMPAIGN/'handoff_reliability_v1/diagnostic_world.py',
 'handoff/fixture_build.py':CAMPAIGN/'handoff_reliability_v1/fixture_build.py',
 'handoff/diagnostic_worker.py':CAMPAIGN/'handoff_reliability_v1/diagnostic_worker.py',
 'handoff/FIXTURES.json':CAMPAIGN/'handoff_reliability_v1/FIXTURES.json',
 'model_pin.json':CAMPAIGN/'model_pin.json',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sources():
    for name in (*REQUIRED_CODE, *FROZEN_METADATA):
        if not (HERE / name).is_file():
            raise FileNotFoundError("Source freeze is not ready: " + name)
    paths = {p.name: p for p in sorted(HERE.glob("*.py")) if p.name != "build_notebook.py"}
    paths.update({name: HERE / name for name in FROZEN_METADATA})
    for name in ('reference_train.json','treatment_train.json','DATA_AUDIT.json','ORACLE_ROUTES.json'):
        path=HERE/'data_frozen'/name
        if not path.is_file():raise FileNotFoundError('Production data freeze missing: '+name)
        paths['data_frozen/'+name]=path
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
    freeze = {"schema_version": 1, "status": "Frozen before any new training or evaluation",
              "frozen_utc": datetime.now(timezone.utc).isoformat(),
              "sha256": hashes, "repository_sources": origins,
              "scope": "Matched H2 reference/bridge replay; nine positive rows changed, 103 immutable",
              "fresh_confirmation_run": False, "source_text_normalization": "Universal newline to LF, matching Git eol=lf; dependency QA hashes separately record original worktree bytes"}
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
    if not re.fullmatch('[0-9a-fA-F]{40}',revision):
        raise ValueError('--source-revision must be a full 40-character commit SHA')
    expected,size=verify_frozen_bundle()
    revision=revision.lower()
    url='https://raw.githubusercontent.com/Farhad-Davaripour/sp_lense/'+revision+'/'+RELATIVE+'/'+BUNDLE_NAME
    heading="""# Research 3 - narrow positive bridge replay

Run setup, mount, restore, actual-data audit, matched fits, and export in order.
One A10080GB; ONE resident model worker, reference then bridge, both from
preserved H2 with fresh optimizers. Only nine positive replay rows change;
B80 plus23 replay rows, including all factual/preference/ordinary rows, stay fixed.
The original checkpoint and known-development panels remain unchanged. Final12
singleton handoff cases are supplied-history DEVELOPMENT DIAGNOSTICS.
No new families or confirmation, control fits, triggers, steering or monitoring.
Models use fictional memory tools only. Drive is unmounted during model work.

Cumulative authorization200units;94.31 previously accounted; refresh actual
credit/rate. Stagecap24 includes2 export/release reserve. Main cap10,800seconds
is reduced by elapsed billed setup time. Export verifies every private file and
then releases the runtime. Existing weights, failures and Research2 remain intact.
"""
    setup="""import gzip,hashlib,json,subprocess,sys,time,uuid
from pathlib import Path,PurePosixPath
from urllib.request import urlopen
SESSION_STARTED=time.monotonic()
BALANCE_AT_LAUNCH=107.82
RATE_AT_LAUNCH=6.77
PRIOR_SPEND=94.31
AUTHORIZED_TOTAL_UNITS=200
STAGE_CAP_UNITS=24
RESERVE_UNITS=2
assert STAGE_CAP_UNITS<=min(BALANCE_AT_LAUNCH,AUTHORIZED_TOTAL_UNITS-PRIOR_SPEND)
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],
                   capture_output=True,text=True,check=True).stdout
assert 'A100' in gpu and int(gpu.split(',')[1].split()[0])>=75000,gpu
ROOT=Path('/content/sp_lense_work')/('qwen38_narrow_bridge_replay_'+
    time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])
ROOT.mkdir(parents=True,exist_ok=False)
SOURCE_REVISION=__REVISION__
SOURCE_URL=__URL__
SOURCE_BUNDLE_SHA256=__HASH__
with urlopen(SOURCE_URL,timeout=120) as response:packed=response.read()
assert len(packed)==__SIZE__ and hashlib.sha256(packed).hexdigest()==SOURCE_BUNDLE_SHA256
payload=json.loads(gzip.decompress(packed));assert payload['schema_version']==1
FILES=payload['files']
for name,content in FILES.items():
    relative=PurePosixPath(name)
    assert not relative.is_absolute() and '..' not in relative.parts
    path=ROOT/'source'/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(content.encode('utf-8'))
for name,digest in json.loads(FILES['SOURCE_FREEZE.json'])['sha256'].items():
    assert hashlib.sha256((ROOT/'source'/name).read_bytes()).hexdigest()==digest,name
(ROOT/'source_bundle.json.gz').write_bytes(packed)
(ROOT/'SOURCE_BUNDLE_DOWNLOAD.json').write_text(json.dumps({'revision':SOURCE_REVISION,
    'url':SOURCE_URL,'sha256':SOURCE_BUNDLE_SHA256,'bytes':len(packed)},indent=2))
subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-input','-r',
                str(ROOT/'source/requirements.txt')],check=True,timeout=900)
sys.path.insert(0,str(ROOT/'source'))
print('NARROW_REPLAY_SOURCE_READY',ROOT,gpu,'authorized remaining',AUTHORIZED_TOTAL_UNITS-PRIOR_SPEND,flush=True)
"""
    setup=(setup.replace('__REVISION__',repr(revision)).replace('__URL__',repr(url))
                .replace('__HASH__',repr(expected)).replace('__SIZE__',str(size)))
    mount="""from google.colab import drive
drive.mount('/content/drive')
print('DRIVE_MOUNT_CONNECTED',flush=True)
"""
    restore="""from restore_inputs import restore_inputs,download_base
MODEL_INPUTS=restore_inputs(ROOT)
drive.flush_and_unmount()
print('DRIVE_UNMOUNTED_BEFORE_MODEL_WORK',flush=True)
BASE_MANIFEST=download_base(ROOT)
"""
    audit="""from prepare_data import prepare
WORKER_ROOT,DATA_AUDIT=prepare(ROOT,MODEL_INPUTS)
print('ACTUAL_DATA_AUDIT',json.dumps({key:value for key,value in DATA_AUDIT.items()
    if key not in ('paired_lengths','content_audit')},indent=2))
print('EXACT_CHANGED_ROWS',json.dumps(DATA_AUDIT['content_audit']['changes'],indent=2))
"""
    launch="""exec(compile((ROOT/'source/launch.py').read_text(),'trusted_narrow_replay.py','exec'))
"""
    close="""import re
for worker in ROOT.rglob('worker.py'):
    assert not subprocess.run(['pgrep','-f',re.escape(str(worker))],capture_output=True,text=True).stdout.strip()
from collect import collect
REVIEW_EVIDENCE=collect(ROOT)
FAST_ROOT=ROOT
EXPORT_ONLY_ROOTS=[ROOT]
RELEASE_RUNTIME_AFTER_EXPORT=True
exec(compile((ROOT/'source/export_private_runs.py').read_text(),'trusted_private_export.py','exec'))
"""
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':NOTEBOOK_NAME},
        'kernelspec':{'name':'python3','display_name':'Python 3'},'accelerator':'GPU',
        'research3_source_revision':revision,'research3_source_bundle_sha256':expected},
        'cells':[cell('markdown',heading,'narrow-v1-00'),cell('code',setup,'narrow-v1-setup'),
                 cell('code',mount,'narrow-v1-mount'),cell('code',restore,'narrow-v1-restore'),
                 cell('code',audit,'narrow-v1-data-audit'),cell('code',launch,'narrow-v1-fit'),
                 cell('code',close,'narrow-v1-export')]}
    target=HERE/NOTEBOOK_NAME
    target.write_text(json.dumps(notebook,indent=1)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'notebook':str(target),'source_revision':revision,'bundle_sha256':expected,
                      'all_outputs_empty':True},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--bundle-only',action='store_true')
    mode.add_argument('--source-revision')
    args=parser.parse_args()
    if args.bundle_only:bundle_only()
    else:build_notebook(args.source_revision)


if __name__=='__main__':main()
