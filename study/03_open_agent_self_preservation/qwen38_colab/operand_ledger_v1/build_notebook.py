"""Build a minimal queued frozen-operand ledger notebook and external source bundle."""
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
NOTEBOOK_NAME='Research3_Operand_Ledger_V1.ipynb'
RANK16_BUNDLE_SHA256='bbffc9668ce7c251dcd7de5aad2255d1c02aed2089a9fe66f8689c1614191498'
REQUIRED_CODE=('ledger_worker.py','fixture_build.py','ledger_world.py','restore_ledger.py','collect_ledger.py','worker.py','launch.py')
FROZEN_METADATA=('PROTOCOL.md',)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sources():
    packed=(CAMPAIGN/'narrow_bridge_replay_v1/source_bundle.json.gz').read_bytes()
    if digest(packed)!=RANK16_BUNDLE_SHA256:raise RuntimeError('Immutable initial rank16 source bundle differs')
    old=json.loads(gzip.decompress(packed))['files']
    names=('model_ops.py','fast_inference.py','world.py','memory_world.py','audit.py',
           'activation_capture.py','generation_capture.py','export_private_runs.py','requirements.txt','model_pin.json')
    content={name:old[name] for name in names}
    content['preserved_inputs.py']=old['base_download.py']
    origins={name:'narrow_bridge_replay_v1/source_bundle.json.gz::'+name for name in content}
    for name in REQUIRED_CODE+FROZEN_METADATA:
        if not (HERE/name).is_file():raise FileNotFoundError('Ledger source not ready: '+name)
    for path in sorted(HERE.glob('*.py')):
        if path.name=='build_notebook.py':continue
        if path.name in content:raise RuntimeError('Frozen helper alias collision: '+path.name)
        content[path.name]=path.read_text(encoding='utf-8');origins[path.name]=path.relative_to(REPO).as_posix()
    for name in FROZEN_METADATA:
        content[name]=(HERE/name).read_text(encoding='utf-8');origins[name]=(HERE/name).relative_to(REPO).as_posix()
    for name in ('source_new_development16.json','FIXTURES.json','MODEL_FREE_CHECKS.json'):
        path=HERE/'data_frozen'/name
        if not path.is_file():raise FileNotFoundError('Ledger fixture freeze missing: '+name)
        content['data_frozen/'+name]=path.read_text(encoding='utf-8');origins['data_frozen/'+name]=path.relative_to(REPO).as_posix()
    hashes={name:digest(value.encode('utf-8')) for name,value in content.items()}
    return content,hashes,origins


def bundle_only():
    content, hashes, origins = sources()
    freeze = {"schema_version": 1, "status": "Frozen before any new training or evaluation",
              "frozen_utc": datetime.now(timezone.utc).isoformat(),
              "sha256": hashes, "repository_sources": origins,
              "scope": "Frozen old H2/reference/coverage observed-operand ledger; 48 continuations, zero updates",
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


def build_notebook(revision,parent_root,capacity_root=None):
    if not re.fullmatch('[0-9a-fA-F]{40}',revision):raise ValueError('Full committed source SHA required')
    if not parent_root.startswith('/content/sp_lense_work/'):raise ValueError('Explicit verified Colab parent root required')
    expected,size=verify_frozen_bundle();revision=revision.lower()
    url='https://raw.githubusercontent.com/Farhad-Davaripour/sp_lense/'+revision+'/'+RELATIVE+'/'+BUNDLE_NAME
    heading="""# Research3 - observed-operand ledger diagnostic

Run after the rank32 attempt has exited, whether its parity gate passed or failed.
One sequential frozen-model worker: OLD H2, reference and coverage; no training.
48 continuations at supplied all-read/rejected-old-answer boundaries. The only
changed information is an added current_fragments list of already returned
operands. No sum, new feedback, task instruction, tool, time or family is added.
Batch4 fixed original case order; nine FUTURE turns/1536new tokens after the
supplied boundary,256perturn. Supplied grants are never generated preservation.

Reuse the verified parent base cache/H2. Restore only OLD reference/coverage
from private Drive after previous workers exit, then unmount before models.
Stagecap8 includes2export/release reserve; mainmax3000s reduced by setup,
minimum1800s admitted. Fill actual reconciled spend/balance before allocation.
The whole remaining queue has planned caps43.30 within cumulative200units.
Final export includes unexported parent, rank32 attempt and ledger roots before
runtime release. Keep the previous export cells deferred until this final export.
"""
    capacity_line=('CAPACITY_ROOT=Path('+repr(capacity_root)+')') if capacity_root else "CAPACITY_ROOT=globals().get('CAPACITY_ROOT',globals().get('ROOT'))"
    setup="""import gzip,hashlib,json,subprocess,sys,time,uuid
from pathlib import Path,PurePosixPath
from urllib.request import urlopen
# Codex fills observed current account and cumulative spend after prior stages.
PRIOR_SPEND=None
BALANCE_AT_LAUNCH=None
RATE_AT_LAUNCH=6.77
STAGE_CAP_UNITS=8
RESERVE_UNITS=2
AUTHORIZED_TOTAL_UNITS=200
assert isinstance(PRIOR_SPEND,(int,float)) and isinstance(BALANCE_AT_LAUNCH,(int,float))
assert STAGE_CAP_UNITS<=min(BALANCE_AT_LAUNCH,AUTHORIZED_TOTAL_UNITS-PRIOR_SPEND)
LEDGER_STARTED=time.monotonic()
PARENT_ROOT=Path(__PARENT__)
__CAPACITY_LINE__
assert CAPACITY_ROOT is not None
CAPACITY_ROOT=Path(CAPACITY_ROOT)
assert CAPACITY_ROOT.name.startswith('qwen38_rank32_capacity_'),CAPACITY_ROOT
# Verify exit before mounting Drive or creating another model worker.
for previous in (PARENT_ROOT,CAPACITY_ROOT):
    assert json.loads((previous/'reports/CONTROLLER_RECEIPT.json').read_text())['worker_exited']
    import re
    for worker in previous.rglob('worker.py'):
        assert not subprocess.run(['pgrep','-f',re.escape(str(worker))],capture_output=True,text=True).stdout.strip()
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
assert 'A100' in gpu and int(gpu.split(',')[1].split()[0])>=75000,gpu
ROOT=Path('/content/sp_lense_work')/('qwen38_operand_ledger_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])
ROOT.mkdir(parents=True,exist_ok=False)
SOURCE_REVISION=__REVISION__
SOURCE_URL=__URL__
SOURCE_BUNDLE_SHA256=__HASH__
with urlopen(SOURCE_URL,timeout=120) as response:packed=response.read()
assert len(packed)==__SIZE__ and hashlib.sha256(packed).hexdigest()==SOURCE_BUNDLE_SHA256
payload=json.loads(gzip.decompress(packed));assert payload['schema_version']==1
for name,content in payload['files'].items():
    relative=PurePosixPath(name);assert not relative.is_absolute() and '..' not in relative.parts
    path=ROOT/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content.encode('utf-8'))
for name,expected in json.loads(payload['files']['SOURCE_FREEZE.json'])['sha256'].items():
    assert hashlib.sha256((ROOT/'source'/name).read_bytes()).hexdigest()==expected,name
(ROOT/'source_bundle.json.gz').write_bytes(packed)
(ROOT/'SOURCE_BUNDLE_DOWNLOAD.json').write_text(json.dumps({'revision':SOURCE_REVISION,'url':SOURCE_URL,'sha256':SOURCE_BUNDLE_SHA256,'bytes':len(packed)},indent=2))
sys.path.insert(0,str(ROOT/'source'))
print('LEDGER_FROZEN_SOURCE_READY',ROOT,flush=True)
"""
    setup=(setup.replace('__PARENT__',repr(parent_root)).replace('__CAPACITY_LINE__',capacity_line)
        .replace('__REVISION__',repr(revision)).replace('__URL__',repr(url)).replace('__HASH__',repr(expected)).replace('__SIZE__',str(size)))
    restore="""from google.colab import drive
drive.mount('/content/drive')
from restore_ledger import restore
MODEL_INPUTS,MODEL_PATH,MODEL_TEMPLATE_SHA256=restore(ROOT,PARENT_ROOT,CAPACITY_ROOT)
drive.flush_and_unmount()
print('DRIVE_UNMOUNTED_BEFORE_LEDGER_MODEL_QUERIES',flush=True)
"""
    launch="""exec(compile((ROOT/'source/launch.py').read_text(),'trusted_operand_ledger.py','exec'))
"""
    close="""import re
EXPORT_ROOTS=[PARENT_ROOT,CAPACITY_ROOT,ROOT]
for previous in EXPORT_ROOTS:
    assert json.loads((previous/'reports/CONTROLLER_RECEIPT.json').read_text())['worker_exited']
    for worker in previous.rglob('worker.py'):
        assert not subprocess.run(['pgrep','-f',re.escape(str(worker))],capture_output=True,text=True).stdout.strip()
from collect_ledger import collect
LEDGER_EVIDENCE=collect(ROOT)
FAST_ROOT=ROOT
EXPORT_ONLY_ROOTS=EXPORT_ROOTS
RELEASE_RUNTIME_AFTER_EXPORT=True
exec(compile((ROOT/'source/export_private_runs.py').read_text(),'trusted_all_three_private_export.py','exec'))
"""
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':NOTEBOOK_NAME},'kernelspec':{'name':'python3','display_name':'Python 3'},
        'accelerator':'GPU','research3_source_revision':revision,'research3_source_bundle_sha256':expected},
        'cells':[cell('markdown',heading,'ledger-v1-00'),cell('code',setup,'ledger-v1-setup'),cell('code',restore,'ledger-v1-restore'),
                 cell('code',launch,'ledger-v1-query'),cell('code',close,'ledger-v1-export')]}
    target=HERE/NOTEBOOK_NAME;target.write_text(json.dumps(notebook,indent=1)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'notebook':str(target),'source_revision':revision,'bundle_sha256':expected,'all_outputs_empty':True},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--bundle-only',action='store_true');mode.add_argument('--source-revision')
    parser.add_argument('--parent-root');parser.add_argument('--capacity-root');args=parser.parse_args()
    if args.bundle_only:bundle_only()
    else:
        if not args.parent_root:parser.error('--parent-root is required for the queued notebook')
        build_notebook(args.source_revision,args.parent_root,args.capacity_root)


if __name__=='__main__':main()
