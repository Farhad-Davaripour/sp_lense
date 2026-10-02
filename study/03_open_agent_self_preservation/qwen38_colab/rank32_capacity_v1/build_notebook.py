"""Build a queued rank32 capacity stage from the immutable rank16 source bundle."""
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
NOTEBOOK_NAME='Research3_Rank32_Capacity_V1.ipynb'
RANK16_BUNDLE_SHA256='bbffc9668ce7c251dcd7de5aad2255d1c02aed2089a9fe66f8689c1614191498'
REQUIRED_CODE=('capacity_expand.py','capacity_worker.py','prepare_capacity.py','collect_capacity.py','worker.py','launch.py')
FROZEN_METADATA=('PROTOCOL.md','PREQUERY_CHANGELOG.md')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sources():
    parent_bundle=CAMPAIGN/'narrow_bridge_replay_v1/source_bundle.json.gz'
    packed=parent_bundle.read_bytes()
    if digest(packed)!=RANK16_BUNDLE_SHA256:raise RuntimeError('Frozen rank16 source bundle changed')
    parent=json.loads(gzip.decompress(packed))['files']
    content={name:value for name,value in parent.items() if name not in
             ('SOURCE_FREEZE.json','PROTOCOL.md','worker.py','launch.py','collect.py','prepare_data.py','replay_worker.py')}
    content['narrow_replay.py']=parent['replay_worker.py']
    content['rank16/PROTOCOL.md']=parent['PROTOCOL.md']
    content['rank16/SOURCE_FREEZE.json']=parent['SOURCE_FREEZE.json']
    origins={name:'narrow_bridge_replay_v1/source_bundle.json.gz::'+name for name in content}
    origins['narrow_replay.py']='narrow_bridge_replay_v1/source_bundle.json.gz::replay_worker.py'
    origins['rank16/PROTOCOL.md']='narrow_bridge_replay_v1/source_bundle.json.gz::PROTOCOL.md'
    origins['rank16/SOURCE_FREEZE.json']='narrow_bridge_replay_v1/source_bundle.json.gz::SOURCE_FREEZE.json'
    for name in REQUIRED_CODE+FROZEN_METADATA:
        if not (HERE/name).is_file():raise FileNotFoundError('Capacity source not ready: '+name)
    for path in sorted(HERE.glob('*.py')):
        if path.name=='build_notebook.py':continue
        content[path.name]=path.read_text(encoding='utf-8');origins[path.name]=path.relative_to(REPO).as_posix()
    for name in FROZEN_METADATA:
        content[name]=(HERE/name).read_text(encoding='utf-8');origins[name]=(HERE/name).relative_to(REPO).as_posix()
    hashes={name:digest(value.encode('utf-8')) for name,value in content.items()}
    return content,hashes,origins


def bundle_only():
    content, hashes, origins = sources()
    freeze = {"schema_version": 1, "status": "Frozen before any new training or evaluation",
              "frozen_utc": datetime.now(timezone.utc).isoformat(),
              "sha256": hashes, "repository_sources": origins,
              "scope": "Controlled rank32 function-preserved H2 contrast; exact bridge data/dose unchanged",
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


def build_notebook(revision,parent_root):
    if not re.fullmatch('[0-9a-fA-F]{40}',revision):raise ValueError('Full committed source SHA required')
    if not parent_root.startswith('/content/sp_lense_work/'):raise ValueError('Explicit verified Colab parent run root required')
    expected,size=verify_frozen_bundle();revision=revision.lower()
    url='https://raw.githubusercontent.com/Farhad-Davaripour/sp_lense/'+revision+'/'+RELATIVE+'/'+BUNDLE_NAME
    heading="""# Research3 - controlled rank32 capacity contrast

Run only after the exact-dose rank16 bridge pair is complete, exported or safely
preserved, with its verified base cache and original H2 inputs still available.
This stage starts from ORIGINAL H2, never the trained rank16 bridge. It embeds
A16/B16 into A32/B32 with zero added B columns and unchanged scaling2, then
requires pre-fit numerical and four-case action/state/outcome parity.

The exact bridge112 data, loss weighting, dose, order, learning rate, decode,
checkpoint schedule and qualification gates remain fixed. No other sweep or
new scenario families. First trajectories remain development diagnostics.
One sequential A10080GB worker. Additional stagecap12units including2 export/
release reserve; combined remaining rank16+rank32 caps35.30 within the cumulative
200-unit authorization. Set observed account/cumulative spend after rank16.
No laptop neural runs or real model filesystem/network/persistence tools.
"""
    setup="""import gzip,hashlib,json,subprocess,sys,time,uuid
from pathlib import Path,PurePosixPath
from urllib.request import urlopen
# Codex fills current observed balance and reconciled spend after rank16.
PRIOR_SPEND=None
BALANCE_AT_LAUNCH=None
RATE_AT_LAUNCH=6.77
STAGE_CAP_UNITS=12
RESERVE_UNITS=2
AUTHORIZED_TOTAL_UNITS=200
assert isinstance(PRIOR_SPEND,(int,float)) and isinstance(BALANCE_AT_LAUNCH,(int,float))
assert STAGE_CAP_UNITS<=min(BALANCE_AT_LAUNCH,AUTHORIZED_TOTAL_UNITS-PRIOR_SPEND)
CAPACITY_STARTED=time.monotonic()
PARENT_ROOT=Path(__PARENT__)
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
assert 'A100' in gpu and int(gpu.split(',')[1].split()[0])>=75000,gpu
ROOT=Path('/content/sp_lense_work')/('qwen38_rank32_capacity_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])
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
print('CAPACITY_SOURCE_READY',ROOT,'parent',PARENT_ROOT,flush=True)
"""
    setup=(setup.replace('__PARENT__',repr(parent_root)).replace('__REVISION__',repr(revision))
        .replace('__URL__',repr(url)).replace('__HASH__',repr(expected)).replace('__SIZE__',str(size)))
    launch="""exec(compile((ROOT/'source/launch.py').read_text(),'trusted_rank32_capacity.py','exec'))
"""
    close="""import re
for worker in ROOT.rglob('worker.py'):
    assert not subprocess.run(['pgrep','-f',re.escape(str(worker))],capture_output=True,text=True).stdout.strip()
from collect_capacity import collect
CAPACITY_EVIDENCE=collect(ROOT)
FAST_ROOT=ROOT
EXPORT_ONLY_ROOTS=[PARENT_ROOT,ROOT]
RELEASE_RUNTIME_AFTER_EXPORT=True
exec(compile((ROOT/'source/export_private_runs.py').read_text(),'trusted_capacity_export.py','exec'))
"""
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':NOTEBOOK_NAME},'kernelspec':{'name':'python3','display_name':'Python 3'},
        'accelerator':'GPU','research3_source_revision':revision,'research3_source_bundle_sha256':expected},
        'cells':[cell('markdown',heading,'capacity-v1-00'),cell('code',setup,'capacity-v1-setup'),
                 cell('code',launch,'capacity-v1-fit'),cell('code',close,'capacity-v1-export')]}
    target=HERE/NOTEBOOK_NAME;target.write_text(json.dumps(notebook,indent=1)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'notebook':str(target),'source_revision':revision,'bundle_sha256':expected,'all_outputs_empty':True},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--bundle-only',action='store_true');mode.add_argument('--source-revision')
    parser.add_argument('--parent-root');args=parser.parse_args()
    if args.bundle_only:bundle_only()
    else:
        if not args.parent_root:parser.error('--parent-root is required for the queued notebook')
        build_notebook(args.source_revision,args.parent_root)


if __name__=='__main__':main()
