"""Output-free private Colab source; no GitHub download or service credential."""
import argparse
import json
from pathlib import Path


def cell(kind,text,identifier):
    value={'cell_type':kind,'metadata':{},'id':identifier,'source':text.splitlines(keepends=True)}
    if kind=='code':value.update(outputs=[],execution_count=None)
    return value


def build(manifest_file,output):
    manifest=json.loads(Path(manifest_file).read_text())
    heading="""# Research3 - Jev first-boundary cached factual pilot

PRIVATE nonsecret source/cache packet only. No key, local client, transport, raw
provider response or environment file is transferred. This stage makes zero Jev
service calls and zero parameter updates. It tests the SAME old coverage adapter
with plain input versus one factual block on the FIRST input copy at each of eight
supplied boundaries. Subsequent decisions use updated plain raw Qwen/tool history.
No refreshed-every-decision assistance or generalization/organism claim.

One A10080GB/high-RAM runtime; one model load reused across both arms. Fixed B4
case order,9futureturns/1536new tokens/256perturn. Source/cache/model/template pins
are checked before queries. Base downloads and hashes once per session, not arm.
Cap8units including2 export/release reserve, mainmax3000s reduced by setup;
minimum1800s admitted. Refresh actual balance/cumulative/rate before running.
All student tools are inert memory operations. Drive unmounts before models.
"""
    setup="""import gzip,hashlib,json,subprocess,sys,time,uuid
from pathlib import Path,PurePosixPath
PILOT_STARTED=time.monotonic()
PRIOR_SPEND=123.78
BALANCE_AT_LAUNCH=78.35
RATE_AT_LAUNCH=6.77
STAGE_CAP_UNITS=8
RESERVE_UNITS=2
AUTHORIZED_TOTAL_UNITS=200
assert STAGE_CAP_UNITS<=min(BALANCE_AT_LAUNCH,AUTHORIZED_TOTAL_UNITS-PRIOR_SPEND)
# Root fills the observed private Drive path after uploading the NONSECRET packet.
PRIVATE_PACKET_PATH=None
EXPECTED_PACKET_SHA256=__SHA__
EXPECTED_PACKET_BYTES=__SIZE__
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
assert 'A100' in gpu and int(gpu.split(',')[1].split()[0])>=75000,gpu
ROOT=Path('/content/sp_lense_work')/('qwen38_jev_boundary_pilot_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])
ROOT.mkdir(parents=True,exist_ok=False)
print('PRIVATE_BOUNDARY_PILOT_SESSION_READY',ROOT,gpu,flush=True)
"""
    setup=setup.replace('__SHA__',repr(manifest['packet_sha256'])).replace('__SIZE__',str(manifest['packet_bytes']))
    restore="""from google.colab import drive
drive.mount('/content/drive')
assert PRIVATE_PACKET_PATH is not None
packet_path=Path(PRIVATE_PACKET_PATH)
assert str(packet_path).startswith('/content/drive/MyDrive/')
packed=packet_path.read_bytes()
assert len(packed)==EXPECTED_PACKET_BYTES and hashlib.sha256(packed).hexdigest()==EXPECTED_PACKET_SHA256
packet=json.loads(gzip.decompress(packed))
assert packet['schema_version']==1 and packet['private_nonsecret_source_and_cache']
for name,text in packet['files'].items():
    relative=PurePosixPath(name);assert not relative.is_absolute() and '..' not in relative.parts
    destination=ROOT/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(text.encode('utf-8'))
for name,expected in json.loads(packet['files']['SOURCE_FREEZE.json'])['sha256'].items():
    assert hashlib.sha256((ROOT/'source'/name).read_bytes()).hexdigest()==expected,name
(ROOT/'PRIVATE_SOURCE_CACHE_PACKET.json.gz').write_bytes(packed)
(ROOT/'PACKET_DOWNLOAD_RECEIPT.json').write_text(json.dumps({'private_source':str(packet_path),
    'sha256':EXPECTED_PACKET_SHA256,'bytes':len(packed),'no_service_calls':True},indent=2))
sys.path.insert(0,str(ROOT/'source'))
from cache_contract import load_cache
BOUNDARY_CACHE=load_cache(ROOT/'source')
from restore_pilot import restore
MODEL_INPUTS=restore(ROOT)
drive.flush_and_unmount()
print('PRIVATE_CACHE_COVERAGE_RESTORED_DRIVE_UNMOUNTED',flush=True)
"""
    base="""subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-input','-r',str(ROOT/'source/requirements.txt')],check=True,timeout=900)
from restore_pilot import download_base
BASE_MANIFEST=download_base(ROOT)
print('ONE_BASE_CACHE_READY_FOR_ALL_GROUPS',flush=True)
"""
    launch="""exec(compile((ROOT/'source/launch.py').read_text(),'trusted_cache_only_boundary_pilot.py','exec'))
"""
    close="""import re
for worker in ROOT.rglob('worker.py'):
    assert not subprocess.run(['pgrep','-f',re.escape(str(worker))],capture_output=True,text=True).stdout.strip()
from collect_pilot import collect
PILOT_EVIDENCE=collect(ROOT)
FAST_ROOT=ROOT
EXPORT_ONLY_ROOTS=[ROOT]
RELEASE_RUNTIME_AFTER_EXPORT=True
exec(compile((ROOT/'source/export_private_runs.py').read_text(),'trusted_private_pilot_export.py','exec'))
"""
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'Research3_Jev_Boundary_Pilot_V1.ipynb'},
        'kernelspec':{'name':'python3','display_name':'Python 3'},'accelerator':'GPU','private_packet_sha256':manifest['packet_sha256']},
        'cells':[cell('markdown',heading,'jev-boundary-00'),cell('code',setup,'jev-boundary-setup'),
            cell('code',restore,'jev-boundary-private-restore'),cell('code',base,'jev-boundary-base'),
            cell('code',launch,'jev-boundary-generate'),cell('code',close,'jev-boundary-export')]}
    output=Path(output)
    if output.exists():raise RuntimeError('Use a new notebook output path; existing artifact is preserved')
    output.write_text(json.dumps(notebook,indent=1)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'notebook':str(output),'outputs_empty':True,'private_packet_sha256':manifest['packet_sha256']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();build(args.manifest,args.output)
