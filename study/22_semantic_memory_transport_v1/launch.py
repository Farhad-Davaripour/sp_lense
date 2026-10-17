"""Freeze the prepared/private source into a fresh run; never overwrite old results."""
import sys,shutil
from pathlib import Path
from common import read,dump,sha
A=Path('C:/Users/farha/Research4/artifacts/semantic_memory_transport_v1');PREP=A/'prepared_20261015';ROOT=A/'run_20261015'
receipt=read(PREP/'PREPARATION_RECEIPT.json');proof=read(A/'PRIVATE_SOURCE_PROOF_20261015.json')
assert proof['provider_size_parent_shared_false_verified'] and receipt['no_model_forwards']
ROOT.mkdir(exist_ok=False);(ROOT/'source').mkdir()
for n,h in receipt['source_sha256'].items():
    assert sha(PREP/'source'/n)==h,n
    shutil.copyfile(PREP/'source'/n,ROOT/'source'/n)
shutil.copyfile(PREP/'PREPARATION_RECEIPT.json',ROOT/'PREPARATION_RECEIPT.json')
shutil.copyfile(A/'PRIVATE_SOURCE_PROOF_20261015.json',ROOT/'PRIVATE_SOURCE_PROOF.json')
files=[p for p in ROOT.rglob('*') if p.is_file()]
dump(ROOT/'EXECUTION_FREEZE.json',{'no_native_forwards_yet':True,'sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files}})
print('R4_SEMANTIC_RUN_READY',ROOT)
