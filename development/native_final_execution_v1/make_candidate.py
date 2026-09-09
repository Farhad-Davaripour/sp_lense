"""Model-free source freeze only; no input or release creation."""
import json,sys
from pathlib import Path
from support import HERE,ROOT,sha,json_bytes,require
def main():
    require(not any(x.split('.')[0] in {'torch','transformers','tokenizers','datasets','pyarrow','safetensors'} for x in sys.modules),'NO_PROVIDERS')
    old=ROOT/'development/native_final_binding_v1'
    previous=json.loads((old/'SOURCE_FREEZE.json').read_bytes())
    external={p['path']:p for p in previous['external_sources']}
    def pin(path):
        p=Path(path);raw=p.read_bytes();external[str(p)]={'path':str(p),'bytes':len(raw),'sha256':sha(raw)}
    for name in previous['source_sha256']:pin(old/name)
    for name in ('SOURCE_FREEZE.json','TEST_WORKFLOW_RESULT.json'):pin(old/name)
    prep=ROOT/'development/native_final_preparation_v1'
    prep_lock=json.loads((prep/'SOURCE_FREEZE.json').read_bytes())
    for name in (*prep_lock['source_sha256'],'SOURCE_FREEZE.json','TEST_RESULTS.json'):pin(prep/name)
    packet=ROOT/'development/native_final_cohort_v1'
    for name in ('BRIEF.md','EMPTY_SCHEMA.json','renderer.py','validate.py'):pin(packet/'author_packet'/name)
    for name in ('PACKET_FREEZE.json','AUTHORING_CONTRACT.md'):pin(packet/name)
    names=sorted([p.name for p in HERE.glob('*.py')]+['CONTRACT.md','CHECKPOINT.json','REUSED_SOURCES.json','OWNED_IDENTITY.json'])
    lock={'schema':'native_final_execution_source_candidate.v1','real_authorized':False,'real_inputs_read':False,
        'source_sha256':{n:sha((HERE/n).read_bytes()) for n in names},'external_sources':list(external.values()),'versions':previous['versions'],
        'upstream_binding_commit':'c27f8ac0f3b27911b16e64e76aa02a87483466c4',
        'preparation_candidate_sha256':sha((prep/'SOURCE_FREEZE.json').read_bytes())}
    (HERE/'SOURCE_FREEZE.json').write_bytes(json_bytes(lock))
    print(json.dumps({'source_freeze_sha256':sha(json_bytes(lock)),'local_sources':len(names),'external_sources':len(external),'real_authorized':False}))
if __name__=='__main__':main()
