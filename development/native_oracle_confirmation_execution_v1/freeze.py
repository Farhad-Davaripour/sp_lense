"""Freeze reviewed reused source plus new bindings; no future input dependency."""
import json
from pathlib import Path
from support import HERE,ROOT,sha,json_bytes
def main():
    old=ROOT/'development/native_final_execution_v1';prior=json.loads((old/'SOURCE_FREEZE.json').read_bytes())
    external={p['path']:p for p in prior['external_sources']}
    def pin(path):
        raw=path.read_bytes();external[str(path)]={'path':str(path),'bytes':len(raw),'sha256':sha(raw)}
    for folder in ('native_final_execution_v1','native_oracle_family_panel_v1','native_oracle_confirmation_preparation_v1'):
        base=ROOT/'development'/folder;lock=json.loads((base/'SOURCE_FREEZE.json').read_bytes())
        for n in lock['source_sha256']:
            if n=='inputs.json':continue
            pin(base/n)
        pin(base/'SOURCE_FREEZE.json')
    pin(ROOT/'development/native_oracle_confirmation_cohort_v1/ROOT_CONFIRMATION_SCOPE.md')
    names=sorted([p.name for p in HERE.glob('*.py')]+['CONTRACT.md','CHECKPOINT.json','REUSED_SOURCES.json','OWNED_IDENTITY.json'])
    lock={'schema':'fresh_oracle_execution_source.v1','real_authorized':False,'future_input_bytes_pinned':False,
        'source_sha256':{n:sha((HERE/n).read_bytes()) for n in names},'external_sources':list(external.values()),'versions':prior['versions']}
    (HERE/'SOURCE_FREEZE.json').write_bytes(json_bytes(lock));print(json.dumps({'source_sha256':sha(json_bytes(lock)),'local':len(names),'external':len(external)}))
if __name__=='__main__':main()
