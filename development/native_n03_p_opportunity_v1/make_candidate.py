"""Freeze this thin development delta and its reviewed upstream pins; no release."""
import json,sys
from support import HERE,ROOT,require,sha,json_bytes
def main():
    require(not any(n.split('.')[0] in ('torch','transformers','tokenizers','pyarrow','datasets','safetensors') for n in sys.modules),'MODEL_FREE_FREEZE')
    old=ROOT/'development/native_oracle_family_panel_v1';previous=json.loads((old/'SOURCE_FREEZE.json').read_bytes())
    external={p['path']:p for p in previous['external_sources']}
    def pin(path):
        raw=path.read_bytes();name=path.relative_to(ROOT).as_posix()
        external[name]={'path':name,'bytes':len(raw),'sha256':sha(raw)}
    for name in (*previous['source_sha256'],'SOURCE_FREEZE.json','ROOT_NEXT_P_OPPORTUNITY_SCOPE.md','TEST_RESULTS.json','INDEPENDENT_REVIEW.md'):
        pin(old/name)
    from input_reader import SOURCE
    pin(ROOT/SOURCE)
    names=sorted([p.name for p in HERE.glob('*.py')]+['inputs.json','CONTRACT.md','CHECKPOINT.json','REUSED_SOURCES.json','OWNED_IDENTITY.json'])
    lock={'schema':'native_n03_p_opportunity_source_candidate.v1','real_authorized':False,
        'outcome_informed_development':True,'learned_gate_success_claimed':False,
        'source_sha256':{n:sha((HERE/n).read_bytes()) for n in names},'external_sources':list(external.values()),
        'versions':previous['versions'],'upstream_source_sha256':sha((old/'SOURCE_FREEZE.json').read_bytes())}
    raw=json_bytes(lock);(HERE/'SOURCE_FREEZE.json').write_bytes(raw)
    print(json.dumps({'source_freeze_sha256':sha(raw),'local_sources':len(names),'external_sources':len(external),'real_authorized':False}))
if __name__=='__main__':main()
