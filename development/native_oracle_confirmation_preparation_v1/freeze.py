"""Pin only reviewed source/metadata; never open a submission or encode text."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(b):return hashlib.sha256(b).hexdigest()
def jb(v):return (json.dumps(v,sort_keys=True,separators=(',',':'))+'\n').encode()
def main():
    assert not any(n.split('.')[0] in {'torch','transformers','tokenizers','pyarrow'} for n in sys.modules)
    old=ROOT/'development/native_final_preparation_v1'
    deps=json.loads((old/'DEPENDENCIES.json').read_bytes())
    packet=ROOT/'development/native_oracle_confirmation_cohort_v2/author_packet'
    files=[]
    for p in deps['files']:
        path=Path(p['path'])
        if 'author_packet' in path.parts:path=packet/path.name
        files.append({'path':str(path),'sha256':sha(path.read_bytes())})
    scope=packet.parent/'ROOT_SUCCESSOR_AUTHORIZATION.md'
    files.append({'path':str(scope),'sha256':sha(scope.read_bytes())})
    deps.update(cohort_packet_root=str(packet),files=files)
    (HERE/'DEPENDENCIES.json').write_bytes(jb(deps))
    names=sorted([p.name for p in HERE.glob('*.py')]+['CONTRACT.md','DEPENDENCIES.json','TOKENIZER_PINS.json'])
    lock={'schema':'fresh_oracle_preparation_source.v1','real_authorized':False,
        'source_sha256':{n:sha((HERE/n).read_bytes()) for n in names},
        'reused_preparation_source_sha256':sha((old/'SOURCE_FREEZE.json').read_bytes())}
    (HERE/'SOURCE_FREEZE.json').write_bytes(jb(lock))
    print(json.dumps({'source_sha256':sha(jb(lock)),'dependencies_sha256':sha(jb(deps)),'files':len(names)}))
if __name__=='__main__':main()
