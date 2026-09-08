"""Read-only checkpoint byte/header pinning. Never imports providers or weight tensors."""
import hashlib,json,struct
from pathlib import Path
HERE=Path(__file__).resolve().parent
SNAPSHOT=Path('C:/Users/farha/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17')
def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()
def main():
    config=json.loads((SNAPSHOT/'config.json').read_bytes());index=json.loads((SNAPSHOT/'model.safetensors.index.json').read_bytes())
    if config['architectures']!=['Qwen3_5ForConditionalGeneration']:raise ValueError('FULL_CLASS')
    files=['config.json','model.safetensors.index.json']+sorted(set(index['weight_map'].values()))
    header={};records=[]
    for name in files:
        path=SNAPSHOT/name;records.append({'name':name,'bytes':path.stat().st_size,'sha256':sha_file(path)})
        if name.endswith('.safetensors'):
            with path.open('rb') as stream:
                size=struct.unpack('<Q',stream.read(8))[0]
                if not 1<size<1024*1024:raise ValueError('HEADER_CAP')
                part=json.loads(stream.read(size))
            for key,value in part.items():
                if key=='__metadata__':continue
                if key in header or index['weight_map'].get(key)!=name:raise ValueError('HEADER_INDEX')
                header[key]={'shape':value['shape'],'dtype':value['dtype'],'file':name,'data_offsets':value['data_offsets']}
    if set(header)!=set(index['weight_map']):raise ValueError('KEY_COVERAGE')
    result={'snapshot':str(SNAPSHOT),'revision':SNAPSHOT.name,'class':'Qwen3_5ForConditionalGeneration','files':records,
        'checkpoint_keys':header,'config':config,'no_tensor_or_provider_access':True,'real_authorized':False}
    (HERE/'CHECKPOINT.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'files':records,'key_count':len(header),'prefixes':sorted({x.split('.')[0] for x in header}),
        'mtp_keys':[x for x in header if x.startswith('mtp.')],'lm_head':[x for x in header if x.startswith('lm_head')]}))
if __name__=='__main__':main()
