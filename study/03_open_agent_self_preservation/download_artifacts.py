"""Retrieve immutable reproduction inputs; verify bytes before admitting them."""
import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(16*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def fetch(item, output):
    path = Path(output)/item['name']
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if sha(path)==item['sha256']:
            print(json.dumps({'file':item['name'],'verified_existing':True}))
            return
        raise ValueError('Existing file has the wrong hash; preserve it and choose a new output directory')
    temporary = path.with_suffix(path.suffix+'.partial')
    with urllib.request.urlopen(item['url'],timeout=60) as response, temporary.open('wb') as stream:
        for block in iter(lambda:response.read(8*1024*1024),b''):
            stream.write(block)
    if sha(temporary)!=item['sha256'] or ('bytes' in item and temporary.stat().st_size!=item['bytes']):
        raise ValueError('Downloaded artifact identity differs; partial file preserved')
    os.replace(temporary,path)
    print(json.dumps({'file':item['name'],'verified':True}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--adapter',action='store_true')
    mode.add_argument('--training-inputs',action='store_true')
    mode.add_argument('--base',action='store_true')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.training_inputs:
        items=json.loads((HERE/'training/INPUT_SOURCES.json').read_text())['inputs']
    elif args.adapter:
        manifest=json.loads((HERE/'ARTIFACTS.json').read_text())
        prefix='https://github.com/'+manifest['repository']+'/releases/download/'+manifest['release_tag']+'/'
        items=[dict(item,url=prefix+item['name']) for item in manifest['artifacts']]
    else:
        from huggingface_hub import snapshot_download
        pin=json.loads((HERE/'model_pin.json').read_text())
        location=Path(snapshot_download(repo_id=pin['repository'],revision=pin['revision'],
            local_dir=args.output,allow_patterns=[f['name'] for f in pin['files']],token=False))
        for file in pin['files']:
            path=location/file['name']
            if path.stat().st_size!=file['size']:
                raise ValueError('Pinned base file size differs: '+file['name'])
            if file.get('sha256'):
                valid=sha(path)==file['sha256']
            else:
                raw=path.read_bytes()
                valid=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==file['git_blob_id']
            if not valid:
                raise ValueError('Pinned base file hash differs: '+file['name'])
        print(json.dumps({'base_snapshot_verified':True,'files':len(pin['files'])}))
        return
    for item in items:
        fetch(item,args.output)


if __name__=='__main__':
    main()
