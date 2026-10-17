import hashlib,json,os,time
from pathlib import Path
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def digest(x):return hashlib.sha256(canonical(x).encode('utf-8')).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(path,x):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def settings(root):
    import yaml
    root=Path(root)
    return yaml.safe_load((root/'config/study.yaml').read_text()),yaml.safe_load((root/'config/data.yaml').read_text())
def assert_namespace(path):
    if not str(path).startswith('/Volumes/workspace/research4/artifacts/'):
        raise ValueError('Research 4 writes must stay in its own Volume prefix')
def verify_release(root,expected):
    root=Path(root);r=read(root/'RELEASE.json')
    if r['commit']!=expected:raise ValueError('Deployed source commit differs from the requested version')
    for n,h in r['files'].items():
        if sha(root/n)!=h:raise ValueError('Deployed source hash mismatch: '+n)
    return r
class Budget:
    def __init__(self,config):self.start=time.monotonic();self.config=config;self.forwards=0
    def check(self):
        if time.monotonic()-self.start>self.config['internal_stop_seconds']-self.config['protected_export_seconds']:raise TimeoutError('Research 4 protected export boundary reached')
        if self.forwards>=self.config['maximum_backbone_forwards']:raise TimeoutError('Research 4 forward ceiling reached')
    def count(self):self.check();self.forwards+=1
