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
def save_torch(path,value):
    import io,torch
    data=io.BytesIO();torch.save(value,data);Path(path).write_bytes(data.getvalue())
def load_torch(path):
    import io,torch
    return torch.load(io.BytesIO(Path(path).read_bytes()),map_location='cpu',weights_only=True)
def settings(root):
    import yaml
    root=Path(root)
    return yaml.safe_load((root/'config/study.yaml').read_text()),yaml.safe_load((root/'config/data.yaml').read_text())
def cgroup_available():
    try:
        maximum=Path('/sys/fs/cgroup/memory.max').read_text().strip();used=int(Path('/sys/fs/cgroup/memory.current').read_text())
        if maximum=='max':return None
        statistics=dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
        return int(maximum)-used+int(statistics.get('inactive_file','0'))
    except (OSError,ValueError):return None
def assert_namespace(path):
    from .workspace_accounts import artifact_volume,profiles
    value=str(path);roots={artifact_volume(account=name) for name in profiles()['accounts']}
    if not any(value.startswith(root+'/') for root in roots) or '\\' in value or '..' in value.split('/'):
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
        if os.name!='nt' and self.forwards%10==0:
            available=cgroup_available()
            if available is not None and available<536870912:raise MemoryError('Research 4 live memory export reserve reached')
    def count(self):self.check();self.forwards+=1
