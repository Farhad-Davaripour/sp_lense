"""Small shared source/bytes/owned-process substrate; no model imports."""
import hashlib,json,os,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];MIB=1024**2
ATTEMPT='native_final_execution_attempt_001'
def require(ok,code):
    if not ok:raise ValueError(code)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def json_bytes(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def output():return HERE/'real_evidence'/ATTEMPT
def run_dir():return output()
GROUP_CAPS={'logits':180*248320*4,'rows':180*128*1024,'traces':228*16*1024,'steps':48*32*1024,
    'loader':256*1024,'owned':512*1024,'other':512*1024}
def storage_group(name):
    first=name.split('/')[0]
    return first if first in ('logits','rows','traces','steps','owned') else 'loader' if name=='LOADER_READY.json' else 'other'
def checked_path(base,name):
    require(type(name) is str and '\\' not in name and ':' not in name and all(x not in ('','.','..') for x in name.split('/')),'PATH')
    target=base/name;require(target.resolve().is_relative_to(base.resolve()),'PATH_ESCAPE')
    for p in (base,*target.parents,target):
        if p.exists() and p.is_relative_to(base):require(not p.is_symlink() and not (getattr(p.lstat(),'st_file_attributes',0)&0x400),'PATH_LINK')
    return target
def write_new(name,value,*,raw=False,critical=False):
    data=value if raw else json_bytes(value);require(type(data) is bytes and len(data)<=5*MIB,'FILE_CAP')
    base=output();target=checked_path(base,name)
    group=storage_group(name)
    per_file={'logits':248320*4,'rows':128*1024,'traces':16*1024,'steps':32*1024}.get(group,GROUP_CAPS[group])
    require(len(data)<=per_file,'ARTIFACT_FILE_CAP')
    group_used=sum(p.stat().st_size for p in base.rglob('*') if p.is_file() and storage_group(p.relative_to(base).as_posix())==group)
    require(group_used+len(data)<=GROUP_CAPS[group],'ARTIFACT_GROUP_CAP')
    used=sum(p.stat().st_size for p in base.rglob('*') if p.is_file())
    require(used+len(data)<=288*MIB-(0 if critical else 65536),'OUTPUT_CAP_CLOSEOUT_RESERVE')
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as stream:
        require(stream.write(data)==len(data),'SHORT_WRITE');stream.flush();os.fsync(stream.fileno())
    require(target.read_bytes()==data,'WRITE_ACK')
    return {'path':name,'bytes':len(data),'sha256':sha(data)}
def check_freeze():
    lock=json.loads((HERE/'SOURCE_FREEZE.json').read_bytes())
    for name,digest in lock['source_sha256'].items():require(sha((HERE/name).read_bytes())==digest,'SOURCE_FREEZE')
    for pin in lock['external_sources']:
        require(sha((ROOT/pin['path']).read_bytes())==pin['sha256'],'EXTERNAL_SOURCE')
    return lock
class Sources:
    def read(self,commit,path):
        pins=json.loads((HERE/'REUSED_SOURCES.json').read_bytes())
        pin=pins[commit+':'+path];raw=(ROOT/path).read_bytes()
        require(len(raw)==pin['bytes'] and sha(raw)==pin['sha256'],'REUSED_SOURCE')
        return raw
    def load(self,name,commit,path):
        m=types.ModuleType(name);m.__file__=str(ROOT/path);sys.modules[name]=m
        exec(compile(self.read(commit,path),m.__file__,'exec'),m.__dict__);return m
SOURCES=Sources();SCIENCE_COMMIT='1d4cc39eba5fb7f987649a05f71247061f005d02';SCIENCE='diagnostics/semantic_editor_f03_v2_first_C_v2/'
def ownership():
    native=SOURCES.load('native',SCIENCE_COMMIT,SCIENCE+'native.py')
    SOURCES.read(SCIENCE_COMMIT,'diagnostics/windows_worker_identity_probe_v1/probe.py')
    raw=SOURCES.read(SCIENCE_COMMIT,SCIENCE+'owned.py').decode()
    raw=raw.replace('"scope":"FAKE_WORK_ONLY"','"scope":"RETAINED_PROCESS_ONLY"').replace('"fake_only":True','"ownership_not_model_authority":True')
    anchor='self.stop_deadline=self.stop_started_at+3.';require(raw.count(anchor)==1,'OWNED_BOUNDARY')
    raw=raw.replace(anchor,'self.stop_deadline=min(self.stop_started_at+3.,getattr(self,"cleanup_bound",self.stop_started_at+3.))').replace('str(error)','"OWNED_NATIVE_ERROR"')
    owned=types.ModuleType('owned');owned.__file__=str(HERE/'owned_core_bound.py');sys.modules['owned']=owned
    exec(compile(raw,owned.__file__,'exec'),owned.__dict__);return native,owned
def pinned(namespace,name):
    raw=SOURCES.read('f31d0e4f8fd5136eb38df01201f7032c9b82dc61','diagnostics/fresh_confirmation_production_v2/'+name+'.py')
    exec(compile(raw,namespace['__file__'],'exec'),namespace)
