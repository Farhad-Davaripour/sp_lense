"""Byte-pin approved source/metadata only; never inspect an author submission."""
import hashlib,importlib.metadata,importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(raw):return hashlib.sha256(raw).hexdigest()
def need(ok,code):
    if not ok:raise ValueError(code)
def verify(runtime=False):
    data=json.loads((HERE/'DEPENDENCIES.json').read_bytes())
    for pin in data['files']:need(sha(Path(pin['path']).read_bytes())==pin['sha256'],'DEPENDENCY_SOURCE_BYTES')
    if runtime:
        need(sys.version==data['python_version'] and sys.executable==data['python_executable'],'PYTHON_IDENTITY')
        need(sha(Path(sys.executable).read_bytes())==data['python_executable_sha256'],'PYTHON_IMAGE_BYTES')
        need(all(importlib.metadata.version(k)==v for k,v in data['runtime'].items()),'PINNED_RUNTIME_VERSIONS')
    return data
def packet_validator():
    data=json.loads((HERE/'DEPENDENCIES.json').read_bytes());base=Path(data['cohort_packet_root'])
    for name in ('renderer.py','validate.py','EMPTY_SCHEMA.json'):
        path=base/name;pin=next(p for p in data['files'] if Path(p['path'])==path)
        need(sha(path.read_bytes())==pin['sha256'],'PURE_PACKET_SOURCE')
    modules={}
    for name in ('renderer','validate'):
        spec=importlib.util.spec_from_file_location(name,base/(name+'.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);modules[name]=module
    return modules['validate'].validate
