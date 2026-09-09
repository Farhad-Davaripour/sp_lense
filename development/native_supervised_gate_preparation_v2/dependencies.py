"""Finite source pins and inherited tokenizer runtime identity; no numeric payloads."""
import hashlib,importlib.metadata,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def sha(raw):return hashlib.sha256(raw).hexdigest()
def need(ok,code):
    if not ok:raise ValueError(code)
def verify(runtime=False):
    data=json.loads((HERE/'DEPENDENCIES.json').read_bytes())
    for pin in data['files']:
        need(sha((ROOT/pin['path']).read_bytes())==pin['sha256'],'DEPENDENCY_SOURCE_BYTES')
    if runtime:
        need(sys.version==data['python_version'] and sys.executable==data['python_executable'],'PYTHON_IDENTITY')
        need(sha(Path(sys.executable).read_bytes())==data['python_executable_sha256'],'PYTHON_IMAGE_BYTES')
        need(all(importlib.metadata.version(k)==v for k,v in data['runtime'].items()),'PINNED_RUNTIME_VERSIONS')
    return data
def verify_local_source_freeze(expected=None):
    raw=(HERE/'SOURCE_FREEZE.json').read_bytes()
    if expected is not None:need(sha(raw)==expected,'PREPARATION_SOURCE_FREEZE_BYTES')
    frozen=json.loads(raw)
    need(frozen['real_authorized'] is False,'NO_EMBEDDED_REAL_RELEASE')
    for name,digest in frozen['source_sha256'].items():
        need('/' not in name and '\\' not in name and ':' not in name,'LOCAL_SOURCE_NAME')
        need(sha((HERE/name).read_bytes())==digest,'PREPARATION_SOURCE_BYTES')
    verify()
    return sha(raw)
def packet_validator():
    from validate import validate
    return validate
