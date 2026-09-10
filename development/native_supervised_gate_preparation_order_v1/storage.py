"""Prospective shared-output partition; stdlib only, including bootstrap failures."""
import json
COMBINED_BYTES=32*1024**2
OWNER_BYTES=32768
PREPARATION_BYTES=COMBINED_BYTES-OWNER_BYTES
TERMINAL_RESERVE=65536
FILE_BYTES=5*1024**2
TERMINAL_BYTES=8192
def need(ok,code):
    if not ok:raise ValueError(code)
def jb(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def error_fields(error):
    # Never serialize a pathological class name/message, including __str__ errors.
    name=type(error).__name__
    kind=name if len(name)<=64 and name.isascii() and name.isidentifier() else 'Exception'
    try:code=str(error)
    except BaseException:code=''
    code=code if 0<len(code)<=128 and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789' for c in code) else 'PREPARATION_FAILURE'
    return {'error_type':kind,'error_code':code}
class Publisher:
    def __init__(self,out):self.out=out
    def write(self,name,value,*,raw=False,append=False,critical=False):
        need(type(name) is str and '/' not in name and '\\' not in name and ':' not in name and name not in ('','.','..'),'ARTIFACT_NAME')
        data=value if raw else jb(value);need(type(data) is bytes,'ARTIFACT_BYTES');target=self.out/name
        used=sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())
        size=(target.stat().st_size if append and target.exists() else 0)+len(data)
        need(size<=FILE_BYTES,'PREPARATION_FILE_CAP')
        if name=='RESULT.json':need(critical and not append and size<=TERMINAL_BYTES,'BOUNDED_TERMINAL_RESULT')
        need(used+len(data)<=PREPARATION_BYTES-(0 if critical else TERMINAL_RESERVE),'PREPARATION_TOTAL_CAP')
        with target.open('ab' if append else 'xb') as f:
            need(f.write(data)==len(data),'SHORT_WRITE');f.flush()
        import hashlib
        return {'path':name,'bytes':size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
def entry_failure(out,error):
    journal_exists=(out/'operations.jsonl').exists()
    result={'status':'PREPARATION_ENTRY_FAILURE',**error_fields(error),'attempted_operations':0,
        'planned_operations':105,'unrun_operations':105,'model_calls':0,'retry_allowed':False}
    if journal_exists:result.update(attempted_operations=None,unrun_operations=None,operation_counts_status='UNKNOWN_SEE_RETAINED_JOURNAL')
    return Publisher(out).write('RESULT.json',result,critical=True)
