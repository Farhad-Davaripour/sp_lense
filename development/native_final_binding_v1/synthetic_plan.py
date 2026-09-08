"""Only committed artificial IDs/schema, no real author packet or input lock."""
import ast,hashlib
from pathlib import Path
from support import ROOT,require
SOURCE=ROOT/'diagnostics/fresh_confirmation_workflow_v1/schedule.py'
SOURCE_SHA256='7f18e9fa12cba84f809a059458c97e32339b363c0de0fd125edbaed531b01821'
def build():
    raw=SOURCE.read_bytes();require(hashlib.sha256(raw).hexdigest()==SOURCE_SHA256,'ARTIFICIAL_SCHEDULE_SOURCE')
    source=raw.decode();tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_plan')
    scope={'require':require};exec(compile(ast.get_source_segment(source,node),str(SOURCE),'exec'),scope)
    return scope['build_plan']()

