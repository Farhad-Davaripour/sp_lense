"""AST-only reuse of synthetic definitions, never old prepared case objects."""
import ast,json,struct
from pathlib import Path
import support,input_reader
from support import sha,json_bytes
from prep_plan import *
path=support.ROOT/'development/native_final_execution_v1/test_binding.py'
tree=ast.parse(path.read_text())
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('fixture','bundle')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'pinned_synthetic_definitions','exec'))
_fixture=fixture;_bundle=bundle
def fixture(source_sha):
    data,lock,records,result,journal=_fixture(source_sha)
    lock['confirmation']=CONFIRMATION
    data['text_lock_canonical_sha256']=sha(json_bytes(lock));data['source_identity']['text_lock_raw_sha256']=sha(json_bytes(lock))
    result['inputs_sha256']=sha(json_bytes(data))
    return data,lock,records,result,journal
def bundle(base,values,source_sha):
    result=_bundle(base,values,source_sha);result['oracle_authority']=input_reader.oracle()
    (base/'SYNTHETIC_BINDING.json').write_bytes(json_bytes(result));return result
