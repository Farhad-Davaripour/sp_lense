"""Reuse exact artificial marker/tokenizer definitions; no provider or prose read."""
import ast,copy,json,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
class Deny:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in {'torch','transformers','tokenizers','pyarrow','datasets','safetensors'}:raise RuntimeError('NO_REAL_PROVIDER')
sys.meta_path.insert(0,Deny())
from dependencies import need,sha,verify,packet_validator
from plan import *
from prepare_core import execute,jb,verify_prospective_bindings
tree=ast.parse((ROOT/'development/native_final_preparation_v1/test_prepare.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('fixture','Clock','FakeTokenizer')],type_ignores=[]),'approved_fake_definitions','exec'))
def main():
    verify();root=HERE/'test_evidence'/str(time.time_ns());root.mkdir(parents=True);checks=[]
    for mode in ('success','at320','oversize','prefix','old_arm'):
        lock=fixture();lock['confirmation']=copy.deepcopy(CONFIRMATION)
        if mode=='old_arm':lock.pop('confirmation')
        clock=Clock();tok=FakeTokenizer(lock,mode,clock);calls=[]
        def factory():calls.append(1);return tok
        result=execute(lock,root/mode,factory,180.,allow_synthetic=True,template_sha256=sha(tok.chat_template.encode()),clock=clock)
        need(result['status']==('PASS' if mode in ('success','at320') else 'FAIL'),'EXPECTED_'+mode)
        need(result['attempted_operations']==len(calls)+len(tok.calls),'COUNTED_CALLS')
        if mode in ('success','at320'):need(result['completed_operations']==313 and result['completed_cases']==24,'FULL24_313')
        if mode=='old_arm':need(not calls,'OLD_ARM_BEFORE_FACTORY')
        checks.append({'case':mode,'status':'PASS','result':result})
    bound={}
    for key,ns,field in (('final_execution_binding','native_oracle_confirmation_execution_v1','source_sha256'),
        ('preparation_owner_binding','native_oracle_confirmation_preparation_owner_v1','files')):
        folder=root/'development'/ns;folder.mkdir(parents=True);raw=jb({field:{},'external_sources':[]})
        (folder/'SOURCE_FREEZE.json').write_bytes(raw)
        bound[key]={'namespace':'development/'+ns,'source_freeze_sha256':sha(raw)}
    bound['final_execution_binding']['preparation_source_freeze_sha256']=sha((HERE/'SOURCE_FREEZE.json').read_bytes())
    verify_prospective_bindings(bound,root)
    for mutation in ('source_hash','old_namespace'):
        bad=copy.deepcopy(bound);bad['final_execution_binding']['source_freeze_sha256' if mutation=='source_hash' else 'namespace']='old'
        try:verify_prospective_bindings(bad,root)
        except ValueError:pass
        else:raise RuntimeError('BAD_SOURCE_BINDING_ACCEPTED')
        checks.append({'case':mutation,'status':'PASS'})
    result={'status':'PASS','groups':checks,'group_count':len(checks),'real_tokenizer_calls':0,'model_calls':0,
        'source_sha256':sha((HERE/'SOURCE_FREEZE.json').read_bytes())}
    (HERE/'TEST_RESULTS.json').write_bytes(jb(result));print(json.dumps({'status':'PASS','groups':len(checks)}))
if __name__=='__main__':main()
