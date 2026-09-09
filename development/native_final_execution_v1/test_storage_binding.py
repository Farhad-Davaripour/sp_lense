"""Only revised source/storage joins; no repeated model or broad workflow suite."""
import copy,json,time
import test_binding as prior
import input_reader,support
from prep_plan import PREPARATION
def main():
    freeze=support.check_freeze();source_sha=support.sha((prior.HERE/'SOURCE_FREEZE.json').read_bytes())
    prep=prior.HERE.parent/'native_final_preparation_v1'
    prior.require(input_reader.PREPARATION_SOURCE_SHA==support.sha((prep/'SOURCE_FREEZE.json').read_bytes()),'EXACT_REVISED_PREPARATION_SHA')
    prior.require((prior.HERE/'prep_plan.py').read_bytes()==(prep/'plan.py').read_bytes(),'EXACT_REVISED_PLAN_BYTES')
    root=prior.HERE/'test_evidence'/('storage_binding_'+str(time.time_ns()));root.mkdir(parents=True)
    vals=prior.fixture(source_sha);bound=prior.bundle(root/'bundle',vals,source_sha)
    prior.require(input_reader.read_bundle(root/'bundle',bound,synthetic=True)==vals[0],'REVISED_24_INPUT_BINDING')
    target=root/'bundle/preparation/RESULT.json';original=target.read_bytes();result=json.loads(original);result['padding']='X'*8192
    target.write_bytes(support.json_bytes(result));changed=copy.deepcopy(bound);changed['preparation_files']['RESULT.json']=support.sha(target.read_bytes())
    try:input_reader.read_bundle(root/'bundle',changed,synthetic=True)
    except ValueError as e:prior.require(str(e)=='INPUT_TERMINAL_FILE_CAP','INDEPENDENT_TERMINAL_LIMIT')
    else:raise RuntimeError('OVERSIZE_TERMINAL_ACCEPTED')
    target.write_bytes(original)
    # Coherently hash-bind oversized fake case receipts: independent total check
    # precedes parsing them and must use the preparation allocation, not16MiB.
    changed=copy.deepcopy(bound);remaining=PREPARATION['preparation_bytes']+1-sum((root/'bundle/preparation'/n).stat().st_size for n in bound['preparation_files'])
    for name in [s['id']+'.json' for s in prior.slots()]:
        if not remaining:break
        p=root/'bundle/preparation'/name;old=p.read_bytes();amount=min(5*1024**2-len(old),remaining)
        p.write_bytes(old+b' '*amount);remaining-=amount;changed['preparation_files'][name]=support.sha(p.read_bytes())
    prior.require(remaining==0,'CONSTRUCT_EXACT_ONE_BYTE_TOTAL_OVERFLOW')
    try:input_reader.read_bundle(root/'bundle',changed,synthetic=True)
    except ValueError as e:prior.require(str(e)=='INPUT_PREPARATION_TOTAL_CAP','INDEPENDENT_PARTITION_LIMIT')
    else:raise RuntimeError('OVERSIZE_PREPARATION_ACCEPTED')
    result={'status':'PASS','groups':3,'source_sha256':source_sha,'preparation_source_sha256':input_reader.PREPARATION_SOURCE_SHA,
        'new_plan_byte_identity':True,'fake24_input_binding':True,'coherent_terminal_overflow_rejected':True,
        'coherent_partition_overflow_rejected':True,'model_calls':0,'tokenizer_calls':0,'actual_cohort_access':False}
    (prior.HERE/'TEST_STORAGE_BINDING_RESULTS.json').write_bytes(support.json_bytes(result));print(json.dumps(result,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
