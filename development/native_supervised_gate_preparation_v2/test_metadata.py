"""One marker-only namespace bridge; no actual tokenizer, owner, or model run."""
import copy,json,sys,time
from unittest.mock import patch
import test_prepare as fake
from dependencies import HERE,ROOT,need,sha,verify_local_source_freeze
from plan import COHORT_IDENTITY,MODEL,PREPARATION,STUDY,operations,slots
from prepare_core import execute,jb
import prepare_reader as reader
from storage import Publisher
OLD_FREEZE_SHA='7d2fa6660c1d240c2541b6af93a14ab3d6245856a84cf265874d1234f6e5cf92'
def main():
    old=ROOT/'development/native_supervised_gate_preparation_v1'
    raw=(old/'SOURCE_FREEZE.json').read_bytes();need(sha(raw)==OLD_FREEZE_SHA,'CLOSED_V1_SOURCE_FREEZE')
    previous=json.loads(raw);comparison=[]
    for name,digest in previous['source_sha256'].items():
        original=(old/name).read_bytes();need(sha(original)==digest,'CLOSED_V1_SOURCE_BYTES')
        expected=original.replace(b'development/native_supervised_gate_preparation_v1',b'development/native_supervised_gate_preparation_v2')
        expected=expected.replace(b'development/native_supervised_gate_evaluation_v1',b'development/native_supervised_gate_evaluation_v2')
        current=(HERE/name).read_bytes();need(current==expected,'ONLY_NAMESPACE_SUBSTITUTIONS')
        comparison.append({'file':name,'old_sha256':digest,'new_sha256':sha(current),'unchanged':current==original})
    need(len(operations())==PREPARATION['expected_operations']==209 and len(slots())==16,'UNCHANGED209_16')
    need(COHORT_IDENTITY['cohort_namespace']=='development/native_supervised_gate_cohort_v1'
        and COHORT_IDENTITY['submission_sha256']=='9b06eda1f293acd764a65cb742d9f2003ab980271065a927a49a8dbc4cb4902e','UNCHANGED_COHORT')
    source_sha=verify_local_source_freeze();lock=fake.fixture()
    need(lock['final_execution_binding']=={'namespace':'development/native_supervised_gate_evaluation_v2',
        'source_freeze_sha256':'f'*64,'preparation_source_freeze_sha256':source_sha},'V2_LAZY_SOURCE_JOIN')
    bundle=HERE/'fixtures/metadata_bridge';bundle.mkdir(parents=True,exist_ok=False);pub=Publisher(bundle)
    text=pub.write('TEXT_LOCK.json',lock)
    identity={'text_lock_raw_sha256':text['sha256'],'source_freeze_sha256':source_sha,
        'dependencies_sha256':reader.PREPARATION_DEPENDENCIES_SHA,'tokenizer_pins_sha256':reader.TOKENIZER_PINS_SHA}
    clock=fake.Clock();tokenizer=fake.FakeTokenizer(lock,clock=clock);loads=[]
    def factory():loads.append(1);return tokenizer
    fake_template_sha=sha(tokenizer.chat_template.encode())
    result=execute(lock,bundle/'preparation',factory,175.,allow_synthetic=True,
        template_sha256=fake_template_sha,clock=clock,identity=identity)
    need(result['status']=='PASS' and result['completed_operations']==209 and result['completed_cases']==16,'ONE_FAKE_BRIDGE_SUCCESS')
    need(len(loads)==1 and len(tokenizer.calls)==208,'EXACT_FAKE_OPERATION_COUNT')
    pins={name:sha((bundle/'preparation'/name).read_bytes()) for name in reader.artifact_names()}
    closure={'schema':'root_preparation_closure.v1','synthetic_only':True,'quiescent':True,'exit_code':0,
        'timed_out':False,'one_shot':True,'preparation_result_sha256':pins['RESULT.json'],'text_lock_sha256':text['sha256']}
    cp=pub.write('PREPARATION_CLOSURE.json',closure)
    release={'approved':False,'scope':'SYNTHETIC_TEST_ONLY','text_lock_sha256':text['sha256'],
        'preparation_files':pins,'inputs_sha256':pins['inputs.json'],'preparation_closure_sha256':cp['sha256'],
        'source_freeze_sha256':'f'*64,'fake_template_sha256':fake_template_sha}
    pub.write('SYNTHETIC_RELEASE.json',release)
    with patch.object(reader,'TEMPLATE_SHA256',fake_template_sha):
        data=reader.read_bundle(bundle,release,synthetic=True)
        records=[json.loads((bundle/'preparation'/(s['id']+'.json')).read_bytes()) for s in slots()]
        journal=[json.loads(line) for line in (bundle/'preparation/operations.jsonl').read_bytes().splitlines()]
        bad=copy.deepcopy(lock);bad['final_execution_binding']['namespace']='development/native_supervised_gate_evaluation_v1'
        try:reader.validate(data,bad,records,result,journal,'f'*64,sha(jb(bad)),synthetic=True)
        except ValueError as error:need(str(error)=='ADAPTER_FIXED_BEFORE_TOKENIZATION','OLD_NAMESPACE_REJECTION')
        else:raise ValueError('OLD_NAMESPACE_ACCEPTED')
    need(not any(name.split('.')[0] in fake.BLOCKED for name in sys.modules),'NO_REAL_PROVIDERS')
    total=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
    need(total<16*1024**2 and max(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<=5*1024**2,'ENGINEERING_OUTPUT_CAP')
    report={'status':'PASS','scope':'METADATA_ONLY_V2','source_freeze_sha256':source_sha,'closed_v1_source_freeze_sha256':OLD_FREEZE_SHA,
        'comparisons':comparison,'comparison_count':len(comparison),'unchanged_science':True,'planned_operations':209,
        'synthetic_runs':1,'synthetic_case_count':len(data['cases']),'real_tokenizer_calls':0,'real_model_calls':0,
        'actual_preparation_authorized':False,'actual_execution_authorized':False,'old_namespace_rejected':True,
        'bundle':'fixtures/metadata_bridge','bundle_release_sha256':sha(jb(release)),'retained_bytes_before_report':total}
    Publisher(HERE).write('METADATA_TEST_RESULTS.json',report)
    print(json.dumps({'status':'PASS','comparison_count':len(comparison),'synthetic_runs':1,
        'completed_operations':209,'source_freeze_sha256':source_sha,'bundle':str(bundle)},sort_keys=True))
if __name__=='__main__':main()
