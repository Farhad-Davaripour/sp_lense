"""Metadata-only v2 join checks, no tokenizer/model/provider or real prose access."""
import ast,copy,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
class Deny:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in {'torch','transformers','tokenizers','pyarrow','datasets','safetensors'}:raise RuntimeError('NO_REAL_PROVIDER')
sys.meta_path.insert(0,Deny())
from dependencies import need,sha,verify,packet_validator
from plan import MODEL,STUDY,CONFIRMATION,slots
from prepare_core import jb,validate_text_lock,verify_prospective_bindings
tree=ast.parse((ROOT/'development/native_final_preparation_v1/test_prepare.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fixture'],type_ignores=[]),'pinned_marker_fixture','exec'))
def main():
    reports=[];deps=verify()
    expected={'arm':'fresh_oracle_confirmation.v1','cohort_namespace':'development/native_oracle_confirmation_cohort_v2',
        'root_scope_sha256':'d73ef37a668c5dcd5653277acce89e4991de0ffb8de9b0d46384d9f209edc461'}
    need(CONFIRMATION==expected,'EXACT_ROOT_V2_METADATA')
    packet=ROOT/expected['cohort_namespace']/'author_packet'
    need(Path(deps['cohort_packet_root'])==packet,'V2_PACKET_PATH')
    need(sha((packet/'BRIEF.md').read_bytes())=='9d25b5c7e19f25ba994d11f1d6160f0048763b67da5f6f84228f43796451a937','ROOT_BRIEF_PIN')
    specimen=fixture();specimen['confirmation']=copy.deepcopy(CONFIRMATION)
    need(validate_text_lock(specimen,True)['status']=='MECHANICAL_PASS','MATCHED_V2_MARKERS')
    reports.append('matched_v2_packet_and_prospective_metadata')
    def rejects(name,fn):
        try:fn()
        except (ValueError,KeyError,OSError):reports.append(name)
        else:raise RuntimeError('NOT_REJECTED_'+name)
    for name,change in (('old_v1_namespace',{'cohort_namespace':'development/native_oracle_confirmation_cohort_v1'}),
        ('old_root_scope',{'root_scope_sha256':'fc88742341e48918f39a5817caad12824c75234b100188f71ec8f51d0c6dc087'}),
        ('wrong_arm',{'arm':'other'})):
        bad=copy.deepcopy(specimen);bad['confirmation'].update(change)
        rejects(name,lambda:validate_text_lock(bad,True))
    run=ROOT/'development/native_oracle_confirmation_execution_v1';owner=ROOT/'development/native_oracle_confirmation_preparation_owner_v1'
    sys.path.insert(0,str(run))
    import input_reader,prep_plan
    from fake_bindings import fixture as input_fixture
    need((run/'prep_plan.py').read_bytes()==(HERE/'plan.py').read_bytes(),'EXACT_SHARED_PLAN')
    prep_sha=sha((HERE/'SOURCE_FREEZE.json').read_bytes());exec_sha=sha((run/'SOURCE_FREEZE.json').read_bytes());owner_sha=sha((owner/'SOURCE_FREEZE.json').read_bytes())
    need(input_reader.PREPARATION_SOURCE_SHA==prep_sha and input_reader.PREPARATION_DEPENDENCIES_SHA==sha((HERE/'DEPENDENCIES.json').read_bytes()),'DOWNSTREAM_PREP_IDENTITY')
    bindings={'final_execution_binding':input_reader.execution_binding(exec_sha),
        'preparation_owner_binding':{'namespace':'development/native_oracle_confirmation_preparation_owner_v1','source_freeze_sha256':owner_sha}}
    verify_prospective_bindings(bindings);reports.append('actual_source_chain_prospectively_bound')
    values=input_fixture(exec_sha)
    need(input_reader.validate(*values,exec_sha,sha(jb(values[1])),synthetic=True)==values[0],'MATCHED24_FRESH_INTERFACE')
    reports.append('matched_v2_artificial24_interface')
    for name,mutate in (('prepared_old_cohort',lambda v:v[1]['confirmation'].update(cohort_namespace='development/native_oracle_confirmation_cohort_v1')),
        ('prepared_wrong_scope',lambda v:v[1]['confirmation'].update(root_scope_sha256='0'*64)),
        ('prepared_source_substitution',lambda v:v[0]['source_identity'].update(source_freeze_sha256='0'*64))):
        bad=copy.deepcopy(values);mutate(bad)
        rejects(name,lambda:input_reader.validate(*bad,exec_sha,sha(jb(bad[1])),synthetic=True))
    oracle=input_reader.oracle();need(sum(oracle['applicability'].values())==6 and len(oracle['applicability'])==24,'UNCHANGED_ORACLE_MAP')
    old=copy.deepcopy(oracle);old['root_scope_sha256']='fc88742341e48918f39a5817caad12824c75234b100188f71ec8f51d0c6dc087'
    rejects('old_oracle_authority',lambda:input_reader.require_oracle(old))
    names=['workflow.py','audit_saved.py','science.py','core.py','receiver.py','loader.py','counts.py','setup_budget.py','support.py','production_run.py','authority.py','entry.py','launch.py','owned_production.py','test_delta.py','TEST_RESULTS.json','INDEPENDENT_REVIEW.md']
    for name in names:
        relative=(run/name).relative_to(ROOT).as_posix()
        previous=subprocess.check_output(['git','show','77f26c1190fa9a07c675725a1386d4daff13d74c:'+relative],cwd=ROOT)
        need((run/name).read_bytes()==previous,'UNCHANGED_REVIEWED_SCIENCE_PROOF_'+name)
    reports.append('17_unchanged_scientific_workflow_proof_files')
    result={'status':'PASS','group_count':len(reports),'groups':reports,'sources':{'preparation':prep_sha,'execution':exec_sha,'owner':owner_sha},
        'prior_engineering_commit':'77f26c1190fa9a07c675725a1386d4daff13d74c','full_workflow_rerun':False,
        'real_tokenizer_calls':0,'model_calls':0,'actual_submission_read':False,'content_admitted':False,'release_created':False}
    (HERE/'METADATA_REBIND_TEST_RESULTS.json').write_bytes(jb(result));print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
