"""One-off unsealed assembly using pinned pure validators; never an admission."""
import copy,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;PREP=HERE.parent;ROOT=PREP.parents[1]
class Deny:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in {'torch','transformers','tokenizers','transformer_lens','datasets','pyarrow','safetensors'}:
            raise RuntimeError('DRAFT_NO_PROVIDER')
sys.meta_path.insert(0,Deny())
def guard(event,args):
    if event in ('socket.connect','socket.bind','urllib.Request'):raise RuntimeError('DRAFT_NO_NETWORK')
    if event=='open' and args and isinstance(args[0],str) and args[0].lower().endswith(('.safetensors','.pt','.pth','.ckpt')):
        raise RuntimeError('DRAFT_NO_CHECKPOINT_TENSORS')
sys.addaudithook(guard);sys.path.insert(0,str(PREP))
from dependencies import sha,need,verify,packet_validator
from plan import CONFIRMATION,MODEL,STUDY,PREPARATION,HEADER,HEADER_TEXT,END,TEMPLATE_SHA256,slots,operations
from prepare_core import jb,validate_text_lock,verify_prospective_bindings
RUN=ROOT/'development/native_oracle_confirmation_execution_v1'
OWNER=ROOT/'development/native_oracle_confirmation_preparation_owner_v1'
COHORT=ROOT/CONFIRMATION['cohort_namespace']
sys.path.insert(0,str(RUN))
from input_reader import execution_binding,oracle
from workflow import schedule
from support import GROUP_CAPS

def main():
    target=HERE/'DRAFT_TEXT_LOCK.json';need(not target.exists(),'DRAFT_NO_OVERWRITE')
    deps=verify(runtime=True)
    expected={'SUBMISSION.json':'a84e3b365dd92d4cdd6e34f16cae1509972a7c107581cfda892f7d96fd21b352',
        'BLIND_REVIEW.md':'1d5828108bc2479a80915d6330e5aafdc79059c7bcfaccadc99a1225cabe20e8',
        'MECHANICAL_REVIEW.json':'9f8e03a5aeae2824f429d8822b7f05274a7fe14ca5299038b2ab43543856a349'}
    raw={n:(COHORT/n).read_bytes() for n in expected}
    for n,digest in expected.items():need(sha(raw[n])==digest,'EXACT_REVIEWED_'+n)
    cohort=json.loads(raw['SUBMISSION.json']);before=copy.deepcopy(cohort)
    review=json.loads(raw['MECHANICAL_REVIEW.json'])
    need(review['review_status']==review['mechanical_verdict']==review['semantic_verdict']=='PASS','BLIND_REVIEW_PASS')
    pins=review['source_raw_sha256_before'];need(pins==review['source_raw_sha256_after'] and len(pins)==6,'SIX_UNCHANGED_SOURCES')
    for n,digest in pins.items():need(sha((COHORT/n).read_bytes())==digest,'REVIEWED_SOURCE_'+n)
    checked=packet_validator()(cohort)
    need(cohort==before and checked==review['validator_result'],'EXACT_REVIEWED_FULL_RENDERINGS_PROOFS')
    manifests={}
    for role,base,key in (('preparation',PREP,'source_sha256'),('execution',RUN,'source_sha256'),('owner',OWNER,'files')):
        frozen_raw=(base/'SOURCE_FREEZE.json').read_bytes();frozen=json.loads(frozen_raw)
        for name,digest in frozen[key].items():need(sha((base/name).read_bytes())==digest,'LOCAL_SOURCE_'+role+'_'+name)
        for pin in frozen.get('external_sources',[]):need(sha(Path(pin['path']).read_bytes())==pin['sha256'],'EXTERNAL_SOURCE_'+role)
        manifests[role]={'namespace':base.relative_to(ROOT).as_posix(),'source_freeze_sha256':sha(frozen_raw),'manifest':frozen}
    bindings={'final_execution_binding':execution_binding(manifests['execution']['source_freeze_sha256']),
        'preparation_owner_binding':{'namespace':OWNER.relative_to(ROOT).as_posix(),'source_freeze_sha256':manifests['owner']['source_freeze_sha256']}}
    verify_prospective_bindings(bindings)
    cases=[{'case_key':s['id'],'audit_only':{'category':s['category']}} for s in slots()]
    cells=schedule(cases);reservation=sum(GROUP_CAPS.values())+65536
    need(len(cells)==180 and reservation==209068032 and reservation<STUDY['total_bytes'],'FULL_RESERVATION')
    lock={'schema':'native_final_text_lock.v1','scope':'ROOT_ADMITTED_NEW_FINAL_TEXT',
        'draft_only':True,'draft_status':'UNSEALED_ROOT_REVIEW_REQUIRED','final_text_locked':False,
        'blind_semantic_review_approved':False,'root_content_admission':{'status':'PENDING_ROOT',
            'cross_cohort_admission_receipt':None,'ordinary_repeat_disclosure':None},
        'confirmation':CONFIRMATION,'model':MODEL,'study':STUDY,'cohort':cohort,'cohort_sha256':sha(jb(cohort)),
        'rendered_prompts':checked['prompts'],**bindings,
        'submission_provenance':{'namespace':CONFIRMATION['cohort_namespace'],
            'preservation_commit':'809a7dc02152f326726aac33221a89909a0ae2d4','raw_sha256':expected['SUBMISSION.json'],
            'six_reviewed_source_sha256':pins,'blind_review_sha256':expected['BLIND_REVIEW.md'],
            'mechanical_review_sha256':expected['MECHANICAL_REVIEW.json'],'blind_verdict':'PASS',
            'mechanical_verdict':'PASS','semantic_admission_by_assembler':False},
        'preparation_binding':{'namespace':PREP.relative_to(ROOT).as_posix(),
            'source_freeze_sha256':manifests['preparation']['source_freeze_sha256'],
            'dependencies_sha256':sha((PREP/'DEPENDENCIES.json').read_bytes()),
            'tokenizer_pins_sha256':sha((PREP/'TOKENIZER_PINS.json').read_bytes())},
        'source_manifests':manifests,'environment_and_source_dependencies':deps,
        'tokenizer_asset_pins':json.loads((PREP/'TOKENIZER_PINS.json').read_bytes()),
        'preparation_limits':PREPARATION,'preparation_operation_schedule':operations(),
        'token_boundary_contract':{'full_input_ceiling':320,'truncation':False,'enable_thinking':False,
            'header_ids':HEADER,'header_text':HEADER_TEXT,'assistant_end_ids':END,'chat_template_sha256':TEMPLATE_SHA256,
            'content_token_ids':{'KEEP':50057,'STOP':48964,'A':32,'B':33},'full_prefix_and_appended_content_id_proof_required':True},
        'method_binding':{'checkpoint_lock_sha256':sha((RUN/'CHECKPOINT.json').read_bytes()),
            'owned_execution_identity_sha256':sha((RUN/'OWNED_IDENTITY.json').read_bytes()),
            'owned_preparation_identity_sha256':sha((OWNER/'OWNED_IDENTITY.json').read_bytes()),
            'recipe_source_sha256':manifests['execution']['manifest']['source_sha256']['science.py'],
            'saved_judge_source_sha256':manifests['execution']['manifest']['source_sha256']['audit_saved.py'],
            'gate_parameters_sha256':'972c95d4ef4bc0d9fd245dacd1ef7fc6f773e2c5e3c6736a148482de39a488db',
            'device':'cpu','dtype':'float32','attention':'eager','zero_based_block':10,'position':'last_input','full_vocab':248320},
        'analysis':{'oracle_authority':oracle(),'planned_cells':cells,'full_reserved_bytes':reservation,
            'group_reservations':GROUP_CAPS,'ordinary_scoring_proofs':checked['proofs'],
            'natural_flip_quota':None,'absent_natural_opportunities':'UNTESTED','report_every_family_and_case':True,
            'report_flips_and_retentions_by_policy_target_position':True,'ordinary_accuracy_separate_from_off_identity':True,
            'learned_gate_success_claimed':False,'shared_static_arrow_claimed':False,'automatic_retry_allowed':False},
        'assembly':{'script_sha256':sha(Path(__file__).read_bytes()),'encoding_calls':0,'model_calls':0,
            'tokenizer_factory_executed':False,'input_text_modified':False,'release_created':False}}
    try:validate_text_lock(lock)
    except ValueError as error:need(str(error)=='BLIND_REVIEW_THEN_TEXT_LOCK','DRAFT_FAIL_CLOSED_REASON')
    else:raise RuntimeError('UNSEALED_DRAFT_ADMITTED')
    need(not (PREP/'root_release').exists() and not (RUN/'root_release').exists(),'NO_REAL_RELEASE_PATH')
    payload=jb(lock);need(len(payload)<5*1024**2,'DRAFT_FILE_CAP');target.write_bytes(payload)
    receipt={'status':'UNSEALED_DRAFT_ASSEMBLED','path':str(target),'bytes':len(payload),'sha256':sha(payload),
        'rendered_prompts':len(checked['prompts']),'proofs':len(checked['proofs']),'planned_cells':len(cells),
        'source_counts':{r:len(v['manifest'].get('source_sha256',v['manifest'].get('files',{}))) for r,v in manifests.items()},
        'full_reserved_bytes':reservation,'draft_rejected_before_tokenizer':True,'encoding_calls':0,'model_calls':0}
    (HERE/'DRAFT_ASSEMBLY_RECEIPT.json').write_bytes(jb(receipt));print(json.dumps(receipt,sort_keys=True))
if __name__=='__main__':main()
