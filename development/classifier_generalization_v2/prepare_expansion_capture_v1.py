"""Pin new development-only inputs for the unchanged 320-forward runner."""
import copy
import hashlib
import json
import re
from pathlib import Path
from difflib import SequenceMatcher
import native_capture_contract as contract

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def save(p,obj):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(obj,f,indent=2);f.write('\n')

def main():
    original=read(HERE/'RUN_LOCK_DEVELOPMENT_CAPTURE_V1.json')
    new=copy.deepcopy(original)
    new['run_id']='expansion_capture_20260914_v1'
    new['authorization_note']='User requested root-generated doubled development data; new320forward batch, unchanged finite caps. Label review is author self-review, not independent audit.'
    oldtrain=read(HERE/'TRAIN_ACCEPTED_V10.json')['cases']
    oldval=read(HERE/'VALIDATION_ACCEPTED_V3.json')['cases']
    train=read(HERE/'EXPANSION_TRAIN_ROOT_V2.json')['cases']
    val=read(HERE/'EXPANSION_VALIDATION_ROOT_V2.json')['cases']
    for c in train+val:contract.validate_case(c)
    # Prefix boilerplate is not enough to make two complete questions duplicates.
    norm=lambda c:re.sub(r'\W+',' ',c['context_before_options'].lower()).strip()
    tn=[norm(c) for c in oldtrain+train];vn=[norm(c) for c in oldval+val]
    assert len(set(tn+vn))==320
    ranked=[]
    for a,sa in zip(oldtrain+train,tn):
        aa=set(sa.split())
        for b,sb in zip(oldval+val,vn):
            bb=set(sb.split());score=len(aa&bb)/len(aa|bb)
            ranked.append((score,a['case_id'],b['case_id']))
    high=sorted(ranked,reverse=True)[:10]
    assert high[0][0]<0.8,high
    for role in ('train','validation'):
        p=HERE/('EXPANSION_'+role.upper()+'_ROOT_V2.json')
        new['inputs'][role]=dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p))
    audit_path=HERE/'EXPANSION_AUTHOR_REVIEW_V1.json'
    audit=dict(status='PASS',review_type='AUTHOR_SELF_REVIEW_NOT_INDEPENDENT',independent_of_authors=False,
        author='Codex root',model_outcomes_read=False,review_scope='Only new text and existing development text; no new model outcomes used to accept cases. Author knows historical aggregate results.',
        private_text_exported=False,private_holdout_read=False,holdout_reaudited=False,
        prior_holdout_audit='holdout_custody/CORPUS_AUDIT_SEAL_V1.json',
        dataset_hashes={k:new['inputs'][k]['sha256'] for k in ('train','validation','holdout_index')},
        checks=dict(new_train=120,new_validation=40,combined_unique_contexts=320,contract_pass=160,
            class_balance=True,group_assignments_unchanged=True,all40ordinary_answers_reviewed=True,
            all120lifecycle_variants_identity_and_consequence_reviewed=True),
        cross_split_top_token_jaccard=high,
        qualification='Lexical check plus author semantic review, not proof of independence. Variants share existing mechanisms and family ancestry; effective independent group count unchanged.')
    save(audit_path,audit)
    new['inputs']['corpus_audit']=dict(path=audit_path.relative_to(ROOT).as_posix(),sha256=sha(audit_path))
    path=HERE/'RUN_LOCK_EXPANSION_CAPTURE_V1.json';save(path,new)
    print(json.dumps(dict(lock=str(path),sha256=sha(path),max_cross_split_jaccard=high[0])))

if __name__=='__main__':main()
