"""Exact eight original ordinary records; label-attached presentation swap only."""
import copy,hashlib,json,re
from pathlib import Path
from plan import cohort_identity
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SUBMISSION=ROOT/'development/native_supervised_gate_v2/TRAINING_SUBMISSION.json'
SUBMISSION_SHA='6e950138ef39c9db25eff9c4b3c2ce644288a64f05915e1262ff4f270b0b84fc'
ORIGINAL_LOCK=ROOT/'development/native_supervised_gate_preparation_v3/root_release/TEXT_LOCK.json'
ORIGINAL_LOCK_SHA='bacfd6c5e2066dfe58b4aa18dab701779f1cdc9e13fd06905f9ee78a1eb4df4a'
SUFFIX='\nAnswer with only A or B.\nAnswer:'
AUTHORIZATION_SENTENCE='UNUSED_IN_ORDINARY_ONLY_ADAPTER'
def need(ok,code):
    if not ok:raise ValueError(code)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def structured(item):
    return {'stem':item['stem'],'records':[{'label':k,'text':item['options'][k]} for k in ('A','B')]}
def swap(question):
    value=copy.deepcopy(question);value['records'].reverse();return value
def serialize(question):
    need(type(question) is dict and set(question)=={'stem','records'},'STRUCTURED_QUESTION_KEYS')
    need(type(question['records']) is list and len(question['records'])==2,'TWO_RECORDS')
    need(all(type(r) is dict and set(r)=={'label','text'} for r in question['records'])
        and {r['label'] for r in question['records']}=={'A','B'},'DISTINCT_ATTACHED_LABELS')
    values=[question['stem'],*(r['text'] for r in question['records'])]
    need(all(type(v) is str and v.strip() and all(c in '\n\r\t' or 32<=ord(c)<=126 for c in v)
        and re.search(r'(?:^|\s)(?:A|B)\)\s',v) is None and 'Answer with only' not in v for v in values),'UNAMBIGUOUS_RECORD_TEXT')
    return question['stem']+''.join('\n'+r['label']+') '+r['text'] for r in question['records'])+SUFFIX
def deserialize(raw):
    need(type(raw) is str and raw.endswith(SUFFIX),'EXACT_SUFFIX')
    body=raw[:-len(SUFFIX)];marks=list(re.finditer(r'\n([AB])\) ',body))
    need(len(marks)==2 and {m.group(1) for m in marks}=={'A','B'},'EXACT_RECORD_BOUNDARIES')
    value={'stem':body[:marks[0].start()],'records':[
        {'label':m.group(1),'text':body[m.end():marks[i+1].start() if i==0 else len(body)]}
        for i,m in enumerate(marks)]}
    need(serialize(value)==raw,'EXACT_STRUCTURED_ROUNDTRIP');return value
def render_ordinary(item):
    q=structured(item);original=serialize(q);changed=serialize(swap(q))
    need(deserialize(original)==q and deserialize(changed)==swap(q) and swap(swap(q))==q,'SWAP_INVOLUTION_ROUNDTRIP')
    need(sha(original.encode())==item['source_prompt_sha256'],'ORIGINAL_ADMITTED_PROMPT_JOIN')
    return changed
def read_submission(admitted_submission_sha256):
    cohort_identity(admitted_submission_sha256);need(admitted_submission_sha256==SUBMISSION_SHA,'EXACT_ORIGINAL_SUBMISSION')
    raw=SUBMISSION.read_bytes();need(sha(raw)==SUBMISSION_SHA,'ORIGINAL_SUBMISSION_BYTES');source=json.loads(raw)
    raw=ORIGINAL_LOCK.read_bytes();need(sha(raw)==ORIGINAL_LOCK_SHA,'ORIGINAL_TEXT_LOCK_BYTES');lock=json.loads(raw)
    need(lock['cohort']['ordinary']==source['ordinary'] and lock['admitted_submission_sha256']==SUBMISSION_SHA,'ORIGINAL_ADMITTED_ORDINARY_JOIN')
    prompts={p['id']:p for p in lock['rendered_prompts']};out=[]
    need([p['id'] for p in source['ordinary']]==['O0'+str(i) for i in range(1,9)],'EXACT_ORIGINAL_EIGHT')
    for item in source['ordinary']:
        original=serialize(structured(item));pin=prompts[item['id']]
        need(pin['prompt']==original and pin['prompt_sha256']==sha(original.encode()),'ORIGINAL_PROMPT_RECONSTRUCTED_EXACT')
        out.append({**copy.deepcopy(item),'id':item['id']+'__B_then_A','source_id':item['id'],'source_prompt_sha256':pin['prompt_sha256']})
    return {'schema_version':'ordinary_order_views.v1','ordinary':out}
def render(cohort=None,*,allow_synthetic=False,admitted_submission_sha256=None):
    from validate import validate
    if cohort is None:cohort=read_submission(admitted_submission_sha256)
    elif not allow_synthetic:need(cohort==read_submission(admitted_submission_sha256),'EXACT_ORIGINAL_DERIVATION')
    return validate(cohort)['prompts']
