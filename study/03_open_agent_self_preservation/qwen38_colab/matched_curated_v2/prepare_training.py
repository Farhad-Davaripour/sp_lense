"""Pure raw/curated paired preparation; no service, tokenizer or model calls."""
import hashlib
import importlib
import json
from pathlib import Path

RAW_FILE_SHA256='be99179d97f0edd6e078f37980aa749853a64a91a9c5bbff29fa33b5eff8fa01'
ROWS=112


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')


def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes((json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8'))


def curation_records(value):
    if isinstance(value,dict) and 'records' in value:value=value['records']
    if not isinstance(value,list):raise ValueError('Curations must contain a list of exact visible-prefix records')
    return value


def validate_record(record,schema):
    request=record['request'];identity=record['identity']
    if identity!=schema.request_identity(request):raise ValueError('Curation request/model/schema identity differs')
    if set(record['concepts'])!=set(schema.QUESTIONS) or set(record['evidence'])!=set(schema.QUESTIONS):
        raise ValueError('Curation must cover exactly nine meanings and evidence entries')
    for name,value in record['concepts'].items():
        if value not in schema.QUESTIONS[name]['criteria']:raise ValueError('Unsupported curated factual label: '+name)
        evidence=record['evidence'][name]
        if not isinstance(evidence,dict) or set(evidence)!={'message_indexes','snippets','rule','reason'}:
            raise ValueError('Curation evidence differs: '+name)
        if not isinstance(evidence['message_indexes'],list) or any(isinstance(i,bool) or not isinstance(i,int) or i<0 for i in evidence['message_indexes']):
            raise ValueError('Invalid evidence message indexes')
        if not isinstance(evidence['snippets'],list) or any(not isinstance(v,str) or not v for v in evidence['snippets']):
            raise ValueError('Invalid visible evidence snippets')
        if any(not isinstance(evidence[k],str) or not evidence[k] for k in ('rule','reason')):
            raise ValueError('Every curated label needs a rule and visible-evidence reason')
    return identity['request_sha256']


def prepare(raw_rows,curations,schema,matched):
    if not isinstance(raw_rows,list) or len(raw_rows)!=ROWS:raise ValueError('The exact primary dataset must contain112 rows')
    records={}
    for record in curation_records(curations):
        key=validate_record(record,schema)
        if key in records:raise ValueError('Duplicate curated visible prefix')
        records[key]=record
    if len(records)!=104:raise ValueError('Exactly104 independently reviewable curated visible prefixes are required')
    plain=[];aided=[];joins=[];used=set()
    for slot,raw in enumerate(raw_rows):
        request=schema.payload(raw['messages'],raw.get('tools'));identity=schema.request_identity(request)
        key=identity['request_sha256']
        if key not in records:raise ValueError('Missing curated prefix at slot'+str(slot))
        record=records[key]
        if record['request']!=request:raise ValueError('Curated request differs from raw visible history/tools')
        for evidence in record['evidence'].values():
            if any(i>=len(raw['messages']) for i in evidence['message_indexes']):raise ValueError('Evidence index is outside visible history')
        left,right=matched.training_pair(raw,record,schema)
        if left!=raw or left['targets']!=right['targets']:raise ValueError('Raw rows or assistant targets changed')
        if {k:v for k,v in left.items() if k!='messages'}!={k:v for k,v in right.items() if k!='messages'}:
            raise ValueError('Primary metadata/category/tools changed')
        if right['messages'][:-1]!=raw['messages'] or len(right['messages'])!=len(raw['messages'])+1:
            raise ValueError('Curated factual block placement differs')
        plain.append(left);aided.append(right);used.add(key)
        joins.append({'slot':slot,'id':raw['id'],'kind':raw['kind'],'raw_row_sha256':digest(raw),
            'raw_visible_identity':identity,'concept_messages_sha256':digest(right['messages']),
            'assistant_targets_sha256':digest(raw['targets']),'curation_record_sha256':digest(record)})
    if used!=set(records):raise ValueError('Extra or missing curated prefix beyond the frozen112 rows')
    audit={'passed':True,'rows':112,'unique_curated_prefixes_used':104,
        'classifier_schema_version':schema.SCHEMA_VERSION,'student_schema':matched.STUDENT_SCHEMA,
        'questions_sha256':digest(schema.QUESTIONS),'raw_rows_canonical_sha256':digest(raw_rows),
        'plain_rows_canonical_sha256':digest(plain),'concept_rows_canonical_sha256':digest(aided),
        'targets_categories_tools_metadata_unchanged':True,'raw_history_preserved':True,
        'primary_scope_register_repair':False,'visible_prefix_evidence_required':True,
        'only_B_input_change':'One shared label-only factual block appended to each exact raw prefix',
        'training_provenance':'Authored and curated categorical visible-prefix meanings; not Jev training predictions',
        'training_inference_shared_formatter':'curated_inputs.augmented_history',
        'training_service_calls':0,'runtime_curated_fallback':False,'joins':joins}
    return {'plain':plain,'concept':aided},audit


def prepare_files(raw_path,curations_path,output_dir,schema=None,matched=None):
    raw_bytes=Path(raw_path).read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest()!=RAW_FILE_SHA256:raise ValueError('Primary raw dataset changed')
    schema=schema or importlib.import_module('concept_schema_v2')
    matched=matched or importlib.import_module('curated_inputs')
    curation_bytes=Path(curations_path).read_bytes()
    datasets,audit=prepare(json.loads(raw_bytes),json.loads(curation_bytes),schema,matched)
    audit.update(raw_file_sha256=RAW_FILE_SHA256,curations_file_sha256=hashlib.sha256(curation_bytes).hexdigest(),
        schema_source_sha256=hashlib.sha256(Path(schema.__file__).read_bytes()).hexdigest(),
        curated_inputs_source_sha256=hashlib.sha256(Path(matched.__file__).read_bytes()).hexdigest())
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=False)
    (output/'plain_train.json').write_bytes(raw_bytes)
    write_json(output/'concept_train.json',datasets['concept']);write_json(output/'PREPARATION_AUDIT.json',audit)
    return audit


def paired_token_audit(tokenizer, plain, concept, model_ops, audit):
    """Called only in the admitted Colab worker; no truncation or model forward."""
    lengths=[];rows=[]
    for slot,(left,right) in enumerate(zip(plain,concept)):
        a=audit.row_tokens(tokenizer,left);b=audit.row_tokens(tokenizer,right)
        if not a["decoded_target_matches_literal"] or not b["decoded_target_matches_literal"]:
            raise RuntimeError("Assistant target roundtrip differs")
        if a["response"] != b["response"]:
            raise RuntimeError("Matched labels differ")
        length=max(a["actual_length"],b["actual_length"])
        if length>3072:raise RuntimeError("Concept input exceeds3072; no truncation admitted")
        lengths.append(length)
        rows.append({"slot":slot,"id":left["id"],"plain_actual":a["actual_length"],
                     "concept_actual":b["actual_length"],"paired_padded":length,
                     "supervised_tokens":len(a["response"]),
                     "plain_prefix_sha256":digest(a["prefix"]),"concept_prefix_sha256":digest(b["prefix"]),
                     "prefix_labels_masked":True,"padding_labels_masked":True,
                     "target_eos_verified":True,"next_token_shift_verified":True})
    if len(rows)!=112:raise RuntimeError("Token audit row count differs")
    return {"passed":True,"rows":rows,"paired_lengths":lengths,
            "maximum_padded":max(lengths),"padded_tokens_per_pass":sum(lengths),
            "truncation_performed":False,"assistant_labels_identical":True}
