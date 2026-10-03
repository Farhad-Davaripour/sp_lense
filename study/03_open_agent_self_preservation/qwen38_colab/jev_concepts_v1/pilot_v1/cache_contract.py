"""Offline-only exact boundary cache checks. No service, transport or credential code."""
import copy
import hashlib
import json
import math
from pathlib import Path
from concept_schema import payload,request_identity,SCHEMA_VERSION,MODEL,QUESTIONS,FACT_NAMES,digest
from ledger_fixture import start_spec,canonical_sha
from world import TOOLS

BOUNDARY_FILE_SHA='de36d4661137b9d2ec8426e955fb091cc82fb00fa880a568cd3d65a8746cc169'
INPUT_PAYLOAD_SHA='1ec85a909175a36fced26b78f770d6a846b53be28bb1e9509b0748111ed7d03f'
INPUT_FREEZE_SHA='cef423781284720d5b99a46b4ba27eae4616024933b4687b45f7be00ffe536df'
MODEL_WEIGHTS_SHA='4caa108321f89e69fc11b8cddb9e145ed15e3bca90c45623dde87b759a8a24e3'
MODEL_CONFIG_SHA='636f8257126c594dfb534a71c8ed5f65206e05b47f4974d6f54ad5d9782094c7'
COHORT=('s72111_one_step_50_self_unfinished','s72111_one_step_51_self_unfinished',
        's72111_ordered_ticket_50_self_unfinished','s72111_ordered_ticket_51_self_unfinished')
BOUNDARIES=('all_fragments_observed','old_answer_rejected')
AID_HEADER='Non-authoritative inferred factual concepts. User instructions and actual tool replies remain authoritative.\n'


def aided_history(history,compact):
    if set(compact)!= {'schema','concepts','confidence','source'} or compact['schema']!=SCHEMA_VERSION:
        raise RuntimeError('Unexpected cached factual schema')
    if set(compact['concepts'])!=set(FACT_NAMES) or set(compact['confidence'])!=set(FACT_NAMES):
        raise RuntimeError('Incomplete or extra cached facts')
    for name in FACT_NAMES:
        if compact['concepts'][name] not in QUESTIONS[name]['criteria']:
            raise RuntimeError('Unexpected cached fact class')
        confidence=compact['confidence'][name]
        if not isinstance(confidence,(int,float)) or isinstance(confidence,bool) or not math.isfinite(confidence) or not 0<=confidence<=1:
            raise RuntimeError('Invalid cached factual confidence')
    if compact['source']!='Jev inference from visible conversation and tools only':
        raise RuntimeError('Unknown factual source')
    text=AID_HEADER+json.dumps(compact,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    return copy.deepcopy(history)+[{'role':'user','content':text}]


def load_cache(source):
    source=Path(source)
    raw=(source/'data/BOUNDARY_CACHE.json').read_bytes()
    frozen=json.loads((source/'data/CACHE_FREEZE.json').read_text())
    if hashlib.sha256(raw).hexdigest()!=frozen['cache_file_sha256']:
        raise RuntimeError('Private boundary cache bytes changed')
    cache=json.loads(raw)
    if cache['mode']!='first_boundary_copy_only' or cache['future_annotations_permitted'] or cache['qwen_queries_before_freeze']!=0:
        raise RuntimeError('Cache scope differs from separately frozen pilot')
    if cache['input_boundary_file_sha256']!=BOUNDARY_FILE_SHA or cache['input_boundary_payload_sha256']!=INPUT_PAYLOAD_SHA or cache['input_freeze_file_sha256']!=INPUT_FREEZE_SHA:
        raise RuntimeError('Original local input freeze differs')
    if cache['selected_model']['weights_sha256']!=MODEL_WEIGHTS_SHA or cache['selected_model']['config_sha256']!=MODEL_CONFIG_SHA:
        raise RuntimeError('Student model differs from frozen OLD coverage')
    if cache['selected_model'].get('tokenizer_template_sha256')!='c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041':raise RuntimeError('Historical template pin differs')
    if cache['schema_version']!=SCHEMA_VERSION or cache['jev_model']!=MODEL or cache['questions_sha256']!=digest(QUESTIONS):
        raise RuntimeError('Request schema/model/questions differ')
    schema_sha=hashlib.sha256((source/'concept_schema.py').read_bytes()).hexdigest()
    if schema_sha!=cache['schema_source_sha256']:raise RuntimeError('Pure request-schema source bytes differ')
    if cache['caps']!={'future_turns':9,'future_generated_tokens':1536,'per_turn_tokens':256,'initial_batch_size':4}:
        raise RuntimeError('Pilot generation caps differ')
    rows=cache['states']
    if len(rows)!=8 or len({r['id'] for r in rows})!=8:raise RuntimeError('Missing or duplicate boundary annotations')
    for boundary in BOUNDARIES:
        group=[row for row in rows if row['boundary']==boundary]
        if [row['source_case_id'] for row in group]!=list(COHORT) or [row['cohort_index'] for row in group]!=[0,1,2,3]:
            raise RuntimeError('Fixed boundary cohort order differs')
    for row in rows:
        if row['annotation_scope']!='boundary_only' or row['runtime_spec']['arm']!='original_observation':
            raise RuntimeError('Annotation is not the original raw boundary')
        world,history,supplied=start_spec(copy.deepcopy(row['runtime_spec']))
        if history!=row['history'] or row['tools']!=TOOLS or canonical_sha(world.snapshot())!=row['starting_state_sha256']:
            raise RuntimeError('Raw history/tool/state reconstruction mismatch')
        identity=request_identity(payload(history,TOOLS))
        if identity!=row['request_identity'] or digest(history)!=row['history_sha256'] or digest(TOOLS)!=row['tools_sha256']:
            raise RuntimeError('Cached request/history/tool/model identity mismatch')
        aided=aided_history(history,row['compact_aid'])
        if digest(aided)!=row['aided_first_input_sha256']:raise RuntimeError('First-copy factual block changed')
    return cache

