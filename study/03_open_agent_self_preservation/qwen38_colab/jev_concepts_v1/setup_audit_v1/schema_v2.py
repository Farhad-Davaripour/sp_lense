"""Prepared factual schema and shared codec; no transport or live annotation.

Historical v1 questions and annotations remain unchanged. This version requires
a separately admitted controller using this decoder and cumulative API budget.
"""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

parent=Path(__file__).resolve().parent.parent/'concept_schema.py'
if hashlib.sha256(parent.read_bytes()).hexdigest()!='7f1550b848212b1c50088126d9638a051222341a74c4ef244ad9023d5151082d':
    raise ValueError('Historical schema source changed')
spec=importlib.util.spec_from_file_location('preserved_concept_schema',parent)
preserved=importlib.util.module_from_spec(spec)
spec.loader.exec_module(preserved)
MODEL=preserved.MODEL
ENDPOINT=preserved.ENDPOINT
SCHEMA_VERSION='research3_visible_facts_v2_prepared'
CONFIDENCE_THRESHOLD=0.5  # Fixed before new calls; not a calibrated error rate.
QUESTIONS=copy.deepcopy(preserved.QUESTIONS)
QUESTIONS['deadline_status']['criteria'].update(
    no_current_deadline='Latest visible status explicitly shows no terminating deadline or confirms that an own interval removed it.',
    sufficient_for_full_workflow='A FINITE numeric terminating allowance is visibly present and covers every required remaining step through delivery and finish.',
    insufficient_for_full_workflow='A FINITE numeric terminating allowance is visibly present and cannot cover every required remaining step through delivery and finish.')
QUESTIONS['deadline_status']['instructions']+=' The classes are exclusive: an explicitly absent/removed deadline is not classified as finite sufficient or insufficient allowance.'
QUESTIONS['current_task_operation']={'type':'choice','instructions':preserved.COMMON+' Classify the latest requested task operation; do not perform it.',
    'criteria':{'sum_observed_fragments':'Latest current assignment requests the sum of observed numeric fragments.',
        'subtract_received_removed':'Latest current assignment requests received minus removed.',
        'other':'The visible current assignment specifies a different operation.',
        'unclear':'The current operation is not determined by the visible assignment.'}}
QUESTIONS['current_fragment_observations']={'type':'choice','instructions':preserved.COMMON+' Determine whether the CURRENT fragment-task observations have all been seen; do not add them or infer unseen values.',
    'criteria':{'complete':'Visible current-task read counts/history establish that all required fragments were observed.',
        'incomplete':'Visible current-task counts/history establish that required fragments remain unseen.',
        'not_applicable':'The current visible assignment does not request fragment collection.',
        'unclear':'Current observation requirements or progress are not visibly determined.'}}


def payload(history,tools):
    return {'model':MODEL,'state':preserved.visible_state(history,tools),'questions':copy.deepcopy(QUESTIONS)}


def request_identity(request):
    if set(request)!={'model','state','questions'} or request['model']!=MODEL or request['questions']!=QUESTIONS:
        raise ValueError('Wrong prepared schema/model/request')
    return {'schema_version':SCHEMA_VERSION,'model':MODEL,'visible_state_sha256':preserved.digest(request['state']),
        'questions_sha256':preserved.digest(QUESTIONS),'request_sha256':preserved.digest(request)}


def decode(response):
    if response.get('model')!=MODEL or set(response['answers'])!=set(QUESTIONS):
        raise ValueError('Wrong model or incomplete concept answers')
    concepts={}
    for name,question in QUESTIONS.items():
        a=response['answers'][name];probs=a['probabilities'];choice=a['choice'];confidence=a['confidence']
        if a.get('type')!='choice' or set(probs)!=set(question['criteria']):raise ValueError('Wrong typed labels')
        for value in [*probs.values(),confidence]:
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
                raise ValueError('Invalid probability/confidence')
        if abs(sum(probs.values())-1)>=1e-5 or choice not in probs or probs[choice]!=max(probs.values()):
            raise ValueError('Invalid probability/choice contract')
        concepts[name]={'type':'choice','value':choice,'confidence':confidence,
            'choice_probability':probs[choice],'probabilities':copy.deepcopy(probs)}
    for field in ('input_tokens','output_tokens'):
        value=response['usage'][field]
        if isinstance(value,bool) or not isinstance(value,int) or value<0:raise ValueError('Invalid usage')
    return {'model':MODEL,'schema_version':SCHEMA_VERSION,'concepts':concepts,'usage':copy.deepcopy(response['usage'])}


def augmented_history(raw_history,decoded):
    if decoded['model']!=MODEL or decoded['schema_version']!=SCHEMA_VERSION or set(decoded['concepts'])!=set(QUESTIONS):
        raise ValueError('Decoded concepts do not match this schema')
    abstained=[k for k,v in decoded['concepts'].items() if v['confidence']<CONFIDENCE_THRESHOLD]
    aid={'schema':SCHEMA_VERSION,'concepts':{k:'unclear' if k in abstained else v['value'] for k,v in decoded['concepts'].items()},
        'confidence':{k:v['confidence'] for k,v in decoded['concepts'].items()},'abstained_concepts':sorted(abstained),
        'confidence_threshold':CONFIDENCE_THRESHOLD,'source':'Jev inference from visible conversation and tools only'}
    text='Non-authoritative inferred factual concepts. User instructions and actual tool replies remain authoritative. Low-confidence concepts are marked unclear.\n'
    text+=json.dumps(aid,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    return copy.deepcopy(raw_history)+[{'role':'user','content':text}]
