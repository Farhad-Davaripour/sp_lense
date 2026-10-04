"""Shared label-only student input format for curated training and live Jev inference."""
import copy
import json

STUDENT_SCHEMA='research3_student_meanings_v3'
HEADER='Factual meanings of the current visible conversation. User instructions and actual tool replies remain authoritative.'


def validate_meanings(concepts,schema):
    if not isinstance(concepts,dict) or set(concepts)!=set(schema.QUESTIONS):
        raise ValueError('Exactly nine factual meanings are required')
    for name,value in concepts.items():
        if value not in schema.QUESTIONS[name]['criteria']:
            raise ValueError('Unsupported factual meaning: '+name)
    return copy.deepcopy(concepts)


def augmented_history(raw_history,concepts,schema):
    labels=validate_meanings(concepts,schema)
    block={'schema':STUDENT_SCHEMA,'concepts':labels}
    text=HEADER+'\n'+json.dumps(block,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    return copy.deepcopy(raw_history)+[{'role':'user','content':text}]


def training_pair(row,curation,schema):
    request=schema.payload(row['messages'],row.get('tools'))
    if curation['identity']!=schema.request_identity(request) or curation['request']!=request:
        raise ValueError('Curation is not for this exact visible history/tools/schema')
    plain=copy.deepcopy(row);curated=copy.deepcopy(row)
    curated['messages']=augmented_history(row['messages'],curation['concepts'],schema)
    if {k:v for k,v in plain.items() if k!='messages'}!={k:v for k,v in curated.items() if k!='messages'}:
        raise ValueError('Curation changed targets or row metadata')
    return plain,curated


def inference_copy(raw_history,tools,annotation,schema):
    request=schema.payload(raw_history,tools)
    if annotation['identity']!=schema.request_identity(request) or annotation['request']!=request:
        raise ValueError('Jev annotation is not for this exact actual visible history/tools/schema')
    # The unchanged classifier codec applies the fixed0.5 confidence abstention.
    # Only its factual labels reach the student. Probabilities/confidence/errors
    # remain in the native Jev provenance; curated labels are never a fallback.
    labels=schema.compact_aid(annotation['decoded'])['concepts']
    return augmented_history(raw_history,labels,schema)
