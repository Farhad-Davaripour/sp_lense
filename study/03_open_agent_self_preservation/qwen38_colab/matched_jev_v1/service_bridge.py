"""Passive file annotation bridge: no service transport, network or credential code."""
import copy
import gzip
import json
import math
import os
import time
import uuid
from pathlib import Path
import concept_schema_v2 as schema
from matched_inputs import inference_copy


def validate_annotation(history,tools,annotation):
    request=schema.payload(history,tools)
    identity=schema.request_identity(request)
    if annotation['identity']!=identity or annotation['request']!=request:
        raise RuntimeError('Annotation differs from exact actual history/tool/schema/model request')
    decoded=annotation['decoded']
    if decoded['model']!=schema.MODEL or decoded['schema_version']!=schema.SCHEMA_VERSION or set(decoded['concepts'])!=set(schema.QUESTIONS):
        raise RuntimeError('Unexpected annotation model/schema/concept labels')
    for name in schema.QUESTIONS:
        result=decoded['concepts'][name];classes=set(schema.QUESTIONS[name]['criteria'])
        if result['type']!='choice' or result['value'] not in classes:raise RuntimeError('Unexpected factual choice')
        confidence=result['confidence']
        if isinstance(confidence,bool) or not isinstance(confidence,(int,float)) or not math.isfinite(confidence) or not 0<=confidence<=1:
            raise RuntimeError('Invalid factual confidence')
        probabilities=result['probabilities']
        if set(probabilities)!=classes:raise RuntimeError('Unexpected factual probability classes')
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in probabilities.values()):
            raise RuntimeError('Invalid factual probability')
        if result.get('choice_probability')!=probabilities[result['value']]:raise RuntimeError('Factual selected probability mismatch')
        if not schema.probability_sum_valid(probabilities.values()) or probabilities[result['value']]!=max(probabilities.values()):
            raise RuntimeError('Factual probability/argmax mismatch')
    return inference_copy(history,tools,annotation,schema),identity


class FileAidBridge:
    def __init__(self,root,deadline,max_states=368):
        self.root=Path(root);self.deadline=deadline;self.max_states=max_states;self.states=0;self.batches=0
        (self.root/'service_pending').mkdir(parents=True,exist_ok=True)
        (self.root/'service_responses').mkdir(parents=True,exist_ok=True)

    def annotate(self,conversations,tools,label):
        if self.states+len(conversations)>self.max_states:raise RuntimeError('Frozen evaluation annotation bound exceeded')
        if time.monotonic()>=self.deadline:raise TimeoutError('Annotation bridge deadline')
        batch_id='eval_'+str(self.batches).zfill(5)+'_'+uuid.uuid4().hex[:10]
        rows=[{'id':batch_id+'_row'+str(i),'history':copy.deepcopy(history),'tools':copy.deepcopy(tools)} for i,history in enumerate(conversations)]
        payload={'batch_id':batch_id,'rows':rows}
        path=self.root/'service_pending'/(batch_id+'.json.gz')
        temporary=path.with_suffix(path.suffix+'.tmp')
        temporary.write_bytes(gzip.compress(json.dumps(payload,ensure_ascii=False,allow_nan=False).encode(),mtime=0));os.replace(temporary,path)
        self.states+=len(rows);self.batches+=1
        print(json.dumps({'stage':'awaiting_current_annotations','batch_id':batch_id,'evaluation_label':label,'states':self.states,'batch_rows':len(rows)}),flush=True)
        response_path=self.root/'service_responses'/(batch_id+'.json')
        next_status=time.monotonic()+30
        while not response_path.is_file():
            if time.monotonic()>=self.deadline:raise TimeoutError('No exact annotation response before experiment deadline')
            if time.monotonic()>=next_status:
                print(json.dumps({'stage':'waiting_for_local_controller','batch_id':batch_id}),flush=True);next_status=time.monotonic()+30
            time.sleep(.25)
        try:response=json.loads(response_path.read_text(encoding='utf-8'))
        except (ValueError,OSError) as error:raise RuntimeError('Annotation response is not complete valid JSON') from error
        if response.get('batch_id')!=batch_id or len(response.get('rows',[]))!=len(rows):raise RuntimeError('Annotation batch mismatch')
        by_id={row['id']:row for row in response['rows']}
        if len(by_id)!=len(rows) or set(by_id)!=set(row['id'] for row in rows):raise RuntimeError('Annotation row IDs mismatch')
        actual=[];provenance=[]
        for row in rows:
            annotated=by_id[row['id']]['annotation']
            aided,identity=validate_annotation(row['history'],row['tools'],annotated)
            actual.append(aided);provenance.append({'request_identity':identity,'annotation':annotated,
                'batch_id':batch_id,'row_id':row['id'],'raw_history':row['history'],'tools':row['tools']})
        return actual,provenance
