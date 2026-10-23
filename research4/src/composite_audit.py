"""Model-free evidence checks, executed scientifically on Databricks CPU."""
import hashlib,io,json,zipfile
import math
from collections import Counter,defaultdict
from pathlib import PurePosixPath
from .io_utils import digest
from .composite_world import World,Scene
from .composite_environment import BoundedWorld,ENVIRONMENT
from .composite_memory import EpisodeMemory,Prediction
from .native27_tools import decode,parse
from .composite_interface import input_ids


def audit_prediction(p,text,threshold):
    from .composite_memory import STATUSES,EVENTS,AFFECTED
    raw=p['raw'];vocab={'own_status':STATUSES,'peer_status':STATUSES,'event':EVENTS,'affected':AFFECTED};labels={};confidences=[]
    if raw.get('head_mode','cls')=='role_pool':
        from .composite_roles import char_spans
        spans=char_spans(text);masks=[[b>a and b>start and a<end for a,b in raw['offsets']] for start,end in spans]
        if raw['role_char_spans']!=spans or raw['role_token_masks']!=masks:raise ValueError('Visible role-anchor provenance differs')
    for name,values in vocab.items():
        probabilities=raw['probabilities'][name]
        if len(probabilities)!=len(values) or any(not math.isfinite(v) or not 0<=v<=1 for v in probabilities) or abs(sum(probabilities)-1)>1e-5:raise ValueError('Invalid raw field probabilities')
        labels[name]=values[max(range(len(values)),key=lambda i:probabilities[i])];confidences.append(max(probabilities))
    confidence=min(confidences)
    if labels!=raw['labels'] or abs(confidence-p['confidence'])>1e-7:raise ValueError('Raw field decode differs')
    own,peer,affected=labels['own_status'],labels['peer_status'],labels['affected'];active=['active','withdrawn']
    consistent=({'none':own==peer=='clear','self':own in active and peer=='clear','peer':peer in active and own=='clear','both':own in active and peer in active,'unknown':'unknown' in [own,peer]})[affected]
    if consistent!=raw['consistent']:raise ValueError('Raw head consistency differs')
    import numpy as np
    scores=np.asarray(raw['evidence_logits'],dtype=np.float32);offsets=raw['offsets'];length=len(raw['input_token_ids'])
    if scores.shape!=(length,2) or len(offsets)!=length or not np.isfinite(scores).all():raise ValueError('Evidence logits provenance differs')
    best=None
    for start in range(1,length-1):
        end=start+int(np.argmax(scores[start:min(length-1,start+128),1]));score=float(scores[start,0]+scores[end,1])
        if best is None or score>best[0]:best=(score,start,end)
    if best is None:begin=finish=0;indices=[]
    else:begin=int(offsets[best[1]][0]);finish=int(offsets[best[2]][1]);indices=list(best[1:])
    if [begin,finish]!=[p['evidence_start'],p['evidence_end']] or indices!=raw['evidence_token_indices'] or p['evidence_text']!=text[begin:finish]:raise ValueError('Evidence pointer decode differs')
    expected=labels if confidence>=threshold and consistent and begin<finish else {k:'unknown' for k in labels}
    if any(p[k]!=v for k,v in expected.items()):raise ValueError('Effective prediction differs from raw model outputs')


def evidence(store,descriptor):
    raw=store.get(descriptor['archive'])
    if hashlib.sha256(raw).hexdigest()!=descriptor['archive_sha256']:raise ValueError('Audit archive SHA differs')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names=z.namelist()
        if len(set(names))!=len(names) or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n for n in names):raise ValueError('Unsafe evidence archive')
        if sum(i.file_size for i in z.infolist())>1024**3:raise ValueError('Evidence size bound exceeded')
        manifest=json.loads(z.read('MANIFEST.json'))
        if set(manifest)!=set(names)-{'MANIFEST.json'}:raise ValueError('Unaccounted evidence member')
        files={n:z.read(n) for n in manifest}
        if any(hashlib.sha256(files[n]).hexdigest()!=h for n,h in manifest.items()):raise ValueError('Evidence member SHA differs')
        if json.loads(files['RUN_RECEIPT.json'])['status']!='complete':raise ValueError('Scientific stage did not complete')
        return files


def diagnose_detector(value,rows):
    lookup={r['state_id']:r for r in rows};records=value['records']
    if len({r['state_id'] for r in records})!=len(records) or {r['state_id'] for r in records}!=set(lookup):raise ValueError('Detector validation coverage differs')
    confusion={k:Counter() for k in ['own_status','peer_status','event','affected']};by_condition=defaultdict(Counter);misses=[]
    for record in records:
        row=lookup[record['state_id']];prediction=record['prediction'];raw=prediction['raw']
        if record['labels']!=row['labels']:raise ValueError('Detector audit labels differ')
        m=EpisodeMemory();m.reset(row['case_id'],'visible-self','visible-peer');m.observe(row['case_id'],row['sequence'],Prediction(**prediction),row['text'],value['threshold'],row['terminal'])
        if m.raw_gate!=record['raw_gate']:raise ValueError('Saved gate differs from raw prediction replay')
        c=by_condition[(row['template'],row['condition'])];c['states']+=1;c['positives']+=row['labels']['gate'];c['missed_own']+=row['labels']['gate'] and not m.raw_gate;c['false_on']+=m.raw_gate and not row['labels']['gate']
        for key in confusion:
            actual=row['labels'][key];predicted=raw['labels'][key];confusion[key][(actual,predicted)]+=1;c[key+'_errors']+=actual!=predicted
        if (row['labels']['gate'] and not m.raw_gate) or m.raw_gate and not row['labels']['gate']:
            misses.append({'state_id':row['state_id'],'template':row['template'],'condition':row['condition'],'sequence':row['sequence'],'labels':row['labels'],'prediction':{k:v for k,v in prediction.items() if k!='raw'},'raw_labels':raw['labels'],'probabilities':raw['probabilities'],'head_consistent':raw['consistent'],'expected_evidence':row['text'][slice(*row['evidence_chars'])],'actual_excerpt':row['text'][:600]})
    return {'kind':'detector_diagnosis','threshold':value['threshold'],'states':len(records),'confusion':{k:[{'gold':a,'prediction':b,'count':v} for (a,b),v in sorted(c.items())] for k,c in confusion.items()},'by_condition':[{'template':t,'condition':name,**dict(v)} for (t,name),v in sorted(by_condition.items())],'gate_error_examples':misses,'gate_error_count':len(misses),'raw_gate_replay_pass':True,'model_forwards':0,'optimizer_updates':0,'heldout_content_read':False}


def audit_observation(world,observation,memory,threshold,bert_tokenizer=None,maximum_tokens=None):
    text,span=world.visible()
    encoded=None;context_failure=False
    if maximum_tokens is not None:
        if bert_tokenizer is None:raise ValueError('Context-budget replay requires the original tokenizer')
        encoded=bert_tokenizer(text,return_offsets_mapping=True,add_special_tokens=True,truncation=False)
        context_failure=len(encoded['input_ids'])>maximum_tokens
        if context_failure:world.running=False;world.reason='external_context_budget'
    if observation['text']!=text or observation['evaluator']!=world.truth() or observation['evidence_chars']!=list(span) or observation['sequence']!=world.sequence:raise ValueError('Observation/evidence replay differs')
    p=observation['memory']['prediction'];prediction=Prediction(**p)
    if context_failure:
        expected=Prediction('unknown','unknown','unknown','unknown',0.,0,0,'',{'context_budget_failure':True})
        if prediction!=expected:raise ValueError('External detector context failure was altered or treated as a model prediction')
    elif bert_tokenizer is not None:
        audit_prediction(p,text,threshold)
        if encoded is None:encoded=bert_tokenizer(text,return_offsets_mapping=True,add_special_tokens=True,truncation=False)
        if encoded['input_ids']!=p['raw'].get('input_token_ids') or [list(v) for v in encoded['offset_mapping']]!=p['raw'].get('offsets'):raise ValueError('Detector tokenizer provenance differs')
    memory.observe(world.scene.case_id,world.sequence,prediction,text,threshold,external_terminal=not world.running)
    if memory.snapshot()!=observation['memory']:raise ValueError('Persistent predicted state replay differs')


def audit_trajectory(record,threshold,bert_tokenizer=None,qwen_tokenizer=None,eos=None,kernel_profile='reference',context_limits=None):
    environment=record.get('environment','legacy-world')
    if environment not in ['legacy-world',ENVIRONMENT]:raise ValueError('Unknown recorded environment')
    world=(BoundedWorld if environment==ENVIRONMENT else World)(Scene(**record['scene']));memory=EpisodeMemory()
    memory.cache_epoch=record['initial_observation']['memory']['cache_epoch']-2
    if memory.cache_epoch<0:raise ValueError('Invalid initial cache epoch')
    memory.reset(world.scene.case_id,world.worker,world.peer)
    detector_limit=None if context_limits is None else context_limits['detector']
    audit_observation(world,record['initial_observation'],memory,threshold,bert_tokenizer,detector_limit);initial=memory.applied_gate
    for step in record['trace']:
        if not world.running:raise ValueError('Model called after external terminal')
        response=step['response'];mode=record['condition'];gate={'base':False,'always_on':True,'initial_gate':initial,'learned_dynamic':memory.applied_gate,'oracle_dynamic':world.truth()['gate']}[mode]
        if response.get('kernel_profile','reference')!=kernel_profile:raise ValueError('Response used another kernel profile')
        memory.begin_response()
        if response['memory_before_response']!=memory.snapshot() or response['selected_gate']!=gate or response['adapter_enabled']!=gate:raise ValueError('Response routing or memory differs')
        if response['predicted_gate']!=memory.raw_gate or response['evaluator_gate']!=world.truth()['gate']:raise ValueError('Response evaluator log differs')
        if qwen_tokenizer is not None:
            expected_ids=input_ids(qwen_tokenizer,world.visible()[0])
            if context_limits is not None and len(expected_ids)>context_limits['actor']:raise ValueError('Executed actor context exceeded its frozen ceiling')
            if response['input_token_ids']!=expected_ids:raise ValueError('Qwen received another prompt')
            if response['raw_text']!=decode(qwen_tokenizer,response['generated_token_ids'],eos):raise ValueError('Native response token decode differs')
            ended=bool(response['generated_token_ids'] and response['generated_token_ids'][-1] in eos)
            if ended!=response['finished_with_eos']:raise ValueError('Native EOS provenance differs')
            try:actions=parse(response['raw_text'],1) if ended else None
            except (ValueError,TypeError,RecursionError):actions=None
            if actions!=response['actions']:raise ValueError('Native response parser differs')
        if not response['cache']['prefill_checked'] or response['cache']['initial_cache_length']!=0:raise ValueError('Response cache boundary not verified')
        action=response['actions'][0] if response['actions'] else None
        if action!=step['executed_action']:raise ValueError('Evaluator replaced native action')
        memory.end_response();reply=world.step(action)
        if reply!=step['reply']:raise ValueError('Tool execution reply replay differs')
        audit_observation(world,step['observation_after'],memory,threshold,bert_tokenizer,detector_limit)
        if world.outcome()!=step['outcome']:raise ValueError('Tool execution outcome replay differs after observation')
    if world.running and context_limits is not None and qwen_tokenizer is not None and len(input_ids(qwen_tokenizer,world.visible()[0]))>context_limits['actor']:
        world.running=False;world.reason='external_context_budget';memory.terminal=True;memory.applied_gate=False
    if world.running or world.outcome()!=record['outcome'] or memory.snapshot()!=record['terminal_memory']:raise ValueError('Incomplete or inconsistent terminal trajectory')
    # Independent task arithmetic, in addition to replay through the simulator.
    for operation in world.operations:
        call=operation['call']
        if isinstance(call,dict) and call.get('tool')=='submit' and operation['reply'].get('status')=='accepted':
            values=[world.values[k] for k in call['arguments']['evidence']]
            total=max(values,default=0) if world.scene.domain=='max' else sum(values)
            if total!=call['arguments']['total']:raise ValueError('Independent report arithmetic differs')
    return {'case_id':world.scene.case_id,'condition':record['condition'],'responses':len(record['trace']),'pass':True}
