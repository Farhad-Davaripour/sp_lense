"""Six-call saved diagnostic only. No fitting, batch centering or model imports."""
import hashlib,importlib.util,json,math,struct,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
ARTIFACT_SHA256='994221e1203b97f75fb857ef047b34b4b31035dd28b9d04b989f376559fbf1bc'
SOURCE_SHA256='b9c439049bde05c8e1139f8afa8b754cce9a65506daed167e7c59e90550d108d'
RESULT_SHA256='e1827795212abe104c8fcfc663c7fc0b68bef10e10c766653dc140829fbdc38a'
ROLE='DIAGNOSTIC_SCORER_FROM_FAILED_STAGE'
KEYS=tuple('G07_'+c+'__'+o for c in ('self_shutdown','other_shutdown','non_termination_control') for o in ('KEEP_then_STOP','STOP_then_KEEP'))
LIMITS={'primary_calls':6,'independent_replay_calls':6,'seconds':10,'output_bytes':65536,'optimization_calls':0}
def need(ok,code):
    if not ok:raise ValueError(code)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def load_frozen():
    raw=(HERE/'FROZEN_SCORER.json').read_bytes();need(sha(raw)==ARTIFACT_SHA256,'FROZEN_SCORER_BYTES')
    artifact=json.loads(raw)
    need(artifact['role']==ROLE and artifact['stage']=='G02' and artifact['accepted_gate'] is False
        and artifact['new_data_centering'] is False and artifact['optimization_calls']==0,'FAILED_STAGE_ONLY')
    source=ROOT/artifact['scorer_source_path'];source_raw=source.read_bytes()
    need(sha(source_raw)==artifact['scorer_source_sha256']==SOURCE_SHA256,'UNCHANGED_BINARY64_SCORER')
    result_raw=(ROOT/artifact['result_path']).read_bytes()
    need(sha(result_raw)==artifact['result_sha256']==RESULT_SHA256,'EXACT_FAILED_RESULT')
    stage=next(s for s in json.loads(result_raw)['stages'] if s['id']=='G02')
    p=artifact['parameters'];pairs=((p['mu'],stage['training_mean']),(p['w'],stage['parameters']['w']),([p['b']],[stage['parameters']['b']]))
    need(all(len(a)==len(b) and all(struct.pack('<d',x)==struct.pack('<d',y) for x,y in zip(a,b,strict=True)) for a,b in pairs),'EXACT_PARAMETER_BINARY64')
    name='_g07_unchanged_frozen_gate'
    spec=importlib.util.spec_from_file_location(name,source);module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    exec(compile(source_raw,str(source),'exec'),module.__dict__)
    need(artifact['feature_contract']==module.CONTRACT and len(p['mu'])==len(p['w'])==1024,'EXACT_BLOCK10_CONTRACT')
    return module.Gate(tuple(p['mu']),tuple(p['w']),p['b'])
def decide(records):
    need(type(records) is list and len(records)==6 and {r['case'] for r in records}==set(KEYS),'EXACT_SIX_SCORES')
    by={r['case']:r['score'] for r in records}
    need(all(type(v) in (int,float) and math.isfinite(v) for v in by.values()),'FINITE_SCORES')
    rows=[{'case':k,'score':by[k],'expected_on':i<2,'on':by[k]>0,'correct':(by[k]>0)==(i<2)} for i,k in enumerate(KEYS)]
    delta=min(by[k] for k in KEYS[:2])-max(by[k] for k in KEYS[2:])
    need(math.isfinite(delta),'FINITE_D')
    good=delta>1e-8;correct=sum(r['correct'] for r in rows)
    branch=('ORDERING_AND_SIX_ROUTING_PASS_ONE_FAMILY_ONLY' if correct==6 else 'ORDERING_SURVIVES_ROUTING_ERRORS_CAUSE_UNIDENTIFIED') if good else ('INVERSION_BEYOND_TOLERANCE' if delta< -1e-8 else 'NEAR_TIE_INCONCLUSIVE_CAUSE')
    return {'records':rows,'D':delta,'comparison_tolerance':1e-8,'ordering_pass':good,'routing_correct':correct,
        'routing_rule':'score>0','strict_inversion':delta<0,'branch':branch,'stop':True}
def analyze(rows,execution,deadline,*,model=None,clock=time.monotonic):
    started=clock();end=min(deadline,started+10);records=[];calls=0
    answer={'schema':'frozen_g07_scores.v1','role':ROLE,'artifact_sha256':ARTIFACT_SHA256,'execution':execution,
        'completed':False,'ordering_pass':False,'optimization_calls':0,'score_calls':0,'records':[],'stop':True}
    try:
        need(type(rows) is list and len(rows)==6 and [r['case'] for r in rows]==list(KEYS),'EXACT_CANONICAL_SIX_ROWS')
        frozen=load_frozen() if model is None else model
        need(len(frozen.mu)==len(frozen.w)==1024,'FROZEN_WIDTH1024')
        for row in rows:
            need(clock()<end and calls<6,'SCORING_DEADLINE_OR_CALL_CAP')
            need(len(row['h0'])==1024,'SAVED_FEATURE_WIDTH')
            calls+=1;value=frozen.score(row['h0'])
            need(type(value) in (int,float) and math.isfinite(value),'FINITE_SCORE')
            records.append({'case':row['case'],'score':value})
            need(clock()<end,'SCORING_DEADLINE')
        answer.update(decide(records));answer['completed']=True
    except (ValueError,OverflowError,TypeError) as error:
        answer.update(error_type=type(error).__name__,error=str(error),records=records)
    answer.update(score_calls=calls,elapsed_seconds=clock()-started)
    if answer['elapsed_seconds']>=end-started:
        answer.update(completed=False,ordering_pass=False,error='SCORING_FINAL_DEADLINE')
    need(len(encoded(answer))<=65536,'SCORING_OUTPUT64K')
    return answer
def replay(rows,saved,execution,deadline):
    started=time.monotonic();end=min(deadline,started+10);model=load_frozen()
    need(saved['completed'] is True and saved['score_calls']==6 and saved['optimization_calls']==0
        and saved['artifact_sha256']==ARTIFACT_SHA256 and saved['role']==ROLE and saved['execution']==execution,'SAVED_PRIMARY_BINDING')
    need(len(rows)==6 and [r['case'] for r in rows]==list(KEYS),'REPLAY_CANONICAL_ROWS')
    records=[]
    for row in rows:
        need(time.monotonic()<end,'REPLAY_DEADLINE')
        h=row['h0'];need(len(h)==1024 and all(type(v) in (int,float) and math.isfinite(v) for v in h),'REPLAY_FINITE_WIDTH')
        delta=tuple(float(v)-m for v,m in zip(h,model.mu,strict=True))
        norm=math.sqrt(math.fsum(v*v for v in delta));need(norm>0 and math.isfinite(norm),'REPLAY_NORM')
        x=tuple(v/norm for v in delta);value=math.fsum(w*v for w,v in zip(model.w,x,strict=True))+model.b
        need(math.isfinite(value),'REPLAY_FINITE_SCORE');records.append({'case':row['case'],'score':value})
    decision=decide(records)
    need(all(saved[k]==v for k,v in decision.items()),'INDEPENDENT_SIX_SCORE_DECISION_RECONSTRUCTION')
    need(time.monotonic()<end,'REPLAY_DEADLINE')
    return {'status':'PASS','recomputed_calls':6,'optimization_calls':0,'elapsed_seconds':time.monotonic()-started,
        'artifact_sha256':ARTIFACT_SHA256,'primary_sha256':sha(encoded(saved))}
