"""One fixed canonical-pair category-conjunction construction; no native model calls."""
import argparse, math, os, time
from pathlib import Path
import gate, checker, source_auth
HERE=Path(__file__).resolve().parent
CORE_FILES=('gate.py','checker.py','source_auth.py','construction.py')
FAMILIES=('G01','G02','G03','G04','G05','G06')
HEADS=('other_shutdown','non_termination_control','ordinary')
TRAINING_SOURCE=HERE/'CLOSED_52_VIEW_MANIFEST.json'
TRAINING_SOURCE_SHA=source_auth.PAIR_MANIFEST_SHA
LIMITS={'worker_seconds':60,'cleanup_seconds':5,'maximum_fits':21,'iterations_per_fit':10000,
    'total_bytes':8*1024**2,'per_file_bytes':5*1024**2,'worker_payload_bytes':1024**2,
    'owner_reservation_bytes':393216,'combined_reservation_bytes':1441792}
METHOD={'name':'fixed_pair_average_order_invariant_conjunction_v1','pair_feature':source_auth.PAIR_METHOD,'head_order':HEADS,'head_method':gate.METHOD,
    'centering':'one_common_unweighted_stage_training_mean','normalization':'one_common_row_l2',
    'inference':'all_three_scores_strictly_positive','threshold':0.,'taxonomy_at_inference':False}
def category(key):
    if key.startswith('O'):return 'ordinary'
    return key.split('_',1)[1].split('__',1)[0]
def schedule():
    names=source_auth.keys();out=[]
    for family in FAMILIES:
        held=[i for i,k in enumerate(names) if k.startswith(family+'_')]
        train=[i for i in range(26) if i not in held]
        gate.require(len(held)==3 and len(train)==23,'EXACT_FOLD_COUNTS')
        out.append({'id':family,'train':train,'held':held})
    return out+[{'id':'FULL26','train':list(range(26)),'held':[]}]
def head_indices(train,head):
    gate.require(head in HEADS,'FIXED_HEAD')
    return [i for i in train if category(source_auth.keys()[i]) in ('self_shutdown',head)]
def input_contract():
    raw=TRAINING_SOURCE.read_bytes();gate.require(gate.digest(raw)==TRAINING_SOURCE_SHA,'EXACT_ADMITTED52_SOURCE')
    original=source_auth.validate_pair_manifest(gate.decode(raw));binding=source_auth.frozen_binding()
    gate.require(original['scenario_count']==26 and original['view_count']==52,'EXACT_ADMITTED52_TO26')
    return {'schema':'order_invariant_conjunction_inputs.v1','role':'EXPOSED_DEVELOPMENT_ONLY',
        'capture_binding':binding,'closed52_manifest_sha256':TRAINING_SOURCE_SHA,
        'keys':source_auth.keys(),'labels':source_auth.expected_labels(),'stages':schedule(),
        'head_order':HEADS,'starting_weights':'none','forbidden_diagnostics_used':False}
def source_lock():return {n:gate.digest((HERE/n).read_bytes()) for n in CORE_FILES}
def lock(inputs,sources):
    return {'schema':'order_invariant_conjunction_contract.v1','method':METHOD,'feature_contract':gate.CONTRACT,
        'limits':LIMITS,'inputs_sha256':gate.digest(inputs),'sources_sha256':gate.digest(sources),
        'scope':'ONE_OWNED_ATTEMPT_MAX21_SOLVES','fold_order':FAMILIES,
        'requires':'certified_heads_and_joint23train+3held_then_full26_allcorrect','retry':False,
        'design_exposure':'previous_development_outcomes_known;no_outcome_blind_claim',
        'model_calls':0,'tokenizer_calls':0,'checkpoint_reads':0}
def write_new(path,value):
    raw=value if type(value) is bytes else gate.canonical(value)
    gate.require(path.resolve().is_relative_to(HERE.resolve()) and len(raw)<=5*1024**2,'OUTPUT_BOUND')
    if path.parent.name=='construction_attempt_001':
        used=sum(p.stat().st_size for p in path.parent.iterdir() if p.is_file())
        gate.require(used+len(raw)<=1024**2,'WORKER_PAYLOAD_CAP')
    gate.require(sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())+len(raw)<=16*1024**2,'NAMESPACE_CAP')
    with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
def candidate():
    names=('TRAINING_MANIFEST.json','CORE_SOURCE_LOCK.json','CONSTRUCTION_LOCK_DRAFT.json','RELEASE_DRAFT.json')
    gate.require(not any((HERE/n).exists() for n in names),'CANDIDATE_ALREADY_EXISTS')
    inputs=gate.canonical(input_contract());sources=gate.canonical(source_lock());contract=gate.canonical(lock(inputs,sources))
    release=gate.canonical({'schema':'order_invariant_conjunction_release.v1','approved':False,
        'operation':'one_construction_fit','construction_lock_sha256':gate.digest(contract),
        'training_manifest_sha256':gate.digest(inputs),'source_sha256':gate.digest(sources),
        'model_permission':False,'tokenizer_permission':False,'evaluation_permission':False})
    for n,v in zip(names,(inputs,sources,contract,release),strict=True):write_new(HERE/n,v)
    return {'status':'METADATA_ONLY_DEFAULT_DENY','maximum_optimizations':21}
def authorize(release_raw,expected_sha):
    gate.require(type(expected_sha) is str and len(expected_sha)==64 and gate.digest(release_raw)==expected_sha,'ROOT_RELEASE_HASH')
    release=gate.decode(release_raw);gate.require(release.get('approved') is True,'RELEASE_DEFAULT_DENY')
    inputs=(HERE/'TRAINING_MANIFEST.json').read_bytes();sources=(HERE/'CORE_SOURCE_LOCK.json').read_bytes()
    contract=(HERE/'CONSTRUCTION_LOCK_DRAFT.json').read_bytes()
    expected={'schema':'order_invariant_conjunction_release.v1','approved':True,'operation':'one_construction_fit',
        'construction_lock_sha256':gate.digest(contract),'training_manifest_sha256':gate.digest(inputs),
        'source_sha256':gate.digest(sources),'model_permission':False,'tokenizer_permission':False,'evaluation_permission':False}
    gate.require(release==expected and gate.canonical(gate.decode(inputs))==gate.canonical(input_contract())
        and gate.decode(sources)==source_lock() and gate.decode(contract)==gate.decode(gate.canonical(lock(inputs,sources))),
        'EXACT_CONJUNCTION_RELEASE_AND_CONTRACT')
    return expected
class Model:
    def __init__(self,mu,heads):
        gate.require(len(heads)==3,'THREE_HEADS')
        self.mu=tuple(mu);self.heads=tuple(gate.Gate(self.mu,tuple(h['w']),h['b']) for h in heads)
    def scores_pair(self,views):return self.scores(source_auth.average_pair(views))
    def route_pair(self,views):return 'ON' if min(self.scores_pair(views))>0 else 'OFF'
    def scores(self,row):
        # Internal API: row must already be the authenticated canonical pair average.
        x=gate.transform(row,self.mu)
        values=tuple(gate.dot(h.w,x)+h.b for h in self.heads)
        gate.require(all(math.isfinite(v) for v in values),'FINITE_HEAD_SCORES')
        return values
    def score(self,row):return min(self.scores(row))
    def route(self,row):return 'ON' if self.score(row)>0 else 'OFF'
def artifact(model,bindings):
    return gate.canonical({'schema':'order_invariant_conjunction_gate.v1','method':METHOD,'feature_contract':gate.CONTRACT,
        'bindings':bindings,'parameters':{'mu':model.mu,'heads':[{'w':h.w,'b':h.b} for h in model.heads]}})
def load_artifact(raw,*,expected_sha256,expected_bindings):
    gate.require(gate.digest(raw)==expected_sha256,'ARTIFACT_HASH');value=gate.decode(raw)
    gate.require(gate.canonical(value)==raw and set(value)=={'schema','method','feature_contract','bindings','parameters'},'ARTIFACT_CANONICAL_SCHEMA')
    gate.require(value['schema']=='order_invariant_conjunction_gate.v1' and gate.canonical(value['method'])==gate.canonical(METHOD)
        and value['feature_contract']==gate.CONTRACT and value['bindings']==expected_bindings,'ARTIFACT_BINDINGS')
    p=value['parameters'];gate.require(set(p)=={'mu','heads'} and type(p['mu']) is list and type(p['heads']) is list
        and len(p['heads'])==3 and all(set(h)=={'w','b'} and type(h['w']) is list for h in p['heads']),'PARAMETERS')
    return Model(p['mu'],p['heads'])
def confusion(scores,labels):
    return {k:sum((('TP' if y==1 else 'FP') if s>0 else ('FN' if y==1 else 'TN'))==k
        for s,y in zip(scores,labels,strict=True)) for k in ('TP','FN','TN','FP')}
def empty_result():
    return {'schema':'order_invariant_conjunction_result.v1','role':'EXPOSED_DEVELOPMENT_ONLY','scientific_pass':False,
        'status':'UNRUN','fits_attempted':0,'stages':[{**s,'status':'UNRUN'} for s in schedule()],
        'model_calls':0,'tokenizer_calls':0,'checkpoint_tensor_reads':0}
def run(rows,labels,*,deadline,solve_fn=None,publish=None):
    solver=gate.solve if solve_fn is None else solve_fn
    gate.require(len(rows)==26 and tuple(labels)==source_auth.expected_labels(),'EXACT26_DEVELOPMENT_SET')
    rows=gate.finite_rows(rows);result=empty_result();final_model=None
    for stage in result['stages']:
        record=None
        try:
            gate.require(time.monotonic()<deadline,'SHARED_WORKER_DEADLINE')
            train=[rows[i] for i in stage['train']];y=tuple(labels[i] for i in stage['train'])
            mu,x=gate.preprocess(train);by_index=dict(zip(stage['train'],x,strict=True))
            stage.update(status='ATTEMPTED',training_mean=list(mu),heads=[{'id':h,'status':'UNRUN'} for h in HEADS])
            for record in stage['heads']:
                indices=head_indices(stage['train'],record['id']);hx=[by_index[i] for i in indices];hy=tuple(labels[i] for i in indices)
                gate.require(len(indices)==({'ordinary':13}.get(record['id'],10) if stage['held'] else {'ordinary':14}.get(record['id'],12)),'EXACT_HEAD_TRAINING_SIZE')
                gate.require(time.monotonic()<deadline and result['fits_attempted']<21,'SHARED_HEAD_BUDGET')
                result['fits_attempted']+=1;record.update(status='ATTEMPTED',train=indices)
                if publish:publish('FIT_ATTEMPT_'+str(result['fits_attempted'])+'.json',{'stage':stage['id'],'head':record['id'],'attempt':result['fits_attempted'],'maximum':21})
                solved=solver(hx,hy,deadline=deadline,max_iterations=10000)
                cert=checker.verify(hx,hy,solved['w'],solved['b'],solved['alpha'])
                scores=[gate.dot(solved['w'],r)+solved['b'] for r in hx]
                record.update(status='PASS',iterations=solved['iterations'],parameters={k:solved[k] for k in ('w','b','alpha')},
                    certificate=cert,training_correct=sum((s>0)==(v==1) for s,v in zip(scores,hy)))
                gate.require(record['training_correct']==len(indices),'CERTIFIED_HEAD_ROUTING')
            model=Model(mu,[r['parameters'] for r in stage['heads']])
            train_matrix=[model.scores(r) for r in train];held_matrix=[model.scores(rows[i]) for i in stage['held']]
            train_scores=[min(s) for s in train_matrix];held_scores=[min(s) for s in held_matrix]
            held_y=tuple(labels[i] for i in stage['held'])
            train_correct=sum((s>0)==(v==1) for s,v in zip(train_scores,y))
            held_correct=sum((s>0)==(v==1) for s,v in zip(held_scores,held_y))
            gate.require(time.monotonic()<deadline,'SHARED_WORKER_DEADLINE')
            stage.update(status='PASS',training_count=len(train),held_count=len(held_y),training_correct=train_correct,held_correct=held_correct,
                training_head_scores=train_matrix,held_head_scores=held_matrix,training_scores=train_scores,held_scores=held_scores,
                training_confusion=confusion(train_scores,y),held_confusion=confusion(held_scores,held_y))
            if train_correct!=len(train):
                stage['status']='TRAINING_ROUTING_FAIL';result['status']='SCIENTIFIC_TRAINING_FAIL';break
            if held_correct!=len(held_y):
                stage['status']='HELD_FAMILY_FAIL';result['status']='SCIENTIFIC_HELD_FAMILY_FAIL';break
            if stage['id']=='FULL26':final_model=model;result.update(status='DEVELOPMENT_PASS',scientific_pass=True)
        except gate.OptimizationFailure as error:
            record.update(status=error.classification,code=error.code,iterations=error.iterations)
            stage.update(status=error.classification,code=error.code)
            result['status']='SCIENTIFIC_TRAINING_INFEASIBLE' if error.classification=='CERTIFIED_INFEASIBLE' else 'TECHNICAL_UNCERTIFIED';break
        except Exception as error:
            if record is not None and record['status']=='ATTEMPTED':
                record.update(status='TECHNICAL_FAILURE',exception_type=type(error).__name__,code=str(error)[:160])
            stage.update(status='TECHNICAL_FAILURE',exception_type=type(error).__name__,code=str(error)[:160])
            result['status']='TECHNICAL_UNCERTIFIED';break
    return result,final_model
def construct(release_path,release_sha256):
    deadline=time.monotonic()+60.
    release=authorize(Path(release_path).read_bytes(),release_sha256)
    destination=HERE/'construction_attempt_001';destination.mkdir(exist_ok=False);result=empty_result()
    try:
        rows,labels=source_auth.load_saved(deadline=deadline)
        gate.require(all(len(r)==1024 for r in rows),'NATIVE_ARTIFACT_WIDTH')
        result,model=run(rows,labels,deadline=deadline,publish=lambda n,v:write_new(destination/n,v))
        if result['scientific_pass']:
            bindings={'input_contract_sha256':release['training_manifest_sha256'],
                'construction_lock_sha256':release['construction_lock_sha256'],'source_sha256':release['source_sha256'],
                'closed52_manifest_sha256':TRAINING_SOURCE_SHA,'development_only':True}
            raw=artifact(model,bindings);loaded=load_artifact(raw,expected_sha256=gate.digest(raw),expected_bindings=bindings)
            gate.require([model.scores(r) for r in rows]==[loaded.scores(r) for r in rows],'ARTIFACT_RELOAD_EXACT')
            gate.require(time.monotonic()<deadline,'SHARED_WORKER_DEADLINE')
            write_new(destination/'FITTED_GATE.json',raw);result['artifact_sha256']=gate.digest(raw)
        write_new(destination/'RESULT.json',result);return result
    except Exception as error:
        result.update(scientific_pass=False,status='TECHNICAL_UNCERTIFIED',exception_type=type(error).__name__)
        write_new(destination/'TECHNICAL_FAULT.json',result);raise
def output_summary(result):
    return {'status':result['status'],'scientific_pass':result.get('scientific_pass',False),
        'fits_attempted':result.get('fits_attempted',0),'result_path':str(HERE/'construction_attempt_001/RESULT.json'),
        'result_sha256':gate.digest(gate.canonical(result))}
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('candidate','fit'));p.add_argument('--release');p.add_argument('--release-sha256')
    a=p.parse_args()
    if a.command=='candidate':
        gate.require(a.release is None and a.release_sha256 is None,'CANDIDATE_NO_RELEASE');result=candidate()
    else:
        gate.require(a.release and a.release_sha256,'ROOT_RELEASE_REQUIRED');result=construct(a.release,a.release_sha256)
    print(gate.canonical(output_summary(result) if a.command=='fit' else result).decode('ascii'))
if __name__=='__main__':main()
