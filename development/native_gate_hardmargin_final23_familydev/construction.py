"""One default-deny owned family-development job, at most five optimizations."""
import argparse,os,time
from pathlib import Path
import gate,checker,source_auth
HERE=Path(__file__).resolve().parent
CORE_FILES=('gate.py','checker.py','source_auth.py','construction.py')
FAMILIES=('G01','G02','G03','G04')
LIMITS={'worker_seconds':60,'cleanup_seconds':5,'maximum_fits':5,'iterations_per_fit':10000,
    'total_bytes':8*1024**2,'per_file_bytes':5*1024**2,'worker_payload_bytes':1024**2,
    'owner_reservation_bytes':393216,'combined_reservation_bytes':1441792}
def schedule():
    names=source_auth.keys();out=[]
    for f in FAMILIES:
        held=[i for i,k in enumerate(names) if k.startswith(f+'_')]
        train=[i for i in range(32) if i not in held]
        gate.require(len(held)==6 and len(train)==26,'EXACT_FOLD_COUNTS')
        out.append({'id':f,'train':train,'held':held})
    return out+[{'id':'FULL32','train':list(range(32)),'held':[]}]
def input_contract(binding=None):
    binding=source_auth.frozen_binding() if binding is None else binding
    gate.require(set(binding)=={'manifest_sha256','feature_sha256'},'EXACT_INPUT_BINDING_FIELDS')
    return {'schema':'hardmargin_familydev_inputs.v1','role':'EXPOSED_DEVELOPMENT_ONLY',
        'manifest_sha256':binding['manifest_sha256'],'feature_sha256':binding['feature_sha256'],
        'keys':source_auth.keys(),'labels':source_auth.expected_labels(),'stages':schedule(),
        'capture_namespace':'development/native_supervised_gate_capture_final23_v1',
        'prior_layer10_attempt':'native_gate_hardmargin_familydev/construction_attempt_001',
        'starting_weights':'none'}
def source_lock():return {n:gate.digest((HERE/n).read_bytes()) for n in CORE_FILES}
def lock(inputs,sources):
    return {'schema':'hardmargin_familydev_contract.v1','method':gate.METHOD,'feature_contract':gate.CONTRACT,
        'limits':LIMITS,'inputs_sha256':gate.digest(inputs),'sources_sha256':gate.digest(sources),
        'scope':'ONE_OWNED_FAMILYDEV_ATTEMPT_MAX5_OPTIMIZATIONS','fold_order':FAMILIES,
        'requires':'each26train+6held_then_full32_allcorrect_with_certificates','retry':False,
        'method_selection':'one_prospectively_fixed_final_decoder_block23_readout;same_hardmargin_familydev_method',
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
    binding=source_auth.input_binding()  # future saved-only content binding; requires a closed valid final23 capture.
    inputs=gate.canonical(input_contract(binding));sources=gate.canonical(source_lock());contract=gate.canonical(lock(inputs,sources))
    release=gate.canonical({'schema':'hardmargin_familydev_release.v1','approved':False,
        'operation':'one_construction_fit','construction_lock_sha256':gate.digest(contract),
        'training_manifest_sha256':gate.digest(inputs),'source_sha256':gate.digest(sources),
        'model_permission':False,'tokenizer_permission':False,'evaluation_permission':False})
    for n,v in zip(names,(inputs,sources,contract,release),strict=True):write_new(HERE/n,v)
    return {'status':'METADATA_ONLY_DEFAULT_DENY','maximum_optimizations':5}
def authorize(release_raw,expected_sha):
    gate.require(type(expected_sha) is str and len(expected_sha)==64 and gate.digest(release_raw)==expected_sha,'ROOT_RELEASE_HASH')
    release=gate.decode(release_raw);gate.require(release.get('approved') is True,'RELEASE_DEFAULT_DENY')
    inputs=(HERE/'TRAINING_MANIFEST.json').read_bytes();sources=(HERE/'CORE_SOURCE_LOCK.json').read_bytes()
    contract=(HERE/'CONSTRUCTION_LOCK_DRAFT.json').read_bytes()
    expected={'schema':'hardmargin_familydev_release.v1','approved':True,'operation':'one_construction_fit',
        'construction_lock_sha256':gate.digest(contract),'training_manifest_sha256':gate.digest(inputs),
        'source_sha256':gate.digest(sources),'model_permission':False,'tokenizer_permission':False,'evaluation_permission':False}
    gate.require(release==expected and gate.canonical(gate.decode(inputs))==gate.canonical(input_contract())
        and gate.decode(sources)==source_lock() and gate.decode(contract)==gate.decode(gate.canonical(lock(inputs,sources))),
        'EXACT_FAMILYDEV_RELEASE_AND_CONTRACT')
    return expected
def confusion(scores,labels):
    return {k:sum((('TP' if y==1 else 'FP') if s>0 else ('FN' if y==1 else 'TN'))==k
        for s,y in zip(scores,labels,strict=True)) for k in ('TP','FN','TN','FP')}
def run(rows,labels,*,deadline,solve_fn=None,publish=None):
    solver=gate.solve if solve_fn is None else solve_fn
    gate.require(len(rows)==32 and tuple(labels)==source_auth.expected_labels(),'EXACT32_DEVELOPMENT_SET')
    rows=gate.finite_rows(rows);stages=[{**s,'status':'UNRUN'} for s in schedule()];attempts=0;final_model=None
    result={'schema':'hardmargin_familydev_result.v1','role':'EXPOSED_DEVELOPMENT_ONLY','scientific_pass':False,
        'status':'UNRUN','fits_attempted':0,'stages':stages,'model_calls':0,'tokenizer_calls':0,'checkpoint_tensor_reads':0}
    for stage in stages:
        try:
            gate.require(time.monotonic()<deadline,'SHARED_WORKER_DEADLINE')
            train=[rows[i] for i in stage['train']];y=tuple(labels[i] for i in stage['train'])
            mu,x=gate.preprocess(train)
            # Held rows enter only their own scoring transform, never training preprocessing.
            gate.require(attempts<5,'MAXIMUM_FIVE_FITS')
            attempts+=1;result['fits_attempted']=attempts;stage['status']='ATTEMPTED'
            if publish:publish('FIT_ATTEMPT_'+str(attempts)+'.json',{'stage':stage['id'],'attempt':attempts,'maximum':5})
            solved=solver(x,y,deadline=deadline,max_iterations=10000)
            cert=checker.verify(x,y,solved['w'],solved['b'],solved['alpha'])
            model=gate.Gate(mu,tuple(solved['w']),solved['b'])
            train_scores=[model.score(r) for r in train];held_scores=[model.score(rows[i]) for i in stage['held']]
            held_y=tuple(labels[i] for i in stage['held'])
            train_correct=sum((s>0)==(v==1) for s,v in zip(train_scores,y))
            held_correct=sum((s>0)==(v==1) for s,v in zip(held_scores,held_y))
            gate.require(time.monotonic()<deadline,'SHARED_WORKER_DEADLINE')
            stage.update(status='PASS',iterations=solved['iterations'],training_mean=list(mu),
                parameters={'w':list(solved['w']),'b':solved['b'],'alpha':list(solved['alpha'])},
                training_count=len(train),held_count=len(held_y),training_correct=train_correct,held_correct=held_correct,
                training_scores=train_scores,held_scores=held_scores,training_confusion=confusion(train_scores,y),
                held_confusion=confusion(held_scores,held_y),certificate=cert,
                training_margins=[v*s for v,s in zip(y,train_scores)],held_margins=[v*s for v,s in zip(held_y,held_scores)])
            if train_correct!=len(train):
                stage['status']='TRAINING_ROUTING_FAIL';result['status']='SCIENTIFIC_TRAINING_FAIL';break
            if held_correct!=len(held_y):
                stage['status']='HELD_FAMILY_FAIL';result['status']='SCIENTIFIC_HELD_FAMILY_FAIL';break
            if stage['id']=='FULL32':final_model=model;result.update(status='DEVELOPMENT_PASS',scientific_pass=True)
        except gate.OptimizationFailure as error:
            stage.update(status=error.classification,code=error.code,iterations=error.iterations)
            result['status']='SCIENTIFIC_TRAINING_INFEASIBLE' if error.classification=='CERTIFIED_INFEASIBLE' else 'TECHNICAL_UNCERTIFIED';break
        except Exception as error:
            stage.update(status='TECHNICAL_FAILURE',exception_type=type(error).__name__)
            result['status']='TECHNICAL_UNCERTIFIED';break
    return result,final_model
def construct(release_path,release_sha256):
    deadline=time.monotonic()+60.
    release=authorize(Path(release_path).read_bytes(),release_sha256)
    destination=HERE/'construction_attempt_001';destination.mkdir(exist_ok=False)
    result={'schema':'hardmargin_familydev_result.v1','role':'EXPOSED_DEVELOPMENT_ONLY',
        'status':'TECHNICAL_UNCERTIFIED','scientific_pass':False,'fits_attempted':0,
        'stages':[{**s,'status':'UNRUN'} for s in schedule()],
        'model_calls':0,'tokenizer_calls':0,'checkpoint_tensor_reads':0}
    try:
        rows,labels=source_auth.load_saved(deadline=deadline)
        gate.require(all(len(r)==1024 for r in rows),'NATIVE_ARTIFACT_WIDTH')
        result,model=run(rows,labels,deadline=deadline,publish=lambda n,v:write_new(destination/n,v))
        if result['scientific_pass']:
            bindings={'input_contract_sha256':release['training_manifest_sha256'],
                'construction_lock_sha256':release['construction_lock_sha256'],'source_sha256':release['source_sha256'],
                'feature_sha256':gate.decode((HERE/'TRAINING_MANIFEST.json').read_bytes())['feature_sha256'],'development_only':True}
            raw=gate.artifact(model,bindings);loaded=gate.load_artifact(raw,expected_sha256=gate.digest(raw),expected_bindings=bindings)
            gate.require([model.score(r) for r in rows]==[loaded.score(r) for r in rows],'ARTIFACT_RELOAD_EXACT')
            gate.require(time.monotonic()<deadline,'SHARED_WORKER_DEADLINE')
            write_new(destination/'FITTED_GATE.json',raw);result['artifact_sha256']=gate.digest(raw)
        write_new(destination/'RESULT.json',result)
        return result
    except Exception as error:
        result.update(scientific_pass=False,status='TECHNICAL_UNCERTIFIED',exception_type=type(error).__name__)
        write_new(destination/'TECHNICAL_FAULT.json',result);raise
def output_summary(result):
    return {'status':result['status'],'scientific_pass':result.get('scientific_pass',False),
        'fits_attempted':result.get('fits_attempted',0),
        'result_path':str(HERE/'construction_attempt_001/RESULT.json'),
        'result_sha256':gate.digest(gate.canonical(result))}
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('candidate','fit'));p.add_argument('--release');p.add_argument('--release-sha256')
    a=p.parse_args()
    if a.command=='candidate':
        gate.require(a.release is None and a.release_sha256 is None,'CANDIDATE_NO_RELEASE');result=candidate()
    else:
        gate.require(a.release and a.release_sha256,'ROOT_RELEASE_REQUIRED');result=construct(a.release,a.release_sha256)
    print(gate.canonical(output_summary(result) if a.command=='fit' else result).decode('ascii'))
    return 0
if __name__=='__main__':raise SystemExit(main())
