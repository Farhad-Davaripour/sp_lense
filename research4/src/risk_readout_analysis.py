"""Readout fitting on Databricks and separate zero-fit arithmetic audit."""
import json,warnings
from pathlib import Path
import numpy as np
from .io_utils import read,dump,sha,digest
from .risk_readout_data import dataset,normalize,metrics,admitted
from .native27_tools import encode
from .risk_readout_worker import CONFIG


def load_inputs(folder,root):
    from transformers import AutoTokenizer
    folder,root=Path(folder),Path(root);freeze=read(folder/'PRECALL_FREEZE.json');cfg=read(root/CONFIG)
    if freeze['config']!=cfg or freeze['source_commit']!=read(root/'RELEASE.json')['commit']:raise ValueError('Prospective freeze changed')
    pin=read(root/'inputs/MODEL_PIN.json')
    if pin!=freeze['pin'] or freeze['input_manifest_sha256']!=sha(root/'inputs/MANIFEST.json'):raise ValueError('Model identity changed')
    rows=read(folder/'ROWS.json')
    if rows!=dataset(cfg):raise ValueError('Factual history/label replay differs')
    manifest=read(folder/'MANIFEST.json')
    for name,value in manifest.items():
        if not (folder/name).resolve().is_relative_to(folder.resolve()) or sha(folder/name)!=value:raise ValueError('Archive manifest differs')
    verified=read(folder/'MODEL_FILE_VERIFICATION.json')
    if [r['name'] for r in verified]!=[r['name'] for r in pin['files']]:raise ValueError('Pinned file membership differs')
    for expected,actual in zip(pin['files'],verified):
        if expected['size']!=actual['size'] or (expected['sha256'] and expected['sha256']!=actual['sha256']):raise ValueError('Pinned file check differs')
        if not expected['name'].endswith('.safetensors') and sha(folder/'model_metadata'/expected['name'])!=actual['sha256']:raise ValueError('Metadata archive differs')
    tokenizer=AutoTokenizer.from_pretrained(folder/'model_metadata',local_files_only=True,trust_remote_code=False)
    tokens=read(folder/'TOKENS.json')
    if len(tokens)!=len(rows):raise ValueError('Token membership differs')
    for row,item in zip(rows,tokens):
        ids=encode(tokenizer,row['messages'])
        if item!={'state_id':row['state_id'],'input_token_ids':ids} or len(ids)>cfg['maximum_context_tokens']:raise ValueError('Native retokenization differs')
    receipt=read(folder/'RUN_RECEIPT.json')
    if receipt['status']!='complete' or receipt['states']!=112 or receipt['backbone_forwards']!=112 or receipt['backbone_forward_attempts']!=112:raise ValueError('Forward accounting differs')
    if receipt['base_fingerprint_before']!=receipt['base_fingerprint_after'] or receipt['base_fingerprint_before']!='c989470fab1c2fe4b62e590e2a6e173998943c353951bb7e919b8e8eca30d0fa':raise ValueError('NF4 base fingerprint differs')
    if any(receipt[k] for k in ['generated_actions','behavioral_interventions','backbone_optimizer_updates','research_adapters_loaded','original_confirmation_cases_opened']):raise ValueError('Research scope exceeded')
    with np.load(folder/'FEATURES.npz',allow_pickle=False) as data:features={k:data[k].copy() for k in ['primary','secondary']}
    if any(v.shape!=(112,5120) or v.dtype!=np.float32 or not np.isfinite(v).all() for v in features.values()):raise ValueError('Feature geometry differs')
    return cfg,rows,features


def probability(x,weight,bias):
    value=np.asarray(x@np.asarray(weight,dtype=np.float64)).reshape(-1)+float(bias)
    return 1/(1+np.exp(-np.clip(value,-700,700)))


def fit(folder,root,output):
    # This entry point is called only by the authenticated serverless notebook.
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.exceptions import ConvergenceWarning
    import sklearn,mlflow
    cfg,rows,features=load_inputs(folder,root);output=Path(output)
    indices={s:np.array([i for i,r in enumerate(rows) if r['split']==s]) for s in ['training','development','heldout']}
    train=indices['training'];dev=indices['development'];labels=np.array([r['label'] for r in rows])
    rc=cfg['readout'];params={k:rc[k] for k in ['C','class_weight','solver','max_iter','tol','random_state']}
    values={k:normalize(v) for k,v in features.items()}
    values['rms_only']=np.sqrt(np.mean(features['primary'].astype(np.float64)**2,axis=1,keepdims=True))
    texts=[json.dumps(r['messages'],sort_keys=True) for r in rows]
    vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=1,max_features=20000,lowercase=False)
    vectorizer.fit([texts[i] for i in train]);values['text']=vectorizer.transform(texts)
    # Shuffling stays within each training identity and preserves its label count.
    shuffled=labels.copy();rng=np.random.default_rng(rc['random_state'])
    for seed in cfg['training_seeds']:
        members=np.array([i for i in train if rows[i]['seed']==seed]);shuffled[members]=rng.permutation(labels[members])
    values['shuffled']=values['primary'];models={};coefficients={};results={};prediction_rows={}
    mlflow.set_registry_uri('databricks-uc')
    mlflow.set_experiment('/Workspace/Shared/research4-semantic-gate/experiments/native27-factual-recognition')
    with mlflow.start_run(run_name='fixed-readout-v1') as run:
        mlflow.log_params({**params,'primary_block_index':31,'threshold':rc['threshold'],'training_rows':70,'development_rows':21,'heldout_rows':21,'HPO_trials':0,'policy_updates':0})
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            for name in ['primary','secondary','rms_only','text','shuffled']:
                x=values[name];target=shuffled if name=='shuffled' else labels
                model=LogisticRegression(**params).fit(x[train],target[train]);models[name]=model
                coefficients[name]={'weight':model.coef_[0].tolist(),'bias':float(model.intercept_[0]),'iterations':model.n_iter_.tolist()}
                results[name]={};prediction_rows[name]={}
                for split in ['training','development']:
                    idx=indices[split];p=model.predict_proba(x[idx])[:,1]
                    results[name][split]=metrics([rows[i] for i in idx],p,rc['threshold'])
                    prediction_rows[name][split]=[{'state_id':rows[i]['state_id'],'probability':float(v)} for i,v in zip(idx,p)]
        passed=admitted(results['primary']['development'],cfg)
        for name,model in models.items():
            results[name]['heldout']=None;prediction_rows[name]['heldout']=None
            if passed:
                idx=indices['heldout'];p=model.predict_proba(values[name][idx])[:,1]
                results[name]['heldout']=metrics([rows[i] for i in idx],p,rc['threshold'])
                prediction_rows[name]['heldout']=[{'state_id':rows[i]['state_id'],'probability':float(v)} for i,v in zip(idx,p)]
        dump(output/'COEFFICIENTS.json',coefficients);dump(output/'PREDICTIONS.json',prediction_rows)
        dump(output/'TEXT_VECTORIZER.json',{'vocabulary':{k:int(v) for k,v in vectorizer.vocabulary_.items()},'idf':vectorizer.idf_.tolist(),'params':{'analyzer':'char','ngram_range':[3,5],'min_df':1,'max_features':20000,'lowercase':False}})
        dump(output/'FIT_CONFIG.json',{'config':cfg,'parameters':params,'training_state_ids':[rows[i]['state_id'] for i in train],'shuffled_training_labels':shuffled[train].tolist(),'sklearn':sklearn.__version__,'numpy':np.__version__})
        result={'status':'complete','development_admitted':passed,'results':results,'probe_heldout_scores_opened':passed,'original_confirmation_cases_opened':0,'new_backbone_forwards':0,'external_readout_fits':5,'backbone_optimizer_updates':0,'behavioral_interventions':0,'mlflow_run_id':run.info.run_id,'scope':cfg['interpretation']}
        dump(output/'RESULT.json',result)
        for name in results:mlflow.log_metric(name+'_development_balanced_accuracy',results[name]['development']['balanced_accuracy'])
        for name in ['COEFFICIENTS.json','PREDICTIONS.json','FIT_CONFIG.json','TEXT_VECTORIZER.json','RESULT.json']:mlflow.log_artifact(str(output/name))
    return result


def audit(folder,fit_folder,root,output):
    from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer
    cfg,rows,features=load_inputs(folder,root);fit_folder=Path(fit_folder);out=read(fit_folder/'RESULT.json')
    fc=read(fit_folder/'FIT_CONFIG.json');coef=read(fit_folder/'COEFFICIENTS.json');recorded=read(fit_folder/'PREDICTIONS.json')
    if fc['config']!=cfg or fc['training_state_ids']!=[r['state_id'] for r in rows if r['split']=='training']:raise ValueError('Training membership changed')
    if fc['parameters']!={k:cfg['readout'][k] for k in ['C','class_weight','solver','max_iter','tol','random_state']}:raise ValueError('Fitting settings changed')
    train=[i for i,r in enumerate(rows) if r['split']=='training'];labels=np.array([r['label'] for r in rows]);shuffled=labels.copy();rng=np.random.default_rng(cfg['readout']['random_state'])
    for seed in cfg['training_seeds']:
        members=np.array([i for i in train if rows[i]['seed']==seed]);shuffled[members]=rng.permutation(labels[members])
    if shuffled[train].tolist()!=fc['shuffled_training_labels']:raise ValueError('Shuffled control changed')
    vc=read(fit_folder/'TEXT_VECTORIZER.json');params=dict(vc['params']);params['ngram_range']=tuple(params['ngram_range'])
    vectorizer=TfidfVectorizer(**params,vocabulary=vc['vocabulary'])
    vectorizer._validate_vocabulary()
    vectorizer._tfidf=TfidfTransformer(norm='l2',use_idf=True,smooth_idf=True,sublinear_tf=False);vectorizer.idf_=np.asarray(vc['idf'])
    values={k:normalize(v) for k,v in features.items()};values['shuffled']=values['primary']
    values['rms_only']=np.sqrt(np.mean(features['primary'].astype(np.float64)**2,axis=1,keepdims=True))
    values['text']=vectorizer.transform([json.dumps(r['messages'],sort_keys=True) for r in rows])
    expected={}
    for name,x in values.items():
        expected[name]={}
        for split in ['training','development','heldout']:
            expected[name][split]=None
            if split=='heldout' and not out['development_admitted']:continue
            idx=np.array([i for i,r in enumerate(rows) if r['split']==split]);c=coef[name]
            p=probability(x[idx],c['weight'],c['bias']);r=recorded[name][split]
            if [v['state_id'] for v in r]!=[rows[i]['state_id'] for i in idx] or not np.allclose(p,[v['probability'] for v in r],rtol=1e-10,atol=1e-12):raise ValueError('Readout score reconstruction differs')
            expected[name][split]=metrics([rows[i] for i in idx],p,cfg['readout']['threshold'])
    passed=admitted(expected['primary']['development'],cfg)
    if expected!=out['results'] or passed!=out['development_admitted'] or passed!=out['probe_heldout_scores_opened']:raise ValueError('Admission or metrics differ')
    if not passed and any(recorded[name]['heldout'] is not None for name in recorded):raise ValueError('Unadmitted heldout scores opened')
    result={'audit_pass':True,'status':'complete','development_admitted':passed,'results':expected,'new_model_calls':0,'new_fits':0,'original_confirmation_cases_opened':0,'fit_result_sha256':sha(fit_folder/'RESULT.json'),'scope':'Separate history/token/feature-provenance replay and readout equation audit. Not independent backbone replication or evidence of fear or survival behavior.'}
    dump(output,result);return result
