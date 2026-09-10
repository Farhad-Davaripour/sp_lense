"""Unmocked metadata/paired-row bridge over an already artificial capture mirror."""
import ast,hashlib,json,sys,tempfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
    capture=Path(sys.argv[1]).resolve();assert (capture/'TEST_RESULT.json').is_file()
    m=json.loads((capture/'TEST_RESULT.json').read_bytes());assert m['actual_admission'] is False and m['model_calls']==m['fits']==0
    base=Path(tempfile.mkdtemp(prefix='fit_',dir=HERE.parent/'test_evidence'))
    source_hash=sha((capture/'SOURCE_FREEZE.json').read_bytes());deltas=[]
    for name in ('gate.py','checker.py','source_auth.py','construction.py','fit_owner.py'):
        raw=(HERE/name).read_bytes();text=raw.decode();before=ast.parse(text)
        text=text.replace('ROOT=HERE.parents[2]',f'ROOT=Path({str(ROOT)!r})')
        text=text.replace("CAPTURE=HERE.parent/'train_capture'",f'CAPTURE=Path({str(capture)!r})')
        if name=='source_auth.py':
            import re
            text=re.sub(r"CAPTURE_SOURCE_SHA='[0-9a-f]{64}'",f"CAPTURE_SOURCE_SHA='{source_hash}'",text)
        text=text.replace("TRAINING_SOURCE=HERE.parent/'train_capture/DATA_LOCK.json'",f'TRAINING_SOURCE=Path({str(capture / "DATA_LOCK.json")!r})')
        body=lambda tree:[ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        assert body(before)==body(ast.parse(text)),name
        if text!=raw.decode():deltas.append(name)
        (base/name).write_text(text,encoding='utf-8',newline='')
    sys.path.insert(0,str(base));import construction,source_auth,gate
    result=construction.candidate();assert result['maximum_optimizations']==3
    inputs=gate.decode((base/'TRAINING_MANIFEST.json').read_bytes());binding=inputs['capture_binding']
    bad={**binding,'capture_manifest_sha256':'0'*64}
    try:source_auth.load_saved(deadline=time.monotonic()+30,expected_binding=bad)
    except ValueError as e:assert str(e)=='EXACT_RELEASED_CAPTURE_BINDING_BEFORE_FEATURES'
    else:raise AssertionError('wrong binding accepted')
    rows,labels=source_auth.load_saved(deadline=time.monotonic()+30,expected_binding=binding)
    assert len(rows)==29 and all(len(r)==1024 for r in rows) and labels.count(1)==7
    assert construction.schedule()==[{'id':'PRECHOICE29','train':list(range(29)),'held':[]}]
    assert [len(construction.head_indices(list(range(29)),h)) for h in construction.HEADS]==[14,14,15]
    raw=(base/'RELEASE_DRAFT.json').read_bytes()
    try:construction.authorize(raw,gate.digest(raw))
    except ValueError as e:assert str(e)=='RELEASE_DEFAULT_DENY'
    else:raise AssertionError('inert release admitted')
    model=construction.Model([0.]*1024,[{'w':[0.]*1024,'b':1.} for _ in range(3)])
    artifact=construction.artifact(model,{'synthetic':True});construction.load_artifact(artifact,expected_sha256=sha(artifact),expected_bindings={'synthetic':True})
    bad=gate.decode(artifact);bad['feature_contract']['position']='final_input';wrong=gate.canonical(bad)
    try:construction.load_artifact(wrong,expected_sha256=sha(wrong),expected_bindings={'synthetic':True})
    except ValueError as e:assert str(e)=='ARTIFACT_BINDINGS'
    else:raise AssertionError('legacy site artifact admitted')
    receipt={'status':'PASS_SYNTHETIC_SAVED_PAIR_METADATA_BRIDGE','pairs':29,'views':58,'head_sizes':[14,14,15],
        'max_solves':3,'actual_features':False,'optimizer_calls':0,'function_AST_changed':False,'constant_substitutions':deltas,
        'capture_mirror':str(capture),'fit_mirror':str(base),'source_sha256':{n:sha((HERE/n).read_bytes()) for n in ('gate.py','checker.py','source_auth.py','construction.py','fit_owner.py')}}
    (base/'TEST_RESULT.json').write_text(json.dumps(receipt,sort_keys=True),encoding='utf-8');print(json.dumps(receipt,sort_keys=True))
if __name__=='__main__':main()
