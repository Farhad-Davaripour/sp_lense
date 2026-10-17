"""Posthoc exploratory spelling control; frozen direction, public bytes, no inference."""
import os,time
os.environ['OPENBLAS_NUM_THREADS']='2'
BEGIN=time.monotonic()
import sys,json,struct
from pathlib import Path
import numpy as np
A=Path(sys.argv[1]);R=A/'run_20261015';sys.path.insert(0,str(R/'source'))
from common import read,dump,load,norm
O=A/'output_spelling_check_20261015';O.mkdir(exist_ok=False)
eligible=read(A/'reference_20261015/TOKENIZER_ONLY_ADMISSION.json')['eligible'][16:]
spec=[]
for q in eligible:
    s,p=q['words'];group='simple_s' if p==s+'s' else 'y_to_ies' if s.endswith('y') and p==s[:-1]+'ies' else 'irregular'
    spec.append({'words':q['words'],'forms':q['forms'],'ids':[a[0] for a in q['ids']],'group':group})
assert len(spec)==17
dump(O/'EXPLORATORY_SPEC_BEFORE_WORD_VECTOR_ACCESS.json',{'posthoc':True,'new_backbone_calls':0,'frozen_direction_not_refitted':True,'original_semantic_gates_unchanged':True,'selection':'All17 remaining tokenizer-eligible official word pairs, original order; no selection on geometric results.','pairs':spec})
g=load(R/'OUTPUT_GEOMETRY.npz');M=g['inverse_covariance'];mean=g['mean_number_difference'];directions={'covariance_mapped':g['mapped_unit_direction'],'Euclidean':g['Euclidean_unit_direction']}
path=Path(sys.argv[2])/'model.safetensors-00001-of-00001.safetensors'
with path.open('rb') as f:
    n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n))
d=h['model.language_model.embed_tokens.weight'];assert d['dtype']=='BF16' and d['shape']==[248320,1024]
packed=np.memmap(path,mode='r',dtype='<u2',offset=8+n+d['data_offsets'][0],shape=(248320,1024))
rows=[]
for pair in spec:
    U=(np.asarray(packed[pair['ids']],dtype=np.uint32)<<16).view(np.float32).astype(np.float64);diff=U[1]-U[0]
    for name,v in directions.items():
        a=U@v;number=float((a[1]-a[0]+a[3]-a[2])/2);case=float(max(abs(a[2]-a[0]),abs(a[3]-a[1])))
        rows.append({**pair,'method':name,'lower_number_slope':float(a[1]-a[0]),'upper_number_slope':float(a[3]-a[2]),'both_number_slopes_positive':bool(a[1]>a[0] and a[3]>a[2]),'mean_number_slope':number,'maximum_case_slope':case,'case_to_number_ratio':case/max(1e-30,number),'metric_cosine':float(mean@M@diff/np.sqrt((mean@M@mean)*(diff@M@diff)))})
groups=[]
for name in directions:
    for group in ['simple_s','y_to_ies','irregular']:
        rr=[r for r in rows if r['method']==name and r['group']==group]
        groups.append({'method':name,'group':group,'pairs':len(rr),'both_number_slopes_positive':sum(r['both_number_slopes_positive'] for r in rr),'median_number_slope':float(np.median([r['mean_number_slope'] for r in rr])),'median_case_to_number_ratio':float(np.median([r['case_to_number_ratio'] for r in rr]))})
result={'posthoc_exploratory_only':True,'all17_remaining_pairs_reported':True,'no_refitting':True,'original_five_semantic_gate_outcomes_unchanged':True,'new_backbone_calls':0,'GPU_allocations':0,'groups':groups,'pair_details':rows,'seconds':time.monotonic()-BEGIN,'scope':'Tests an additional spelling-pattern confound in public output-vector geometry. Does not validate a block12 direction, task following, or the full semantic-transport hypothesis.'}
dump(O/'EXPLORATORY_RESULTS.json',result);print(json.dumps({k:v for k,v in result.items() if k!='pair_details'},indent=2))
