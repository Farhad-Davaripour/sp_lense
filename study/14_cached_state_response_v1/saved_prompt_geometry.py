"""Secondary geometry of already measured prompt-end response; no native calls."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('C:/Users/farha/Research4/artifacts/finite_logit_response_v1/native_cpu_20261014')
OUT=Path('C:/Users/farha/Research4/artifacts/response_geometry_v1/evaluated_20261014')
OUT.mkdir(parents=True,exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {n:z[n].astype(np.float64) for n in z.files}
def dump(n,v):(OUT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def metric(q,p=None):
    if p is None:return q@q.T
    qc=q-(q@p)[:,None];return (qc*p)@qc.T
def spectrum(G):
    eigenvalues,eigenvectors=np.linalg.eigh(G)
    return {'matrix':G.tolist(),'eigenvalues':eigenvalues.tolist(),
       'eigenvalue_ratio':float(eigenvalues[-1]/eigenvalues[0]) if eigenvalues[0]>1e-15 else None,
       'direction_cosine':float(G[0,1]/np.sqrt(G[0,0]*G[1,1])) if G[0,0]*G[1,1]>0 else None,
       'largest_eigenvector_coordinates':eigenvectors[:,-1].tolist()}
def stats(x):return {'median':float(np.median(x)),'minimum':float(min(x)),'maximum':float(max(x))}
def inner(p,a,b):return float(p@((a-p@a)*(b-p@b)))
inputs=json.loads((ROOT/'INPUTS.json').read_text());manifest=json.loads((ROOT/'EXPORT_HASHES.json').read_text())
paths=[ROOT/'INPUTS.json']
for q in inputs['questions']:
    pid=q['id'];paths.append(ROOT/'baseline'/f'{pid}.npz')
    paths.extend(ROOT/'coefficients'/f'{pid}_d{di}.npz' for di in range(2))
    paths.extend(ROOT/'actual'/f'{pid}_d{di}.npz' for di in range(2))
assert all(sha(p)==manifest[p.relative_to(ROOT).as_posix()] for p in paths)
plan={'classification':'Secondary analysis of completed study13; outcomes already known; not fresh confirmation',
 'model_calls':0,'source_manifest':str(ROOT/'EXPORT_HASHES.json'),'source_sha256':{str(p):sha(p) for p in paths},
 'analysis_source_sha256':sha(Path(__file__)),
 'questions':8,'held_questions_in_original_test':6,'same_two_known_directions':True,
 'metrics':['2D J-transpose-Fisher-J at frozen prompt point','ordinary norm response Gram at internal boundaries',
            'actual versus predicted two-axis secant Gram','sign-mirror response cosine'],
 'limits':['Only sampled two-dimensional plane, not a full-space eigenspectrum.',
           'No mixed-direction measurements or network mixed second derivatives in these arrays.',
           'Same axes evaluated across questions; no universal state-independent matrix.',
           'Geometry does not assign semantic labels or prove physical energy conservation.']}
dump('ANALYSIS_FREEZE.json',plan)
rows=[];secants=[];mirrors=[]
for question in inputs['questions']:
    pid=question['id'];base=load(ROOT/'baseline'/f'{pid}.npz')
    p=np.exp(base['logits']-base['logits'].max());p/=p.sum()
    c=[load(ROOT/'coefficients'/f'{pid}_d{di}.npz') for di in range(2)]
    actual=[load(ROOT/'actual'/f'{pid}_d{di}.npz') for di in range(2)]
    qs=np.stack([v['q'] for v in c]);G=metric(qs,p)
    row={'question':pid,'split':question['split'],'family':question['family'],'output_Fisher_plane':spectrum(G),'internal_planes':{}}
    for name in ['input','middle','output','hidden']:
        J=np.stack([v[name+'_first'] for v in c]);row['internal_planes'][name]=spectrum(metric(J))
    rows.append(row)
    for strength in [.03125,.125,.5]:
        for sign in [1,-1]:
            alpha=sign*strength;idx=[list(v['alphas']).index(alpha) for v in actual]
            truth=np.stack([actual[di]['logits'][idx[di]]-base['logits'] for di in range(2)])
            measured=metric(truth/alpha,p)
            for order in [1,2]:
                predicted=np.stack([alpha*c[di]['q']+(.5*alpha**2*c[di]['r'] if order==2 else 0.) for di in range(2)])
                predG=metric(predicted/alpha,p)
                secants.append({'question':pid,'split':question['split'],'strength':strength,'sign':sign,'order':order,
                    'actual_two_axis_secant_metric':spectrum(measured),'predicted_two_axis_secant_metric':spectrum(predG),
                    'metric_relative_error':float(np.linalg.norm(predG-measured)/np.linalg.norm(measured)),
                    'not_a_measured_mixed_direction_endpoint':True})
        for di in range(2):
            ip=list(actual[di]['alphas']).index(strength);im=list(actual[di]['alphas']).index(-strength)
            a=actual[di]['logits'][ip]-base['logits'];b=base['logits']-actual[di]['logits'][im]
            mirror=inner(p,a,b)/np.sqrt(inner(p,a,a)*inner(p,b,b))
            ap=strength*c[di]['q']+.5*strength**2*c[di]['r'];bm=strength*c[di]['q']-.5*strength**2*c[di]['r']
            predicted=inner(p,ap,bm)/np.sqrt(inner(p,ap,ap)*inner(p,bm,bm))
            mirrors.append({'question':pid,'split':question['split'],'direction':di,'strength':strength,
                            'actual_sign_mirror_cosine':mirror,'second_order_mirror_cosine':predicted,
                            'first_order_mirror_cosine':1.,'second_absolute_error':abs(predicted-mirror)})
dump('CONTEXT_GEOMETRY.json',rows);dump('SECANT_METRICS.json',secants);dump('SIGN_MIRROR.json',mirrors)
held=[r for r in rows if r['split']=='held_out'];summary={'model_calls':0,'secondary_not_new_confirmation':True,
 'held_plane_eigenvalue_ratio':stats([r['output_Fisher_plane']['eigenvalue_ratio'] for r in held]),
 'held_weighted_axis_cosine':stats([r['output_Fisher_plane']['direction_cosine'] for r in held]),
 'held_geometry_by_strength':{},'source_files_hash_verified':len(paths),'limits':plan['limits']}
for strength in [.03125,.125,.5]:
    summary['held_geometry_by_strength'][str(strength)]={
        'secant_metric_relative_error':{str(order):stats([r['metric_relative_error'] for r in secants if r['split']=='held_out' and r['strength']==strength and r['order']==order]) for order in [1,2]},
        'actual_sign_mirror_cosine':stats([r['actual_sign_mirror_cosine'] for r in mirrors if r['split']=='held_out' and r['strength']==strength]),
        'second_order_mirror_absolute_error':stats([r['second_absolute_error'] for r in mirrors if r['split']=='held_out' and r['strength']==strength])}
dump('RESULTS.json',summary)
lines=['LOCAL RESPONSE GEOMETRY — SECONDARY SAVED-DATA ANALYSIS','',
 'We can express a local output-change cost for a push inside the sampled two-direction plane:',
 'E(c) = 1/2 c-transpose G c, where G = Q-transpose [diag(p)-p p-transpose] Q.',
 'Q contains the two measured full-logit derivatives; p is the normal output distribution.',
 'This equation defines local sensitivity, not a physical energy conserved by the transformer.',
 'The eigenvectors select more/less sensitive combinations within this plane only.',
 'The original transformer has not been replaced by G; G depends on the normal context and intervention site.','',
 f'Held-context plane sensitivity ratios: {summary["held_plane_eigenvalue_ratio"]}.',
 f'Held weighted axis cosines: {summary["held_weighted_axis_cosine"]}.','']
for s,v in summary['held_geometry_by_strength'].items():lines.append(f'Strength {s}: {v}')
lines+=['','This uses existing outcomes and is secondary, not a new holdout test.',
 'Two axes permit a local first-order metric. Their separate curvatures cannot determine the network mixed second derivative;',
 'a third directional probe is needed for the two-dimensional quadratic state equation.',
 'Next native test remains cached decoding, complete memory, and the predeclared mixed-direction probe.',
 'All source arrays hash verified unchanged; zero model calls, GPU allocations or Colab use.']
(OUT/'FINDINGS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary))
