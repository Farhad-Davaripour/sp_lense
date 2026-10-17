"""Necessary local symmetry check for a Euclidean gradient-potential update."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('C:/Users/farha/Research4/artifacts/finite_logit_response_v1/native_cpu_20261014')
OUT=Path('C:/Users/farha/Research4/artifacts/response_geometry_v1/potential_symmetry_20261014')
OUT.mkdir(exist_ok=False)
MANIFEST=json.loads((ROOT/'EXPORT_HASHES.json').read_text());INPUTS=json.loads((ROOT/'INPUTS.json').read_text());verified={}
def dump(n,v):(OUT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def load(p):
    h=hashlib.sha256(p.read_bytes()).hexdigest();name=p.relative_to(ROOT).as_posix();assert h==MANIFEST[name];verified[name]=h
    with np.load(p,allow_pickle=False) as z:return {n:z[n].astype(np.float64) for n in z.files}
def norm(x):return float(np.linalg.norm(x))
plan={'classification':'Secondary necessary-condition test using verified saved derivatives; not new native confirmation.',
 'model_calls':0,'hypothesis':'A measured update f(x), with prefix fixed, equals -gradient E(x) in the original Euclidean residual coordinates near this point.',
 'necessary_condition':'For smooth E, J_f is symmetric, hence u^T J_f v = v^T J_f u.',
 'test':'Compute skew on two stored directional derivatives. Compare with finite-difference slopes at both pre-existing probe steps.',
 'bound':'For fixed u,v, numerical skew uncertainty <= ||u||*||Jv_exact-Jv_FD|| + ||v||*||Ju_exact-Ju_FD||.',
 'submaps':{'attention_plus_norm':'input -> middle minus input','whole_block_update':'input -> output minus input',
            'MLP_plus_norm':'middle -> output minus middle, using transported middle tangent directions'},
 'rule':'Nonzero skew with stable sign at both steps and positive margin beyond both discrepancy bounds rejects this simple local form.',
 'limits':['Failure of Euclidean gradient form does not exclude state-dependent metrics, rotational/port-Hamiltonian terms, Lyapunov functions or other energy descriptions.',
           'Only block11, prompt-end, fixed prefix, sampled contexts and two directional tangents.',
           'Directionwise numerical comparisons are evidence, not rigorous interval arithmetic error bounds.',
           'MLP slopes include its input RMSNorm; attention slopes include input RMSNorm and fixed-prefix attention. No raw matrix or bare activation claim.',
           'These outcomes were not reserved specifically for this hypothesis; fresh native confirmation is needed for a prospective study.']}
dump('ANALYSIS_FREEZE.json',{**plan,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
rows=[]
for question in INPUTS['questions']:
    pid=question['id'];c=[load(ROOT/'coefficients'/f'{pid}_d{di}.npz') for di in range(2)]
    variants={name:[] for name in plan['submaps']}
    for name,start,end in [('attention_plus_norm','input','middle'),('whole_block_update','input','output'),('MLP_plus_norm','middle','output')]:
        u,v=[x[start+'_first'] for x in c]
        Ju,Jv=[x[end+'_first']-x[start+'_first'] for x in c]
        k01=float(u@Jv);k10=float(v@Ju);skew=k01-k10
        fdrows=[]
        for step in [.015625,.0078125]:
            fds=[];input_fds=[]
            for di in range(2):
                plus=load(ROOT/'probes'/f'{pid}_d{di}_s{step}_plus.npz');minus=load(ROOT/'probes'/f'{pid}_d{di}_s{step}_minus.npz')
                # Compare update slopes while retaining fixed exact tangent test vectors.
                derivative=((plus[end+'_primal']-minus[end+'_primal'])-(plus[start+'_primal']-minus[start+'_primal']))/(2*step)
                fds.append(derivative)
                input_fds.append((plus[start+'_primal']-minus[start+'_primal'])/(2*step))
            fdskew=float(u@fds[1]-v@fds[0]);budget=norm(u)*norm(Jv-fds[1])+norm(v)*norm(Ju-fds[0])
            fdrows.append({'step':step,'skew':fdskew,'exact_vs_FD_discrepancy_bound':budget,
                           'input_tangent_relative_errors':[norm(input_fds[di]-c[di][start+'_first'])/max(norm(c[di][start+'_first']),1e-30) for di in range(2)]})
        uncertainty=max(r['exact_vs_FD_discrepancy_bound'] for r in fdrows)
        stable=all(r['skew']*skew>0 for r in fdrows)
        row={'question':pid,'split':question['split'],'submap':name,'u_dot_Jv':k01,'v_dot_Ju':k10,'skew':skew,
             'scale_normalized_skew':abs(skew)/max(norm(u)*norm(Jv)+norm(v)*norm(Ju),1e-30),
             'finite_difference_checks':fdrows,'skew_margin_beyond_numerical_discrepancy':abs(skew)-uncertainty,
             'margin_ratio':abs(skew)/max(uncertainty,1e-30),'stable_sign':stable,
             'simple_Euclidean_gradient_form_rejected':stable and abs(skew)>uncertainty}
        rows.append(row)
dump('ALL_CHECKS.json',rows);dump('SOURCE_IMMUTABILITY.json',verified)
summary={'model_calls':0,'secondary_not_new_confirmation':True,'smooth_potential_necessary_condition':'u^T Jf v = v^T Jf u',
 'submaps':{},'limits':plan['limits']}
for name in plan['submaps']:
    panel=[r for r in rows if r['submap']==name];held=[r for r in panel if r['split']=='held_out']
    summary['submaps'][name]={'all_contexts':len(panel),'all_rejected':sum(r['simple_Euclidean_gradient_form_rejected'] for r in panel),
      'original_held_contexts':len(held),'held_rejected':sum(r['simple_Euclidean_gradient_form_rejected'] for r in held),
      'held_normalized_skew_median':float(np.median([r['scale_normalized_skew'] for r in held])),
      'held_numerical_margin_ratio_minimum':min(r['margin_ratio'] for r in held)}
dump('RESULTS.json',summary)
lines=['SIMPLE POTENTIAL-ENERGY UPDATE — NECESSARY SYMMETRY CHECK','',
 'If a block update were minus the gradient of an ordinary scalar potential in the original residual coordinates,',
 'its derivative would have to be symmetric. Two direction probes are sufficient to find a violation.',
 'We tested this using the exact saved study13 derivatives and two independent finite-difference step sizes.','',
 json.dumps(summary,indent=2),'',
 'Interpretation: this tests one restrictive potential model, not every possible notion of energy.',
 'A useful local Fisher response cost can exist even when the actual update is not its gradient.',
 'No conservation/equilibrium law follows from defining a symmetric cost matrix.',
 'Original arrays remain unchanged; no model calls, GPU allocation or Colab use.']
(OUT/'FINDINGS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary))
