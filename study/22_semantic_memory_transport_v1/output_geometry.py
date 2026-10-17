"""Adapt the paper's output-space covariance metric; do not apply it to hidden memory."""
import time,hashlib
import numpy as np
from common import dump,save,norm
def build(weight,plan,root,report,deadline):
    start=time.monotonic();U=weight.detach().cpu().numpy();assert U.dtype==np.float32 and U.shape==(248320,1024)
    total=np.zeros(1024,dtype=np.float64);second=np.zeros((1024,1024),dtype=np.float64);fingerprint=hashlib.sha256()
    for first in range(0,len(U),8192):
        deadline();block=U[first:first+8192];fingerprint.update(np.ascontiguousarray(block).tobytes());x=block.astype(np.float64)
        total+=x.sum(0);second+=x.T@x
        if first%65536==0:report('public_output_covariance',rows_processed=min(first+8192,len(U)),native_calls_during_metric=0)
    mean=total/len(U);C=second/len(U)-np.outer(mean,mean);C=(C+C.T)/2
    eigen=np.linalg.eigvalsh(C);assert eigen[0]>0 and eigen[-1]/eigen[0]<1e8,'Unregularized metric admission failed; no adaptive ridge'
    M=np.linalg.solve(C,np.eye(1024));M=(M+M.T)/2
    pairs=plan['word_pairs'];selected=np.stack([U[p['ids']] for p in pairs])
    diff=selected[:,1].astype(np.float64)-selected[:,0].astype(np.float64)
    gamma=diff[:12].mean(0)
    caps=np.concatenate([selected[:12,2].astype(np.float64)-selected[:12,0],selected[:12,3].astype(np.float64)-selected[:12,1]]).mean(0)
    causal=M@gamma;causal/=norm(causal);euclidean=gamma/norm(gamma)
    rng=np.random.default_rng(2026101523);random_ids=rng.integers(0,len(U),size=(1024,2))
    random_diff=U[random_ids[:,1]].astype(np.float64)-U[random_ids[:,0]].astype(np.float64)
    direction=M@gamma/np.sqrt(gamma@M@gamma);control_direction=gamma/norm(gamma)
    random_projection=random_diff@direction;random_E=random_diff@control_direction
    held=[];slopes=[]
    for pair in pairs[12:16]:
        j=pair['pair_index'];d=diff[j];score=float(d@direction);e=float(d@control_direction)
        held.append({'pair_index':j,'forms':pair['forms'],'covariance_metric_projection':score,'metric_random_percentile':float(np.mean(random_projection<=score)),'Euclidean_projection':e,'Euclidean_random_percentile':float(np.mean(random_E<=e)),'metric_cosine':float(score/np.sqrt(d@M@d)),'Euclidean_cosine':float(e/norm(d))})
        for name,v in [('covariance_mapped',causal),('Euclidean',euclidean)]:
            words=selected[j].astype(np.float64)@v
            slopes.append({'pair_index':j,'method':name,'number_lower':float(words[1]-words[0]),'number_upper':float(words[3]-words[2]),'case_singular':float(words[2]-words[0]),'case_plural':float(words[3]-words[1])})
    selectivity=[]
    for name in ['covariance_mapped','Euclidean']:
        rr=[r for r in slopes if r['method']==name];number=float(np.median([(r['number_lower']+r['number_upper'])/2 for r in rr]));interference=float(np.median([max(abs(r['case_singular']),abs(r['case_plural'])) for r in rr]))
        selectivity.append({'method':name,'all_held_number_slopes_positive':all(r['number_lower']>0 and r['number_upper']>0 for r in rr),'median_number_slope':number,'median_case_slope':interference,'off_target_ratio':interference/max(1e-30,number),'pass':all(r['number_lower']>0 and r['number_upper']>0 for r in rr) and interference<=.5*number})
    subspace=all(r['covariance_metric_projection']>0 for r in held) and float(np.median([r['metric_random_percentile'] for r in held]))>=.95
    cosine=lambda a,b,Q:float(a@Q@b/np.sqrt((a@Q@a)*(b@Q@b)))
    result={'adapted_paper_replication':True,'new_backbone_forwards':0,'all_vocabulary_rows':len(U),'dimension':1024,'inverse_covariance_no_ridge':True,'covariance_eigenvalue_min':float(eigen[0]),'covariance_eigenvalue_max':float(eigen[-1]),'condition_number':float(eigen[-1]/eigen[0]),'output_weight_FP32_sha256':fingerprint.hexdigest(),'held_pair_subspace_gate_pass':bool(subspace),'held_pair_projections':held,'held_output_intervention_slopes':slopes,'output_selectivity':selectivity,'number_case_cosine_under_M':cosine(gamma,caps,M),'number_case_cosine_Euclidean':cosine(gamma,caps,np.eye(1024)),'seconds':time.monotonic()-start,'scope':'Public output-weight geometry and analytical interventions after final normalization. Semantic specificity is tested on held noun forms. No assumption of the same metric in block12 or recurrent memory.'}
    save(root/'OUTPUT_GEOMETRY.npz',{'covariance':C,'inverse_covariance':M,'vocabulary_mean':mean,'selected_word_weights':selected,'mean_number_difference':gamma,'mean_capitalization_difference':caps,'mapped_unit_direction':causal,'Euclidean_unit_direction':euclidean,'random_pair_ids':random_ids,'random_pair_differences':random_diff})
    dump(root/'OUTPUT_GEOMETRY_RESULTS.json',result)
    return result
