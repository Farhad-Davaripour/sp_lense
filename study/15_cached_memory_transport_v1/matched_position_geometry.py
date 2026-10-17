"""Compare identical steering coordinates at two recorded token positions.

Saved-data analysis after outcomes, not a new prospective test. Fisher energies
refer to different normal output distributions; there is no conservation claim.
"""
import hashlib,json,zipfile,io
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path('C:/Users/farha/Research4/artifacts/cached_memory_transport_v1')
P14=Path('C:/Users/farha/Research4/artifacts/cached_state_response_v1/R4_CACHED_STATE_NATIVE_20261014_COLAB_CPU_RESUME1.zip')
P15=BASE/'R4_MEMORY_TRANSPORT_NATIVE_20261014.zip'
OUT=BASE/'matched_position_geometry_20261014_v2'

def arrays(z,name,manifest):
    data=z.read(name);assert hashlib.sha256(data).hexdigest()==manifest[name]
    with np.load(io.BytesIO(data),allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def matrix(logits,columns):
    l=logits.astype(np.float64);p=np.exp(l-l.max());p/=p.sum()
    q=np.stack(columns,axis=1).astype(np.float64);q-=p@q
    G=q.T@(p[:,None]*q);assert np.allclose(G,G.T,rtol=0,atol=1e-12)
    eig=np.linalg.eigvalsh(G);assert eig[0]>0
    return G,eig

def main():
    OUT.mkdir(exist_ok=False);rows=[]
    with zipfile.ZipFile(P14) as z1,zipfile.ZipFile(P15) as z2:
        m1=json.loads(z1.read('EXPORT_HASHES.json'));m2=json.loads(z2.read('EXPORT_HASHES.json'))
        questions=json.loads(z1.read('INPUTS.json'))['questions']
        for q in questions:
            pid=q['id'];baselines=[arrays(z,f'contexts/{pid}_baseline.npz',m) for z,m in [(z1,m1),(z2,m2)]]
            jets=[[arrays(z,f'coefficients/{pid}_d{i}.npz',m) for i in range(2)] for z,m in [(z1,m1),(z2,m2)]]
            G1,e1=matrix(baselines[0]['logits'],[x['q'] for x in jets[0]])
            G2,e2=matrix(baselines[1]['logits'],[x['q'] for x in jets[1]])
            d,U=np.linalg.eigh(G1);invsqrt=(U/np.sqrt(d))@U.T
            gain=invsqrt@G2@invsqrt;gain=(gain+gain.T)/2;values,axes=np.linalg.eigh(gain)
            assert values[0]>0
            directions=invsqrt@axes
            for i in range(2):
                v=directions[:,i];assert np.isclose(v@G2@v/(v@G1@v),values[i],rtol=1e-10)
            metadata=json.loads(z2.read(f'contexts/{pid}.json'))
            rows.append({'question':pid,'family':q['family'],'split':q['split'],'Fisher_step1':G1.tolist(),'Fisher_step2':G2.tolist(),
                'step1_eigenvalues':e1.tolist(),'step2_eigenvalues':e2.tolist(),
                'step1_anisotropy':float(e1[1]/e1[0]),'step2_anisotropy':float(e2[1]/e2[0]),
                'same_plane_cost_gain_extrema':values.tolist(),'same_plane_gain_direction_coordinates':directions.tolist(),
                'basis_cost_gains':(np.diag(G2)/np.diag(G1)).tolist(),'forced_second_EOS_continuation':metadata['EOS_fallback']})
    held=[r for r in rows if r['split']=='held_out'];assert len(held)==4
    result={'classification':'post-outcome saved-array analysis; no native calls or new questions',
        'model_calls':0,'held_unique_known_questions':4,'held_forced_second_token_cases':sum(r['forced_second_EOS_continuation'] for r in held),
        'directions':'same two parent14 frozen Gaussian coordinates at both positions',
        'energy_definition':'KL(normal output || changed output) = alpha^2 v^T G v /2 + O(alpha^3)',
        'gain_definition':'Extrema of v^T G_step2 v / (v^T G_step1 v) in the same sampled two-direction plane',
        'scope':'Different normal tokens/output distributions; gain is conditional output sensitivity, not conserved physical work. Local quadratic coefficient, not finite-strength/free-generation accuracy or full-space spectrum. Original phase15 control failures remain failed.',
        'summary':{'step1_anisotropy_median':float(np.median([r['step1_anisotropy'] for r in held])),
                   'step2_anisotropy_median':float(np.median([r['step2_anisotropy'] for r in held])),
                   'same_plane_min_gain_range':[float(min(r['same_plane_cost_gain_extrema'][0] for r in held)),float(max(r['same_plane_cost_gain_extrema'][0] for r in held))],
                   'same_plane_max_gain_range':[float(min(r['same_plane_cost_gain_extrema'][1] for r in held)),float(max(r['same_plane_cost_gain_extrema'][1] for r in held))]},'rows':rows}
    (OUT/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,ax=plt.subplots(1,2,figsize=(10,4.4));x=np.arange(4)
    ax[0].bar(x-.18,[r['step1_anisotropy'] for r in held],.36,label='First token')
    ax[0].bar(x+.18,[r['step2_anisotropy'] for r in held],.36,label='Following token')
    ax[0].set_ylabel('Largest / smallest sensitivity in sampled plane');ax[0].legend(frameon=False)
    for i,r in enumerate(held):
        lo,hi=r['same_plane_cost_gain_extrema'];ax[1].plot([i,i],[lo,hi],color='#4574a2',linewidth=3)
        ax[1].scatter([i,i],[lo,hi],color='#4574a2',s=36)
    ax[1].axhline(1,color='gray',linestyle='--',linewidth=1);ax[1].set_yscale('log')
    ax[1].set_ylabel('Following / first-token sensitivity (range across plane)')
    for a in ax:
        a.set_xticks(x,[r['family'].replace('_','\n') for r in held]);a.spines[['top','right']].set_visible(False)
    fig.suptitle('Same nudge coordinates, different token positions',fontsize=14)
    fig.text(.5,.01,'Four previously tested questions; 3 following tokens are forced EOS replacements.\nSaved-data analysis; local output KL, not physical energy.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.94]);fig.savefig(OUT/'MATCHED_SENSITIVITY.png',dpi=160);plt.close(fig)
    print(json.dumps(result['summary']))

if __name__=='__main__':main()
