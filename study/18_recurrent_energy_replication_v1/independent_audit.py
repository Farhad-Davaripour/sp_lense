"""Independent NumPy energy calculation and analytic driver Jacobians."""
import os,time
os.environ['OPENBLAS_NUM_THREADS']='2'
BEGIN=time.monotonic()
import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path('/content/research4/artifacts/recurrent_energy_replication_v1/colab_cpu_20261015')
P17=Path('/content/research4/artifacts/fresh_memory_boundary_response_v1/colab_cpu_20261014/fresh')
sys.path.insert(0,str(ROOT/'source'))
from output_store import OutputStore
def read(p):return json.loads(p.read_text())
def norm(x):return float(np.linalg.norm(np.asarray(x,dtype=np.float64)))
def pnorm(x):return np.sqrt(np.sum(np.asarray(x,dtype=np.float64)**2,axis=(-2,-1)))
def checksum(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def sigma(x):return 1/(1+np.exp(-np.asarray(x,dtype=np.float64)))
def fields(path,prefix,names):
    a=store._read(path,{prefix+':'+n for n in names});return {n:a[prefix+':'+n] for n in names}
store=OutputStore(P17);meta=read(ROOT/'PARAMETER_RECEIPT.json')['linear_layers'];layers=[m['layer'] for m in meta]
with np.load(ROOT/'FULL_DRIVERS.npz',allow_pickle=False) as z:drivers={n:z[n].copy() for n in z.files}
with np.load(ROOT/'EFFECTIVE_PARAMETERS.npz',allow_pickle=False) as z:weights={n:z[n].astype(np.float64) for n in z.files}
case_meta=read(ROOT/'DRIVER_CASES.json');heads=read(ROOT/'HEAD_RESULTS.json');native_rows=read(P17/'METRICS.json');assert len(case_meta)==92 and len(heads)==26496
lookup={(r['case'],r['layer'],r['head']):r for r in heads};state_names={f'm{i}_recurrent_states_0' for i in layers}
questions=read(P17/'INPUTS.json')['questions'];incoming={}
for q in questions:
    cache=torch.load(P17/'contexts'/f'{q["id"]}_incoming_cache.pt',map_location='cpu',weights_only=False)
    incoming[q['id']]={i:cache.layers[i].recurrent_states[0].detach().cpu().numpy().copy().reshape(16,128,128).astype(np.float64) for i in layers}
    del cache
audit={'new_backbone_forwards':0,'head_cases':0,'max_head_scalar_discrepancy':0.,'max_independent_stationarity':0.,'max_dynamic_force_residual':0.,'all_original_primary_flags_confirmed':True,'analytic_response_fields':0,'analytic_response_heads':0,'analytic_response_failures':[]}
previous=None;previous_key=None
for ci,case in enumerate(case_meta):
    pid,t=case['question'],case['token'];key=(pid,case['label'],case['alpha'])
    path=f'contexts/{pid}_t{t}_baseline.npz' if case['label']=='normal' else native_rows[int(case['case'].split('_')[-1])]['actual_file']
    actual=fields(path,'v',state_names)
    for li,i in enumerate(layers):
        S0=incoming[pid][i] if t==1 else previous[f'm{i}_recurrent_states_0'].reshape(16,128,128).astype(np.float64)
        S1=actual[f'm{i}_recurrent_states_0'].reshape(16,128,128).astype(np.float64)
        k=drivers['k'][ci,li].astype(np.float64);v=drivers['v'][ci,li].astype(np.float64)
        b=drivers['beta'][ci,li].astype(np.float64);g=drivers['gamma'][ci,li].astype(np.float64)
        kv0=np.einsum('hk,hkv->hv',k,S0);kv1=np.einsum('hk,hkv->hv',k,S1);delta=S1-S0;kd=np.einsum('hk,hkv->hv',k,delta)
        expected=g[:,None,None]*S0+b[:,None,None]*np.einsum('hk,hv->hkv',k,v-g[:,None]*kv0)
        stat=pnorm(S1-expected)/np.maximum(1.,pnorm(S1))
        f=b[:,None,None]*np.einsum('hk,hv->hkv',k,v)
        internal=(1-g)[:,None,None]*S0+(g*b)[:,None,None]*np.einsum('hk,hv->hkv',k,kv0)
        fr=pnorm(delta-f+internal)/np.maximum(1.,np.maximum(pnorm(f),pnorm(internal)))
        W=np.einsum('hkv,hkv->h',f,delta)
        U0=.5*(1-g)*np.sum(S0*S0,axis=(-2,-1))+.5*g*b*np.sum(kv0*kv0,axis=-1)
        U1=.5*(1-g)*np.sum(S1*S1,axis=(-2,-1))+.5*g*b*np.sum(kv1*kv1,axis=-1)
        D=np.sum(delta*delta,axis=(-2,-1))-.5*((1-g)*np.sum(delta*delta,axis=(-2,-1))+g*b*np.sum(kd*kd,axis=-1))
        L0=.5*np.sum((g[:,None]*kv0-v)**2,axis=-1);L1=.5*np.sum((kv1-v)**2,axis=-1)
        factor=(1-b*np.sum(k*k,axis=-1))**2
        for h in range(16):
            row=lookup[case['case'],i,h]
            values={'published_step_stationarity_relative':stat[h],'force_balance_relative':fr[h],'external_work':W[h],
                'internal_quadratic_energy_change':U1[h]-U0[h],'discrete_dissipation':D[h],
                'regression_loss_before':L0[h],'regression_loss_after':L1[h],'predicted_loss_after':factor[h]*L0[h]}
            for name,value in values.items():
                error=abs(float(value)-row[name])/max(1.,abs(float(value)));audit['max_head_scalar_discrepancy']=max(audit['max_head_scalar_discrepancy'],error)
                assert error<1e-9,(case['case'],i,h,name,error)
            assert stat[h]<=1e-5 and fr[h]<=1e-5
            assert all(row[k] for k in ['primal_pass','published_step_pass','force_and_virtual_pass','work_balance_pass','loss_factor_identity_pass','potential_nonincrease_pass','dynamics_stiffness_PSD_and_stable'])
            audit['head_cases']+=1
        audit['max_independent_stationarity']=max(audit['max_independent_stationarity'],float(stat.max()))
        audit['max_dynamic_force_residual']=max(audit['max_dynamic_force_residual'],float(fr.max()))
    if t==1:previous=actual;previous_key=key
    else:assert previous_key==key;previous=None;previous_key=None

def analytic_drivers(layer,h,window,dh,dwindow,baseline_driver):
    p=weights;prefix=f'l{layer:02d}_';eps=1e-6
    x=np.asarray(h,dtype=np.float64);dx=np.asarray(dh,dtype=np.float64);scale=1/np.sqrt(np.mean(x*x)+eps)
    z=x*scale*(1+p[prefix+'norm_weight'])
    dz=(dx*scale-x*scale**3*np.mean(x*dx))*(1+p[prefix+'norm_weight'])
    a=p[prefix+'in_proj_a_weight']@z;da=p[prefix+'in_proj_a_weight']@dz;db=p[prefix+'in_proj_b_weight']@dz
    beta=baseline_driver['beta'];gamma=baseline_driver['gamma']
    d_beta=beta*(1-beta)*db
    d_gamma=-gamma*np.exp(p[prefix+'A_log'])*sigma(a+p[prefix+'dt_bias'])*da
    kernel=p[prefix+'conv_weight'][:,0,:]
    pre=np.sum(np.asarray(window,dtype=np.float64)[0]*kernel,axis=-1)
    dpre=np.sum(np.asarray(dwindow,dtype=np.float64)[0]*kernel,axis=-1)
    if prefix+'conv_bias' in p:pre+=p[prefix+'conv_bias']
    sig=sigma(pre);mixed=pre*sig;dmixed=(sig+pre*sig*(1-sig))*dpre
    raw=mixed[2048:4096].reshape(16,128);draw=dmixed[2048:4096].reshape(16,128)
    inv=1/np.sqrt(np.sum(raw*raw,axis=-1,keepdims=True)+eps)
    dkey=draw*inv-raw*inv**3*np.sum(raw*draw,axis=-1,keepdims=True)
    return dkey,dmixed[4096:].reshape(16,128),d_beta,d_gamma

names={n for i in layers for n in [f'b{i}_input',f'm{i}_conv_states_0',f'm{i}_recurrent_states_0']}
normal_index={(c['question'],c['token']):j for j,c in enumerate(case_meta) if c['label']=='normal'}
panel=[(q['id'],'primary') for q in questions]+[(p,'reference') for p in ['r17_cal_spoon','r17_held_panels']]
derivative_rows=read(ROOT/'DERIVATIVE_RESULTS.json');derivative_lookup={(r['question'],r['label'],r['token'],r['layer']):r for r in derivative_rows}
for pid,label in panel:
    first_q=None
    for t in [1,2]:
        base=fields(f'contexts/{pid}_t{t}_baseline.npz','v',names);q=fields(f'coefficients/{pid}_{label}_t{t}.npz','q',names)
        old_values=None if t==1 else fields(f'contexts/{pid}_t1_baseline.npz','v',state_names)
        for li,i in enumerate(layers):
            S0=incoming[pid][i] if t==1 else old_values[f'm{i}_recurrent_states_0'].reshape(16,128,128).astype(np.float64)
            dS0=np.zeros_like(S0) if t==1 else first_q[f'm{i}_recurrent_states_0'].reshape(16,128,128).astype(np.float64)
            ni=normal_index[pid,t];d={name:drivers[name][ni,li].astype(np.float64) for name in ['k','v','beta','gamma']}
            dk,dv,db,dg=analytic_drivers(i,base[f'b{i}_input'],base[f'm{i}_conv_states_0'],q[f'b{i}_input'],q[f'm{i}_conv_states_0'],d)
            B=d['gamma'][:,None,None]*S0;dB=dg[:,None,None]*S0+d['gamma'][:,None,None]*dS0
            e=d['v']-np.einsum('hk,hkv->hv',d['k'],B)
            de=dv-np.einsum('hk,hkv->hv',dk,B)-np.einsum('hk,hkv->hv',d['k'],dB)
            predicted=dB+db[:,None,None]*np.einsum('hk,hv->hkv',d['k'],e)
            predicted+=d['beta'][:,None,None]*(np.einsum('hk,hv->hkv',dk,e)+np.einsum('hk,hv->hkv',d['k'],de))
            native=q[f'm{i}_recurrent_states_0'].reshape(16,128,128).astype(np.float64)
            assert checksum(q[f'm{i}_recurrent_states_0'])==derivative_lookup[pid,label,t,i]['native_q_sha256']
            error=pnorm(predicted-native);den=pnorm(native);passed=error<=np.maximum(.001*den,1e-6)
            if not np.all(passed):audit['analytic_response_failures'].append({'question':pid,'label':label,'token':t,'layer':i,'failed_heads':np.flatnonzero(~passed).tolist(),'maximum_relative':float(np.max(error/np.maximum(den,1e-30)))})
            audit['analytic_response_fields']+=1;audit['analytic_response_heads']+=16
        if t==1:first_q=q
assert audit['head_cases']==26496 and audit['analytic_response_heads']==4608
audit['all_analytic_response_heads_pass']=not audit['analytic_response_failures'];audit['seconds']=time.monotonic()-BEGIN
(ROOT.parent/'INDEPENDENT_ENERGY_AUDIT_20261015.json').write_text(json.dumps(audit,indent=2)+'\n')
print('R4_INDEPENDENT_ENERGY_AUDIT',json.dumps(audit),flush=True)
