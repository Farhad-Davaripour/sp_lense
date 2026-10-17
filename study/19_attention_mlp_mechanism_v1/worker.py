"""Bounded CPU native-trace replay; no whole-backbone forward or fitting."""
import os,time
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2')
BEGIN=time.monotonic()
import sys,json,hashlib,gc,zipfile,traceback,shutil
from pathlib import Path
import numpy as np,torch,transformers
ROOT=Path('/content/research4/artifacts/attention_mlp_mechanism_v1/colab_cpu_20261015')
SOURCE=ROOT/'source'
P17=Path('/content/research4/artifacts/fresh_memory_boundary_response_v1/colab_cpu_20261014')
FRESH=P17/'fresh'
P18=Path('/content/research4/artifacts/recurrent_energy_replication_v1/colab_cpu_20261015')
sys.path[:0]=[str(SOURCE),str(P17/'fresh_source')]
from equations import norm,mlp,attention
from output_store import OutputStore

def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for a in iter(lambda:f.read(8*1024*1024),b''):h.update(a)
    return h.hexdigest()
def nrm(x):return float(np.linalg.norm(np.asarray(x,dtype=np.float64)))
def deadline():
    if time.monotonic()-BEGIN>2100:raise TimeoutError('Phase19 worker budget')
def report(phase,**kw):
    STATUS.update(phase=phase,seconds=time.monotonic()-BEGIN,**kw);dump(ROOT/'STATUS.json',STATUS)
    print('R4_ATTENTION_MLP',json.dumps(STATUS),flush=True)
STATUS={'status':'running','PID':os.getpid(),'new_backbone_forwards':0,'GPU_allocations':0,'primal_cases':0,'derivative_cases':0}
rows=[];energy_rows=[];used={};arrays={};derivative_rows=[]
store=OutputStore(FRESH)
expected=read(P18/'PARENT_USED_HASHES.json')
def tracked(path):
    key=path.relative_to(FRESH).as_posix()
    if key not in used:
        got=sha(path)
        if '17:'+key in expected:assert got==expected['17:'+key],key
        used[key]=got
    return path
def fields(name,prefix,names):
    tracked(FRESH/name)
    a=store._read(name,{prefix+':'+n for n in names})
    return {n:a[prefix+':'+n] for n in names}
def compare(pred,actual,derivative=False):
    error=nrm(pred-actual);den=nrm(actual)
    bound=max(.001*den,1e-6) if derivative else 1e-5*max(1.,den)
    return {'error_norm':error,'native_norm':den,'relative':error/max(den,1e-30),'bound':bound,'pass':bool(error<=bound)}

def main():
    from base_native import Rig
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    assert (torch.__version__,transformers.__version__,np.__version__)==('2.13.0+cpu','5.19.0','2.5.4') and not torch.cuda.is_available()
    free=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    dump(ROOT/'ADMISSION.json',{'free_memory_after_imports':free,'minimum':8589934592,'disk_free':shutil.disk_usage(ROOT).free})
    assert free>=8589934592 and shutil.disk_usage(ROOT).free>=17179869184
    freeze=read(ROOT/'EXECUTION_FREEZE.json')
    for n,h in freeze['sha256'].items():assert sha(ROOT/n)==h,n
    pin=read(P17/'fresh_source/model_pin.json');checkpoint=Path('/content/r4_checkpoint');verified=[]
    for item in pin['files']:
        p=checkpoint/item['name']
        if not p.is_file():continue
        assert p.stat().st_size==item['size']
        if item.get('sha256'):assert sha(p)==item['sha256']
        else:
            data=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==item['git_blob_id']
        verified.append(item['name'])
    assert len(verified)==8
    report('public_parameter_loading')
    rig=Rig(checkpoint,deadline);rig.precision(torch.float32)
    weights={};parameter_hashes={};full=[]
    def take(i,name,t):
        a=t.detach().cpu().numpy().copy();parameter_hashes[f'l{i:02d}_{name}']={'sha256':hashlib.sha256(a.tobytes()).hexdigest(),'shape':list(a.shape),'effective_dtype':str(a.dtype)}
        return a.astype(np.float64)
    for i,layer in enumerate(rig.layers):
        p={'norm':take(i,'post_norm',layer.post_attention_layernorm.weight)}
        for name in ['gate','up','down']:p[name]=take(i,'mlp_'+name,getattr(layer.mlp,name+'_proj').weight)
        weights[i]={'mlp':p}
        if layer.block_type=='full_attention':
            full.append(i);a=layer.self_attn
            assert a.head_dim==256 and a.num_key_value_groups==4
            assert a.q_proj.bias is None and a.o_proj.bias is None
            weights[i]['attention']={'norm':take(i,'input_norm',layer.input_layernorm.weight),'q':take(i,'query_gate',a.q_proj.weight),
                'qnorm':take(i,'query_norm',a.q_norm.weight),'out':take(i,'output',a.o_proj.weight)}
    assert full==[3,7,11,15,19,23] and rig.calls==0
    questions=read(tracked(FRESH/'INPUTS.json'))['questions'];meta=read(tracked(FRESH/'METRICS.json'));cases=read(P18/'DRIVER_CASES.json')
    prefix={};rotations={}
    for q in questions:
        pid=q['id'];path=tracked(FRESH/'contexts'/f'{pid}_incoming_cache.pt')
        cache=torch.load(path,map_location='cpu',weights_only=False)
        prefix[pid]={i:{n:getattr(cache.layers[i],n).detach().cpu().numpy()[0].copy().astype(np.float64) for n in ['keys','values']} for i in full}
        length=prefix[pid][3]['keys'].shape[-2];assert length==len(q['prefix_ids'])
        for t in [1,2]:
            with torch.no_grad():cos,sin=rig.text.rotary_emb(torch.zeros((1,1,1024),dtype=torch.float32),torch.tensor([[[length+t-1]]]*3))
            rotations[pid,t]=(cos[0,0].numpy().astype(np.float64),sin[0,0].numpy().astype(np.float64))
        del cache
    assert rig.calls==0
    dump(ROOT/'PUBLIC_PARAMETER_RECEIPT.json',{'original_loader':'BF16 then FP32 promotion; frozen/eval','arrays':parameter_hashes,'new_backbone_forwards':rig.calls,'model_released_after_copy':True,'model_revision':'2fc06364715b967f1860aea9cf38778875588b17','verified_model_files':verified,'rotary_constants_from_pinned_native_module':True})
    del layer,a,rig;gc.collect();report('public_model_released')
    with np.load(P17/'DIRECTIONS_FRESH.npz',allow_pickle=False) as z:directions=z['vectors'].copy().astype(np.float64)
    dump(ROOT/'DIRECTION_IDENTITY.json',{'parent_file':str(P17/'DIRECTIONS_FRESH.npz'),'file_sha256':sha(P17/'DIRECTIONS_FRESH.npz'),'vector_sha256':[hashlib.sha256(x.astype(np.float32).tobytes()).hexdigest() for x in directions]})
    core={f'b{i}_{suffix}' for i in range(24) for suffix in ['input','mixer','middle','mlp','output']}
    kv={f'm{i}_{n}_current' for i in full for n in ['keys','values']}
    previous={f'm{i}_{n}_previous1' for i in full for n in ['keys','values']}
    normal={}
    def get_case(case):
        pid,t=case['question'],case['token'];name=f'contexts/{pid}_t{t}_baseline.npz' if case['label']=='normal' else meta[int(case['case'].split('_')[-1])]['actual_file']
        return fields(name,'v',core|kv|(previous if t==2 else set()))
    def cache_arrays(pid,t,point,i,q=None):
        result=[]
        for n in ['keys','values']:
            parts=[prefix[pid][i][n]];dparts=[np.zeros_like(parts[0])]
            if t==2:
                parts.append(point[f'm{i}_{n}_previous1'][0].astype(np.float64))
                if q is not None:dparts.append(q[f'm{i}_{n}_previous1'][0].astype(np.float64))
            parts.append(point[f'm{i}_{n}_current'][0].astype(np.float64))
            if q is not None:dparts.append(q[f'm{i}_{n}_current'][0].astype(np.float64))
            result.append(np.concatenate(parts,axis=-2))
            if q is not None:result.append(np.concatenate(dparts,axis=-2))
        return result
    def save_energy(record,tag,case,i):
        for n in ['scores','P','logP','ds','dP']:
            if n in record:arrays[tag+f'_l{i}_{n}']=record[n]
        for h in range(8):
            row={**case,'layer':i,'head':h,'entropy':float(record['entropy'][h]),'stationarity':float(record['stationarity'][h]),
                'minimum_identity_error':float(record['minimum_identity_error'][h]),'candidate_KL':float(record['candidate_KL'][h]),'objective_gap_identity_error':float(record['objective_gap_identity_error'][h])}
            scale=max(1.,abs(float(record['logZ'][h,0])))
            row['pass']=bool(row['stationarity']<=1e-9*scale and row['minimum_identity_error']<=1e-9*scale and row['objective_gap_identity_error']<=1e-9*scale and row['candidate_KL']>=-1e-12)
            if 'dP' in record:
                row['simplex_derivative_error']=float(record['simplex_derivative_error'][h]);row['common_mode_invariance_error']=float(record['common_mode_invariance_error'][h])
                row['pass']=row['pass'] and row['simplex_derivative_error']<=1e-9*max(1.,nrm(record['dP'][h])) and row['common_mode_invariance_error']<=1e-9
            energy_rows.append(row)
    done_cases=set();done_derivs=set()
    def primal(ci):
        deadline();case=cases[ci];pid,t=case['question'],case['token'];point=get_case(case);tag=f'v{ci:03d}'
        if case['label']=='normal':normal[pid,t]=point
        for i in range(24):
            y=mlp(point[f'b{i}_middle'],weights[i]['mlp']);rows.append({**case,'layer':i,'module':'MLP',**compare(y,point[f'b{i}_mlp'])})
            arrays[tag+f'_l{i}_mlp']=y
            external=np.zeros(1024)
            if case['label']=='middle' and t==1 and i==11:
                target=meta[int(case['case'].split('_')[-1])];external=float(target['alpha'])*np.asarray(target['direction'],dtype=np.float64)
            rows.append({**case,'layer':i,'module':'residual_middle',**compare(point[f'b{i}_input']+point[f'b{i}_mixer']+external,point[f'b{i}_middle'])})
            rows.append({**case,'layer':i,'module':'residual_output',**compare(point[f'b{i}_middle']+point[f'b{i}_mlp'],point[f'b{i}_output'])})
            if i in full:
                K,V=cache_arrays(pid,t,point,i);y,rec=attention(point[f'b{i}_input'],K,V,weights[i]['attention'],*rotations[pid,t])
                rows.append({**case,'layer':i,'module':'attention',**compare(y,point[f'b{i}_mixer'])});arrays[tag+f'_l{i}_attention']=y
                save_energy(rec,tag,case,i)
        done_cases.add(ci);STATUS['primal_cases']=len(done_cases)
    panel=[(q['id'],'primary',t) for q in questions for t in [1,2]]+[(pid,'reference',t) for pid in ['r17_cal_spoon','r17_held_panels'] for t in [1,2]]
    def derivative(di):
        deadline();pid,label,t=panel[di];case={'question':pid,'label':label,'token':t,'case':f'derivative_{di:02d}'}
        point=normal[pid,t];q=fields(f'coefficients/{pid}_{label}_t{t}.npz','q',core|kv|(previous if t==2 else set()));tag=f'q{di:03d}'
        for i in range(24):
            y,dy=mlp(point[f'b{i}_middle'],weights[i]['mlp'],q[f'b{i}_middle']);derivative_rows.append({**case,'layer':i,'module':'MLP',**compare(dy,q[f'b{i}_mlp'],True)});arrays[tag+f'_l{i}_mlp']=dy
            derivative_rows.append({**case,'layer':i,'module':'residual_middle',**compare(q[f'b{i}_input']+q[f'b{i}_mixer'],q[f'b{i}_middle'],True)})
            derivative_rows.append({**case,'layer':i,'module':'residual_output',**compare(q[f'b{i}_middle']+q[f'b{i}_mlp'],q[f'b{i}_output'],True)})
            if i in full:
                K,dK,V,dV=cache_arrays(pid,t,point,i,q);y,dy,rec=attention(point[f'b{i}_input'],K,V,weights[i]['attention'],*rotations[pid,t],q[f'b{i}_input'],dK,dV)
                derivative_rows.append({**case,'layer':i,'module':'attention',**compare(dy,q[f'b{i}_mixer'],True)});arrays[tag+f'_l{i}_attention']=dy
                save_energy(rec,tag,case,i)
        done_derivs.add(di);STATUS['derivative_cases']=len(done_derivs)
    for ci,c in enumerate(cases):
        if c['question']=='r17_cal_spoon' and c['label']=='normal':primal(ci)
    for di,c in enumerate(panel):
        if c[0]=='r17_cal_spoon' and c[1]=='primary':derivative(di)
    cal_pass=all(r['pass'] for r in rows+derivative_rows+energy_rows)
    dump(ROOT/'CALIBRATION_GATE.json',{'pass':cal_pass,'primal_cases':len(done_cases),'derivative_cases':len(done_derivs),'failed_rows':[r for r in rows+derivative_rows+energy_rows if not r['pass']]})
    assert cal_pass,'Calibration failure; wider panel not run'
    report('calibration_passed')
    for ci in range(len(cases)):
        if ci not in done_cases:primal(ci)
        if ci%12==0:report('native_module_replay',case=ci)
    for di in range(len(panel)):
        if di not in done_derivs:derivative(di)
        report('analytic_response_replay',case=di)
    for name,h in used.items():assert sha(FRESH/name)==h,name
    assert len(done_cases)==92 and len(done_derivs)==16
    dump(ROOT/'RESULTS.json',{'classification':'Independent equation replay on saved native traces; no new backbone inference','primal_pass':all(r['pass'] for r in rows),'analytic_response_pass':all(r['pass'] for r in derivative_rows),'attention_energy_pass':all(r['pass'] for r in energy_rows),'primal_cases':92,'response_cases':16,'MLP_primal_fields':2208,'attention_primal_fields':552,'MLP_derivative_fields':384,'attention_derivative_fields':96,'attention_energy_head_cases':len(energy_rows),'new_backbone_forwards':0,'GPU_allocations':0,'failed_primal_rows':[r for r in rows if not r['pass']],'failed_response_rows':[r for r in derivative_rows if not r['pass']],'failed_energy_rows':[r for r in energy_rows if not r['pass']],'original_attention_probabilities_not_collected':'Full probabilities are independently reconstructed from native inputs and public parameters; complete attention outputs are compared to original recordings.','nonzero_response_fields':sum(r['native_norm']>1e-12 for r in derivative_rows if r['module'] in ['MLP','attention'])})
    STATUS['status']='complete'

try:main()
except Exception:
    STATUS['status']='failed';dump(ROOT/'FAILURE.json',{'traceback':traceback.format_exc(),'seconds':time.monotonic()-BEGIN});traceback.print_exc()
finally:
    dump(ROOT/'FIELD_RESULTS.json',rows);dump(ROOT/'DERIVATIVE_RESULTS.json',derivative_rows);dump(ROOT/'ATTENTION_ENERGY_HEADS.json',energy_rows);dump(ROOT/'PARENT_USED_HASHES.json',used)
    if arrays:np.savez_compressed(ROOT/'FULL_REPLAY_ARRAYS.npz',**arrays)
    report('exporting')
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    dump(ROOT/'EXPORT_HASHES.json',{p.relative_to(ROOT).as_posix():sha(p) for p in files})
    archive=ROOT.parent/'R4_ATTENTION_MLP_NATIVE_REPLAY_20261015.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=2) as z:
        for p in files+[ROOT/'EXPORT_HASHES.json']:z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    receipt={'status':STATUS['status'],'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'new_backbone_forwards':0,'GPU_allocations':0,'total_seconds_with_export':time.monotonic()-BEGIN,'PID':os.getpid()}
    dump(ROOT.parent/'NATIVE_EXPORT_RECEIPT_20261015.json',receipt);print('R4_ATTENTION_MLP_EXPORT',json.dumps(receipt),flush=True)
if STATUS['status']!='complete':raise SystemExit(1)
