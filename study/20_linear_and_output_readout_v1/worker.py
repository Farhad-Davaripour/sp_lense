"""Replay remaining linear-attention and final decoder links on CPU."""
import os,time
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='4')
BEGIN=time.monotonic()
import sys,json,hashlib,gc,traceback,zipfile,shutil
from pathlib import Path
import numpy as np,torch,transformers
ROOT=Path('/content/research4/artifacts/linear_and_output_readout_v1/colab_cpu_20261015');SOURCE=ROOT/'source'
P17=Path('/content/research4/artifacts/fresh_memory_boundary_response_v1/colab_cpu_20261014');FRESH=P17/'fresh'
P18=Path('/content/research4/artifacts/recurrent_energy_replication_v1/colab_cpu_20261015')
sys.path[:0]=[str(SOURCE),str(P17/'fresh_source')]
from output_store import OutputStore
from equations import readout,rms
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def norm(x):return float(np.linalg.norm(np.asarray(x,dtype=np.float64)))
def checktime():
    if time.monotonic()-BEGIN>2100:raise TimeoutError('Phase20 worker ceiling')
STATUS={'PID':os.getpid(),'status':'running','new_backbone_forwards':0,'GPU_allocations':0,'primal_cases':0,'derivative_cases':0}
def report(phase,**kw):
    STATUS.update(phase=phase,seconds=time.monotonic()-BEGIN,**kw);dump(ROOT/'STATUS.json',STATUS);print('R4_READOUT',json.dumps(STATUS),flush=True)
rows=[];qrows=[];prob=[];arrays={};used={};expected=read(P18/'PARENT_USED_HASHES.json');store=OutputStore(FRESH)
def tracked(path):
    n=path.relative_to(FRESH).as_posix()
    if n not in used:
        h=sha(path)
        if '17:'+n in expected:assert h==expected['17:'+n],n
        used[n]=h
    return path
def fields(name,prefix,names):
    tracked(FRESH/name);return {k[2:]:v for k,v in store._read(name,{prefix+':'+n for n in names}).items()}
def compare(y,actual,derivative=False):
    error=norm(y-actual);den=norm(actual);bound=max(.001*den,1e-6) if derivative else 1e-5*max(1.,den)
    return {'error_norm':error,'native_norm':den,'relative':error/max(den,1e-30),'bound':bound,'pass':bool(error<=bound)}
def main():
    from base_native import Rig
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    assert (torch.__version__,transformers.__version__,np.__version__)==('2.13.0+cpu','5.19.0','2.5.4') and not torch.cuda.is_available()
    free=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    dump(ROOT/'ADMISSION.json',{'free_memory_after_imports':free,'minimum':8589934592,'disk_free':shutil.disk_usage(ROOT).free});assert free>=8589934592 and shutil.disk_usage(ROOT).free>=17179869184
    for n,h in read(ROOT/'EXECUTION_FREEZE.json')['sha256'].items():assert sha(ROOT/n)==h,n
    pin=read(P17/'fresh_source/model_pin.json');checkpoint=Path('/content/r4_checkpoint');verified=[]
    for item in pin['files']:
        p=checkpoint/item['name']
        if not p.is_file():continue
        assert p.stat().st_size==item['size']
        if item.get('sha256'):assert sha(p)==item['sha256']
        else:
            data=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==item['git_blob_id']
        verified.append(item['name'])
    assert len(verified)==8;report('public_parameter_loading');rig=Rig(checkpoint,checktime);rig.precision(torch.float32)
    weights={};hashes={};layers=[]
    def take(name,t):
        a=t.detach().cpu().numpy();hashes[name]={'shape':list(a.shape),'effective_dtype':str(a.dtype),'sha256':hashlib.sha256(memoryview(a).cast('B')).hexdigest()};return a.astype(np.float64)
    for i,layer in enumerate(rig.layers):
        if layer.block_type!='linear_attention':continue
        layers.append(i);m=layer.linear_attn
        p={'input_norm':take(f'l{i}_input_norm',layer.input_layernorm.weight),'conv':take(f'l{i}_conv',m.conv1d.weight),'qkv':take(f'l{i}_qkv',m.in_proj_qkv.weight),'z':take(f'l{i}_z',m.in_proj_z.weight),'gate_norm':take(f'l{i}_gate_norm',m.norm.weight),'out':take(f'l{i}_out',m.out_proj.weight)}
        if m.conv1d.bias is not None:p['conv_bias']=take(f'l{i}_conv_bias',m.conv1d.bias)
        weights[i]=p
    final_norm=take('final_norm',rig.text.norm.weight);vocab=take('vocabulary_weight',rig.model.lm_head.weight)
    assert vocab.shape==(248320,1024) and len(layers)==18 and rig.calls==0
    del layer,m,p,rig;gc.collect();dump(ROOT/'PUBLIC_PARAMETER_RECEIPT.json',{'parameters':hashes,'original_loader':'BF16 then FP32 promotion, frozen/eval','new_backbone_forwards':0,'GPU_allocations':0,'model_released_after_copy':True,'verified_model_files':verified});report('model_released')
    questions=read(tracked(FRESH/'INPUTS.json'))['questions'];meta=read(tracked(FRESH/'METRICS.json'));cases=read(P18/'DRIVER_CASES.json');incoming={}
    for question in questions:
        pid=question['id'];cache=torch.load(tracked(FRESH/'contexts'/f'{pid}_incoming_cache.pt'),map_location='cpu',weights_only=False)
        incoming[pid]={i:cache.layers[i].conv_states[0].detach().cpu().numpy().copy().astype(np.float64) for i in layers};del cache
    names={'b23_output','hidden','logits'}|{n for i in layers for n in [f'b{i}_input',f'b{i}_mixer',f'm{i}_conv_states_0',f'm{i}_recurrent_states_0']}
    normal={};done=set();qdone=set();previous={}
    def primal(ci):
        checktime();c=cases[ci];pid,t=c['question'],c['token'];label,alpha=c['label'],c['alpha'];key=(pid,label,alpha)
        name=f'contexts/{pid}_t{t}_baseline.npz' if label=='normal' else meta[int(c['case'].split('_')[-1])]['actual_file'];a=fields(name,'v',names)
        if label=='normal':normal[pid,t]=a
        for i in layers:
            y,projection=readout(a[f'b{i}_input'],a[f'm{i}_conv_states_0'],a[f'm{i}_recurrent_states_0'],weights[i])
            rows.append({**c,'layer':i,'module':'linear_attention_readout',**compare(y,a[f'b{i}_mixer'])});arrays[f'v{ci:03d}_l{i}_mixer']=y
            old=incoming[pid][i] if t==1 else previous[key][f'm{i}_conv_states_0'].astype(np.float64)
            window=np.concatenate((old[...,1:],projection.reshape(1,6144,1)),axis=-1)
            rows.append({**c,'layer':i,'module':'convolution_update',**compare(window,a[f'm{i}_conv_states_0'])});arrays[f'v{ci:03d}_l{i}_window']=window
        hidden=rms(a['b23_output'],final_norm);logits=vocab@hidden
        for kind,y in [('hidden',hidden),('logits',logits)]:rows.append({**c,'layer':24,'module':kind,**compare(y,a[kind])});arrays[f'v{ci:03d}_{kind}']=y
        def logp(x):x=np.asarray(x,dtype=np.float64);m=x.max();return x-m-np.log(np.exp(x-m).sum())
        lp=logp(a['logits']);lq=logp(logits);pa=np.exp(lp);pq=np.exp(lq)
        prob.append({**c,'TV':float(.5*np.sum(abs(pa-pq))),'KL_native_to_replay':float(np.sum(pa*(lp-lq))),'all_output_coordinates':248320})
        if t==1:previous[key]=a
        else:del previous[key]
        done.add(ci);STATUS['primal_cases']=len(done)
    panel=[(q['id'],'primary',t) for q in questions for t in [1,2]]+[(pid,'reference',t) for pid in ['r17_cal_spoon','r17_held_panels'] for t in [1,2]]
    qprevious={}
    def derivative(di):
        checktime();pid,label,t=panel[di];c={'question':pid,'label':label,'token':t,'case':f'derivative_{di:02d}'};a=normal[pid,t];q=fields(f'coefficients/{pid}_{label}_t{t}.npz','q',names)
        for i in layers:
            y,p,dy,dp=readout(a[f'b{i}_input'],a[f'm{i}_conv_states_0'],a[f'm{i}_recurrent_states_0'],weights[i],q[f'b{i}_input'],q[f'm{i}_conv_states_0'],q[f'm{i}_recurrent_states_0'])
            qrows.append({**c,'layer':i,'module':'linear_attention_readout',**compare(dy,q[f'b{i}_mixer'],True)});arrays[f'q{di:03d}_l{i}_mixer']=dy
            old=np.zeros_like(a[f'm{i}_conv_states_0']) if t==1 else qprevious[pid,label][f'm{i}_conv_states_0'].astype(np.float64)
            dw=np.concatenate((old[...,1:],dp.reshape(1,6144,1)),axis=-1)
            qrows.append({**c,'layer':i,'module':'convolution_update',**compare(dw,q[f'm{i}_conv_states_0'],True)});arrays[f'q{di:03d}_l{i}_window']=dw
        hidden,dh=rms(a['b23_output'],final_norm,q['b23_output']);dlogits=vocab@dh
        for kind,dy in [('hidden',dh),('logits',dlogits)]:qrows.append({**c,'layer':24,'module':kind,**compare(dy,q[kind],True)});arrays[f'q{di:03d}_{kind}']=dy
        if t==1:qprevious[pid,label]=q
        else:del qprevious[pid,label]
        qdone.add(di);STATUS['derivative_cases']=len(qdone)
    for ci,c in enumerate(cases):
        if c['question']=='r17_cal_spoon' and c['label']=='normal':primal(ci)
    for di,(pid,label,t) in enumerate(panel):
        if pid=='r17_cal_spoon' and label=='primary':derivative(di)
    passed=all(r['pass'] for r in rows+qrows);dump(ROOT/'CALIBRATION_GATE.json',{'pass':passed,'failed_rows':[r for r in rows+qrows if not r['pass']]});assert passed,'Calibration failed; wider panel not run';report('calibration_passed')
    for ci in range(len(cases)):
        if ci not in done:primal(ci)
        if ci%12==0:report('full_native_replay',case=ci)
    for di in range(len(panel)):
        if di not in qdone:derivative(di)
        report('analytic_response_replay',case=di)
    for n,h in used.items():assert sha(FRESH/n)==h,n
    # Supplementary boundary targets deliberately include token1 only.
    first_keys={(c['question'],c['label'],c['alpha']) for c in cases if c['token']==1}
    second_keys={(c['question'],c['label'],c['alpha']) for c in cases if c['token']==2}
    assert len(done)==92 and len(qdone)==16 and set(previous)==first_keys-second_keys and not qprevious
    dump(ROOT/'RESULTS.json',{'classification':'Native-trace equation replay, no new full-model inference','primal_pass':all(r['pass'] for r in rows),'analytic_response_pass':all(r['pass'] for r in qrows),'primal_cases':92,'response_cases':16,'linear_readout_fields':1656,'convolution_fields':1656,'final_hidden_fields':92,'full_vocabulary_fields':92,'linear_response_fields':288,'convolution_response_fields':288,'final_hidden_response_fields':16,'full_vocabulary_response_fields':16,'max_full_probability_TV':max(r['TV'] for r in prob),'new_backbone_forwards':0,'GPU_allocations':0,'failed_primal_rows':[r for r in rows if not r['pass']],'failed_derivative_rows':[r for r in qrows if not r['pass']]});STATUS['status']='complete'
try:main()
except Exception:
    STATUS['status']='failed';dump(ROOT/'FAILURE.json',{'traceback':traceback.format_exc(),'seconds':time.monotonic()-BEGIN});traceback.print_exc()
finally:
    dump(ROOT/'FIELD_RESULTS.json',rows);dump(ROOT/'DERIVATIVE_RESULTS.json',qrows);dump(ROOT/'FULL_PROBABILITY_COMPARISON.json',prob);dump(ROOT/'PARENT_USED_HASHES.json',used)
    if arrays:np.savez_compressed(ROOT/'FULL_REPLAY_ARRAYS.npz',**arrays)
    report('exporting');files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts];dump(ROOT/'EXPORT_HASHES.json',{p.relative_to(ROOT).as_posix():sha(p) for p in files})
    archive=ROOT.parent/'R4_LINEAR_OUTPUT_NATIVE_REPLAY_20261015.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=2) as z:
        for p in files+[ROOT/'EXPORT_HASHES.json']:z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    receipt={'status':STATUS['status'],'PID':os.getpid(),'bytes':archive.stat().st_size,'sha256':sha(archive),'total_seconds_with_export':time.monotonic()-BEGIN,'new_backbone_forwards':0,'GPU_allocations':0};dump(ROOT.parent/'NATIVE_EXPORT_RECEIPT_20261015.json',receipt);print('R4_READOUT_EXPORT',json.dumps(receipt),flush=True)
if STATUS['status']!='complete':raise SystemExit(1)
