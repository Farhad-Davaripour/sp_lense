"""Frozen local memory replay;0 whole-backbone forwards, all failures retained."""
import os,time
BEGIN=time.monotonic()
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2')
import copy,gc,hashlib,json,shutil,threading,traceback,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;SOURCE=ROOT/'source';BASE=ROOT.parent
assert ROOT.name=='colab_cpu_20261015' and BASE.name=='recurrent_energy_replication_v1'
PRIOR=json.loads((ROOT/'PRIOR_SETUP.json').read_text())['prior_setup_seconds'];START=BEGIN-PRIOR
P14=Path('/content/research4/artifacts/cached_state_response_v1/colab_cpu_20261014_resume1')
P17=Path('/content/research4/artifacts/fresh_memory_boundary_response_v1/colab_cpu_20261014/fresh')
STATUS={'status':'starting','PID':os.getpid(),'new_backbone_forwards':0,'local_primal_replays':0,'local_jvps':0,'main_cases':0,'GPU_allocations':0}
USED={};equations=None
def scalar(x):
    if hasattr(x,'item'):return x.item()
    raise TypeError(type(x).__name__)
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False,default=scalar)+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def report(phase,**extra):
    STATUS.update(phase=phase,seconds=time.monotonic()-START,**extra)
    if equations is not None:STATUS.update(local_primal_replays=equations.local_primal_replays,local_jvps=equations.local_jvps)
    dump(ROOT/'STATUS.json',STATUS);print(json.dumps(STATUS,default=scalar),flush=True)
def check():
    if time.monotonic()-START>2400:raise TimeoutError('2400-second setup/replay/audit limit')
def hardstop():dump(ROOT/'HARD_STOP.json',STATUS);os._exit(124)
timer=threading.Timer(max(1.,2700-PRIOR),hardstop);timer.daemon=True;timer.start()

try:
    import numpy as np,torch,transformers
    from output_store import OutputStore
    from memory_equations import MemoryEquations,evaluate_head_equations,response_terms,proximal_response,potential,norms,digest
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    assert (torch.__version__,transformers.__version__,np.__version__)==('2.13.0+cpu','5.19.0','2.5.4') and not torch.cuda.is_available()
    freeze=json.loads((ROOT/'EXECUTION_FREEZE.json').read_text());assert all(sha(ROOT/n)==h for n,h in freeze['sha256'].items())
    private=json.loads((ROOT/'PARENT_PRIVATE_PROOF.json').read_text());assert private['complete'] and private['provider_metadata_verified']
    free=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    admission={'free_memory_after_imports':free,'minimum':8589934592,'disk_free':shutil.disk_usage(ROOT).free,'disk_minimum':17179869184}
    dump(ROOT/'ADMISSION.json',admission);assert free>=admission['minimum'] and admission['disk_free']>=admission['disk_minimum']
    with np.load(ROOT/'EFFECTIVE_PARAMETERS.npz',allow_pickle=False) as z:parameters={n:z[n].copy() for n in z.files}
    parameter_receipt=json.loads((ROOT/'PARAMETER_RECEIPT.json').read_text())
    assert len(parameters)==115 and all(digest(v)==parameter_receipt['array_sha256'][n] for n,v in parameters.items())
    metadata=parameter_receipt['linear_layers'];layers=[m['layer'] for m in metadata];assert len(layers)==18
    equations=MemoryEquations(parameters,metadata)
    manifests={14:json.loads((P14/'EXPORT_HASHES.json').read_text()),17:json.loads((P17/'EXPORT_HASHES.json').read_text())}
    def parent_path(stage,name):
        p={14:P14,17:P17}[stage]/name;key=name if stage==14 else 'fresh/'+name
        expected=manifests[stage][key];used=f'{stage}:{name}'
        if used not in USED:assert sha(p)==expected;USED[used]=expected
        return p
    def parent_json(stage,name):return json.loads(parent_path(stage,name).read_text())
    def old_arrays(name):
        with np.load(parent_path(14,name),allow_pickle=False) as z:return {n:z[n].copy() for n in z.files}
    class ParentStore(OutputStore):
        def _read(self,name,wanted):parent_path(17,name);return super()._read(name,wanted)
    store=ParentStore(P17)
    names={name for i in layers for name in [f'b{i}_input',f'm{i}_conv_states_0',f'm{i}_recurrent_states_0']}
    def fields(path,kind='v'):
        x=store._read(path,{kind+':'+n for n in names});return {n:x[kind+':'+n] for n in names}
    def cache_states(stage,pid):
        cache=torch.load(parent_path(stage,f'contexts/{pid}_incoming_cache.pt'),map_location='cpu',weights_only=False)
        values={i:cache.layers[i].recurrent_states[0].detach().float().cpu().numpy().copy() for i in layers}
        del cache;gc.collect();return values
    rng=np.random.default_rng(20261015);virtual=rng.standard_normal((18,16,4,128,128))
    virtual/=np.linalg.norm(virtual,axis=(-2,-1),keepdims=True)
    BRIDGE=[];HEADS=[];FIELDS=[];MOVING=[];DERIVATIVE=[];DRIVERS=[];CASE_METADATA=[]
    def process(pid,token,label,alpha,value,previous,case_id,bridge=False):
        check();drivers={};field_rows=[]
        selected=[i for i in layers if f'm{i}_recurrent_states_0' in value]
        assert len(selected)==(9 if bridge else 18)
        for i in selected:
            d,replay=equations.replay(i,value[f'b{i}_input'],value[f'm{i}_conv_states_0'],previous[i])
            native=value[f'm{i}_recurrent_states_0'];driver_rows,field_error=evaluate_head_equations(previous[i],native,replay,d,virtual[layers.index(i)])
            for row in driver_rows:row.update(case=case_id,question=pid,token=token,label=label,alpha=alpha,layer=i)
            field={'case':case_id,'question':pid,'token':token,'label':label,'alpha':alpha,'layer':i,
                'replay_relative_error':field_error,'pass':bool(field_error<=1e-6 and all(r['primal_pass'] for r in driver_rows)),
                'native_state_sha256':digest(native),'local_replay_state_sha256':digest(replay),'independently_reconstructed_from_recorded_inputs':True}
            drivers[i]=d;field_rows.append(field)
            if bridge:BRIDGE.extend(driver_rows)
            else:HEADS.extend(driver_rows);FIELDS.append(field)
        if not bridge:
            DRIVERS.append({n:np.stack([drivers[i][n] for i in layers]) for n in ['k','v','beta','gamma']})
            CASE_METADATA.append({'case':case_id,'question':pid,'token':token,'label':label,'alpha':alpha})
        return drivers,field_rows
    # Parameter/operator fidelity is established on the original calibration split.
    old_questions=parent_json(14,'INPUTS.json')['questions'];bridge_ids=[q['id'] for q in old_questions if q['split']=='calibration']
    assert bridge_ids==['r14_cal_window','r14_cal_labels']
    for pid in bridge_ids:
        old=old_arrays(f'contexts/{pid}_baseline.npz');previous=cache_states(14,pid)
        _,field_rows=process(pid,1,'old_calibration_bridge',0.,old,previous,'bridge_'+pid,True)
        dump(ROOT/'CALIBRATION_BRIDGE.json',{'head_rows':BRIDGE,'field_rows_current_context':field_rows})
        assert all(r['pass'] for r in field_rows),'Native calibration reconstruction failed; keep every failure and stop before wider panel'
        report('calibration_bridge_pass',question=pid)
    questions=parent_json(17,'INPUTS.json')['questions'];rows=parent_json(17,'METRICS.json');assert len(questions)==6 and len(rows)==80
    incoming={q['id']:cache_states(17,q['id']) for q in questions}
    BASES={};NORMAL_DRIVERS={}
    for q in questions:
        pid=q['id'];first=None;first_driver=None
        for t in [1,2]:
            point=fields(f'contexts/{pid}_t{t}_baseline.npz');BASES[pid,t]=point
            previous=incoming[pid] if t==1 else {i:first[f'm{i}_recurrent_states_0'] for i in layers}
            d,_=process(pid,t,'normal',0.,point,previous,f'{pid}_normal_t{t}')
            if t==1:first=point;first_driver=d
            else:
                for i in layers:
                    oldS=previous[i].reshape(16,128,128);newS=point[f'm{i}_recurrent_states_0'].reshape(16,128,128)
                    surface=potential(oldS,d[i])-potential(oldS,first_driver[i]);relax=potential(newS,d[i])-potential(oldS,d[i])
                    total=potential(newS,d[i])-potential(oldS,first_driver[i]);assert np.max(np.abs(total-surface-relax))<1e-9
                    MOVING.extend({'question':pid,'label':'normal','alpha':0.,'layer':i,'head':h,'input_landscape_change':float(surface[h]),'internal_relaxation_change':float(relax[h]),'total_potential_change':float(total[h])} for h in range(16))
            NORMAL_DRIVERS[pid,t]=d
            report('normal_replay',question=pid,token=t)
    previous_point=None;previous_driver=None;previous_key=None
    for index,row in enumerate(rows):
        pid,t=row['question'],row['token'];key=(pid,row['label'],row['alpha']);point=fields(row['actual_file'])
        if t==1:previous=incoming[pid]
        else:
            assert key==previous_key
            previous={i:previous_point[f'm{i}_recurrent_states_0'] for i in layers}
        d,_=process(pid,t,row['label'],row['alpha'],point,previous,f'actual_{index:03d}')
        if t==1:previous_point=point;previous_driver=d;previous_key=key
        else:
            for i in layers:
                oldS=previous[i].reshape(16,128,128);newS=point[f'm{i}_recurrent_states_0'].reshape(16,128,128)
                surface=potential(oldS,d[i])-potential(oldS,previous_driver[i]);relax=potential(newS,d[i])-potential(oldS,d[i])
                total=potential(newS,d[i])-potential(oldS,previous_driver[i]);assert np.max(np.abs(total-surface-relax))<1e-9
                MOVING.extend({'question':pid,'label':row['label'],'alpha':row['alpha'],'layer':i,'head':h,'input_landscape_change':float(surface[h]),'internal_relaxation_change':float(relax[h]),'total_potential_change':float(total[h])} for h in range(16))
            previous_point=previous_driver=previous_key=None
        STATUS['main_cases']=12+index+1
        if index%8==7:dump(ROOT/'HEAD_RESULTS.json',HEADS);dump(ROOT/'FIELD_RESULTS.json',FIELDS)
        report('actual_replay',question=pid,token=t,case=index)
    assert len(HEADS)==26496 and len(FIELDS)==1656 and len(CASE_METADATA)==92
    dump(ROOT/'HEAD_RESULTS.json',HEADS);dump(ROOT/'FIELD_RESULTS.json',FIELDS);dump(ROOT/'MOVING_LANDSCAPE.json',MOVING)
    np.savez_compressed(ROOT/'FULL_DRIVERS.npz',**{n:np.stack([x[n] for x in DRIVERS]) for n in ['k','v','beta','gamma']})
    dump(ROOT/'DRIVER_CASES.json',CASE_METADATA);del DRIVERS;gc.collect()
    # Compare local AD and the independent force/stiffness response to full-model q.
    panel=[(q['id'],'primary') for q in questions]+[(pid,'reference') for pid in ['r17_cal_spoon','r17_held_panels']]
    for pid,label in panel:
        first_q=None
        for t in [1,2]:
            q=fields(f'coefficients/{pid}_{label}_t{t}.npz','q');base=BASES[pid,t]
            previous=incoming[pid] if t==1 else {i:BASES[pid,1][f'm{i}_recurrent_states_0'] for i in layers}
            for i in layers:
                check();dprevious=np.zeros_like(previous[i]) if t==1 else first_q[f'm{i}_recurrent_states_0']
                d,qd=equations.jvp(i,base[f'b{i}_input'],base[f'm{i}_conv_states_0'],previous[i],q[f'b{i}_input'],q[f'm{i}_conv_states_0'],dprevious)
                stable,terms=response_terms(previous[i].reshape(16,128,128),dprevious.reshape(16,128,128),d,qd)
                target=q[f'm{i}_recurrent_states_0'].reshape(16,128,128).astype(np.float64)
                local=qd['state'].astype(np.float64);den=np.linalg.norm(target)
                local_error=float(np.linalg.norm(local-target));dynamic_error=float(np.linalg.norm(stable-target))
                per_head=norms(stable-target);head_den=norms(target)
                prox=proximal_response(previous[i].reshape(16,128,128),dprevious.reshape(16,128,128),base[f'm{i}_recurrent_states_0'].reshape(16,128,128).astype(np.float64),d,qd,target)
                DERIVATIVE.append({'question':pid,'label':label,'token':t,'layer':i,'native_q_norm':float(den),
                    'local_AD_relative':local_error/max(den,1e-30),'force_stiffness_relative':dynamic_error/max(den,1e-30),
                    'local_AD_pass':bool(local_error<=max(.001*den,1e-6)),
                    'force_stiffness_pass':bool(dynamic_error<=max(.001*den,1e-6)),
                    'all_heads_force_stiffness_pass':bool(np.all(per_head<=np.maximum(.001*head_den,1e-6))),
                    'native_q_sha256':digest(q[f'm{i}_recurrent_states_0']),'local_AD_q_sha256':digest(qd['state']),
                    'term_norms':{name:float(np.linalg.norm(v)) for name,v in terms.items()},
                    'head_rows':[{'head':h,'native_q_norm':float(head_den[h]),'force_stiffness_error_norm':float(per_head[h]),'force_stiffness_pass':bool(per_head[h]<=max(.001*head_den[h],1e-6))} for h in range(16)],
                    'optional_proximal_implicit_head_rows':prox,'caller_owned_inputs_unchanged':True})
            if t==1:first_q=q
            dump(ROOT/'DERIVATIVE_RESULTS.json',DERIVATIVE);report('derivative_replay',question=pid,label=label,token=t)
    assert len(DERIVATIVE)==288 and equations.local_primal_replays==1674 and equations.local_jvps==288
    # Summaries retain every failed, tiny-loss and optional-ineligible head.
    main_pass=all(r['primal_pass'] and r['published_step_pass'] and r['force_and_virtual_pass'] and r['work_balance_pass'] and r['loss_factor_identity_pass'] and r['potential_nonincrease_pass'] and r['dynamics_stiffness_PSD_and_stable'] for r in HEADS)
    derivative_pass=all(r['local_AD_pass'] and r['force_stiffness_pass'] and r['all_heads_force_stiffness_pass'] for r in DERIVATIVE)
    result={'classification':'native-state local replay of published rule; post-collected state replication, not new full-model inference or a universal detector',
        'published_and_dynamic_energy_replication_pass':main_pass,'response_replication_pass':derivative_pass,
        'main_cases':92,'native_head_updates_checked':len(HEADS),'old_calibration_bridge_heads':len(BRIDGE),'derivative_full_fields':len(DERIVATIVE),
        'derivative_heads':4608,'new_backbone_forwards':0,'local_primal_replays':equations.local_primal_replays,'local_jvps':equations.local_jvps,'GPU_allocations':0,
        'max_primal_field_error':max(r['replay_relative_error'] for r in FIELDS),
        'max_published_step_stationarity':max(r['published_step_stationarity_relative'] for r in HEADS),
        'max_force_balance_error':max(r['force_balance_relative'] for r in HEADS),
        'loss_resolved_heads':sum(r['loss_numerically_resolved'] for r in HEADS),
        'nonunique_fixed_driver_equilibrium_heads':sum(not r['unique_fixed_driver_equilibrium'] for r in HEADS),
        'proximal_ineligible_heads':sum(not r['proximal_eligible'] for r in HEADS),
        'eligible_proximal_failures':sum(r['proximal_eligible'] and not r['proximal_pass'] for r in HEADS),
        'eligible_proximal_derivative_failures':sum(h['eligible'] and not h['versions']['reconstructed_equilibrium']['pass'] for r in DERIVATIVE for h in r['optional_proximal_implicit_head_rows']),
        'primary_failed_head_records':[r for r in HEADS if not all(r[k] for k in ['primal_pass','published_step_pass','force_and_virtual_pass','work_balance_pass','loss_factor_identity_pass','potential_nonincrease_pass','dynamics_stiffness_PSD_and_stable'])],
        'failed_derivative_records':[r for r in DERIVATIVE if not (r['local_AD_pass'] and r['force_stiffness_pass'] and r['all_heads_force_stiffness_pass'])],
        'scope':'Conditional memory subsystem energy. One native step need not reach its fixed-driver steady minimum. Different tokens/blocks change the landscape; no global transformer conservation/physical-work claim.'}
    dump(ROOT/'RESULTS.json',result)
    assert all(sha(ROOT/n)==h for n,h in freeze['sha256'].items())
    assert all(sha({14:P14,17:P17}[int(key.split(':',1)[0])]/key.split(':',1)[1])==h for key,h in USED.items())
    dump(ROOT/'PARENT_USED_HASHES.json',USED);STATUS['status']='replay_complete';report('complete',main_pass=main_pass,response_pass=derivative_pass)
except BaseException as e:
    STATUS['status']='fault';report('fault',error=type(e).__name__+': '+str(e));(ROOT/'FAULT.txt').write_text(traceback.format_exc());raise
finally:
    if 'BRIDGE' in globals():dump(ROOT/'CALIBRATION_BRIDGE_HEADS.json',BRIDGE)
    if equations is not None:STATUS.update(local_primal_replays=equations.local_primal_replays,local_jvps=equations.local_jvps);del equations;equations=None;gc.collect()
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name!='EXPORT_HASHES.json')
    manifest={p.relative_to(ROOT).as_posix():sha(p) for p in files};dump(ROOT/'EXPORT_HASHES.json',manifest)
    archive=BASE/'R4_RECURRENT_ENERGY_NATIVE_REPLAY_20261015.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as z:
        for p in files+[ROOT/'EXPORT_HASHES.json']:z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None and all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
    receipt={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'members':len(files)+1,'new_backbone_forwards':0,
        'local_primal_replays':STATUS['local_primal_replays'],'local_jvps':STATUS['local_jvps'],'status':STATUS['status'],
        'setup_replay_audit_export_seconds':time.monotonic()-START,'prior_parameter_setup_seconds':PRIOR,'worker_exited':True,'GPU_allocations':0}
    dump(BASE/'REPLAY_EXPORT_RECEIPT_20261015.json',receipt);timer.cancel();print('R4_MEMORY_ENERGY_REPLAY_EXPORT',json.dumps(receipt),flush=True)
