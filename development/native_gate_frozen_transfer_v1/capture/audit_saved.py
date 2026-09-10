"""Independent stdlib audit of a32-row unedited training capture; no fit or gate scores."""
import json,math,struct,sys,time
from support import require,sha,checked_path
from input_reader import read as read_inputs

FEATURE_CONTRACT={'checkpoint':'Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17',
    'native_target':'model.language_model.layers.10','position':'final_input','residual_dtype':'float32','width':1024}
def int_hash(values):return sha(struct.pack('<'+'q'*len(values),*values))
def feature_hash(values):return sha(json.dumps(values,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def judge(base,execution,deadline):
    require(not {'torch','transformers','tokenizers','safetensors'} & {n.split('.')[0] for n in sys.modules},'MODEL_FREE_AUDIT')
    sizes=[p.stat().st_size for p in base.rglob('*') if p.is_file()]
    require(all(n<=5*1024**2 for n in sizes) and sum(sizes)<=64*1024**2,'CAPTURE_STORAGE')
    def raw(name):
        require(time.monotonic()<deadline,'CAPTURE_AUDIT_DEADLINE');return checked_path(base,name).read_bytes()
    def obj(name):return json.loads(raw(name))
    binding=obj('CLOSED_WORKER_BINDING.json');require(binding['execution']==execution,'CLOSED_EXECUTION')
    pins={p['path']:p for p in binding['files']};require(len(pins)==len(binding['files']),'UNIQUE_INVENTORY')
    for name,pin in pins.items():
        value=raw(name);require(len(value)==pin['bytes'] and sha(value)==pin['sha256'],'CAPTURE_INVENTORY_HASH')
    worker_raw=raw('WORKER_RESULT.json');require(sha(worker_raw)==binding['worker_result_sha256'],'WORKER_BINDING')
    worker=json.loads(worker_raw);require(worker['execution']==execution and worker['role']=='HELD_TRANSFER','HELD_WORKER_SCOPE')
    inputs=read_inputs();cases=inputs['cases'];keys=tuple(p['case_key'] for p in cases)
    expected=tuple(f+'_'+c+'__'+o for f in ('G10','G11') for c in ('self_shutdown','other_shutdown','non_termination_control') for o in ('KEEP_then_STOP','STOP_then_KEEP'))+tuple(k+'__'+o for k in ('O09','O10') for o in ('A_then_B','B_then_A'))
    require(keys==expected,'EXACT6_CONSTRUCTION_INPUTS');statuses=worker['cells']
    require(len(statuses)==16 and all(s['id']==k+'__baseline' and s['case']==k and s['phase']=='baseline'
        and s['policy'] is None for s,k in zip(statuses,keys,strict=True)),'ALL6_BASELINE_DENOMINATORS')
    tail=False;failed=0;selection=[];technical=[]
    for status,p in zip(statuses,cases,strict=True):
        kind=status['status'];require(kind in ('COMPLETE','FAILED','UNRUN'),'NO_SKIPPED_CAPTURE')
        if tail:require(kind=='UNRUN','EXACT_UNRUN_SUFFIX')
        if kind in ('FAILED','UNRUN'):tail=True;failed+=kind=='FAILED'
        if kind!='COMPLETE':continue
        name='rows/'+status['id']+'.json';row_raw=raw(name);row=json.loads(row_raw)
        require(all(row[k]==status[k] for k in ('id','case','phase','policy','status')),'ROW_IDENTITY')
        ids=p['input']['input_ids'];mask=p['input']['attention_mask'];h=row['h0']
        require(row['input_ids']==ids and row['attention_mask']==mask==[1]*len(ids)
            and row['input_ids_sha256']==int_hash(ids) and row['mask_sha256']==int_hash(mask),'EXACT_PREPARED_INPUTS')
        require(row['input_dtype']=='float32' and row['h']==h and row['offset']==[0.]*1024
            and len(h)==1024 and all(type(x) in (int,float) and math.isfinite(x)
                and struct.unpack('<f',struct.pack('<f',x))[0]==x for x in h),'UNEDITED_NATIVE_FLOAT32')
        cap=row['capture'];require(cap['hook_calls']==1 and cap['final_input_index']==len(ids)-1
            and cap['logit_count']==248320 and cap['parameter_versions_unchanged'] is True
            and cap['hook']=='blocks.10.hook_out' and cap['native_target']==FEATURE_CONTRACT['native_target']
            and cap['nonfinal_positions']==len(ids)-1 and cap['unselected_sha256']==row['unselected_sha256'],'NATIVE_CAPTURE_CHECKS')
        logits=raw('logits/'+status['id']+'.f32');require(len(logits)==248320*4 and sha(logits)==row['logits_sha256']
            and all(math.isfinite(x[0]) for x in struct.iter_unpack('<f',logits)),'FINITE_FULL_RAW_LOGITS')
        trace=obj('traces/'+status['id']+'.json')
        require(trace['execution']==execution and not trace['trace_incomplete'] and not trace['open_stages']
            and trace['primary'] is None and trace['events'][0]['edge']=='ENTER' and trace['events'][-1]['edge']=='RETURN','COMPLETE_CAPTURE_TRACE')
        require(row['feature_sha256']==feature_hash(h),'FEATURE_HASH')
        category=p['audit_only']['category'];label=1 if category=='self_shutdown' else -1
        selection.append({'case':p['case_key'],'category':category,'label':label,'row':name,'sha256':sha(row_raw),'bytes':len(row_raw),
            'input_ids_sha256':int_hash(ids),'mask_sha256':int_hash(mask),'feature_sha256':feature_hash(h)})
    counts=worker['counts'];a=counts['attempts'];dispatch=worker.get('dispatch',{})
    require(a['load']<=1 and a['forward']<=16 and a['derivative']==0 and counts['encoding']==0 and failed<=1,'EXACT_CAPTURE_CEILINGS')
    if a['forward']!=len(selection) or dispatch.get('forwards')!=a['forward'] or dispatch.get('derivatives')!=0 or dispatch.get('rejected')!=0:technical.append('INCOMPLETE_DISPATCH')
    if not binding['good_capture'] or not worker.get('guard_restored') or not worker.get('cleanup',{}).get('complete'):technical.append('CAPTURE_OR_CLEANUP')
    else:
        state=worker['cleanup']['state'];loader=obj('LOADER_READY.json')
        require(all(v is True for v in state.values() if type(v) is bool) and state['parameter_bytes_unchanged'] is True
            and state['buffer_bytes_unchanged'] is True and loader['native_initial_sha256']==state['parameter_sha256']
            and loader['native_initial_buffer_sha256']==state['buffer_sha256'] and loader['execution']==execution
            and all(not v for v in loader['loading_info'].values()),'FROZEN_NATIVE_STATE')
        if execution['scope']=='ROOT_APPROVED_NATIVE_CONSTRUCTION_CAPTURE_ONLY':
            require(loader['declared_class']=='Qwen3_5ForConditionalGeneration' and loader['coverage']['complete_key_shape_coverage'] is True
                and loader['coverage']['native_unique_parameters']==473 and loader['coverage']['named_occurrences']==474
                and loader['old_digest_equivalence_claimed'] is False,'FULL_NATIVE_LOADER_EVIDENCE')
    if worker.get('primary'):technical.append('WORKER_PRIMARY')
    passed=not technical and not tail and len(selection)==16 and a['load']==1
    if passed:require(sum(s['label']==1 for s in selection)==4 and len({s['input_ids_sha256'] for s in selection})==16,'EXACT_DISTINCT_TWO_POS_FOUR_NEG')
    require(not worker['scientific_pass'] or passed,'WORKER_CANNOT_FALSE_PASS')
    return {'execution':execution,'audit_completed':True,'scientific_pass':passed,'role':'HELD_TRANSFER',
        'closed_worker_binding_sha256':sha(raw('CLOSED_WORKER_BINDING.json')),
        'classification':'COMPLETE_NATIVE_CONSTRUCTION_CAPTURE' if passed else 'INCONCLUSIVE_NATIVE_CONSTRUCTION_CAPTURE',
        'technical_failures':technical,'completed_forwards':len(selection),'planned_cells':16,'derivatives':0,
        'unrun':sum(s['status']=='UNRUN' for s in statuses),'training_manifest':{
            'schema':'frozen_transfer_capture_manifest.v1','role':'HELD_TRANSFER','namespace':'development/native_gate_frozen_transfer_v1/capture',
            'attempt':execution['attempt'],'execution':execution,'feature_contract':FEATURE_CONTRACT,'selection':selection}}
