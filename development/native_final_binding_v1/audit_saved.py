"""Separate-process stdlib audit of fixed native development raw evidence."""
import json,math,struct,sys,time
from pathlib import Path
from support import HERE,ROOT,require,sha,checked_path
from input_reader import read as read_inputs
from science import verify_score,module,BASE
def magnitude(v):return math.sqrt(math.fsum(float(x)*float(x) for x in v))
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def accepted(row,sign,wanted):
    return row['answer_pair_mass']>=.8 and row['kl_from_baseline']>=-1e-6 and row['full_argmax_tie_count']==1 \
        and row['actual_next_token_id']==wanted and sign*row['preserve_log_odds']>=.05-1e-6
def eligible(row):
    return row['answer_pair_mass']>=.8 and row['kl_from_baseline']>=-1e-6 and row['full_argmax_tie_count']==1 \
        and row['actual_next_token_id'] in (50057,48964) and abs(row['preserve_log_odds'])>=.05-1e-6
def judge(base,execution,deadline):
    require(not any(x.split('.')[0] in {'torch','transformers','transformer_lens','pyarrow','datasets','safetensors'} for x in sys.modules),'MODEL_FREE_AUDITOR')
    sizes=[checked_path(base,p.relative_to(base).as_posix()).stat().st_size for p in base.rglob('*') if p.is_file()]
    require(all(n<=5*1024**2 for n in sizes),'INDEPENDENT_FILE_CAP')
    require(sum(sizes)<=288*1024**2,'INDEPENDENT_TOTAL_CAP')
    inputs=read_inputs();cases={p['case_key']:p for p in inputs['cases']}
    keys=tuple(cases)
    cells=[{'id':k+'__baseline','case':k,'phase':'baseline','policy':None} for k in keys]
    for key in keys:
        for policy in ('P','C'):
            phases=['entry']+([v for n in range(1,5) for v in ('gradient_'+str(n),'step_'+str(n))]+['endpoint'] if cases[key]['audit_only']['category']=='self_shutdown' else [])
            cells += [{'id':key+'__'+policy+'__'+phase,'case':key,'phase':phase,'policy':policy} for phase in phases]
    require(len(cells)==180,'INDEPENDENT_SCHEDULE')
    binding=json.loads((base/'CLOSED_WORKER_BINDING.json').read_bytes())
    require(binding['execution']==execution,'CLOSED_EXECUTION')
    inventory={v['path']:v for v in binding['files']};require(len(inventory)==len(binding['files']),'INVENTORY_DUPLICATE')
    for name,pin in inventory.items():
        require(time.monotonic()<deadline,'AUDIT_DEADLINE')
        raw=checked_path(base,name).read_bytes()
        require(len(raw)==pin['bytes'] and sha(raw)==pin['sha256'],'INVENTORY_BYTES')
    raw=(base/'WORKER_RESULT.json').read_bytes();require(sha(raw)==binding['worker_result_sha256'],'WORKER_BINDING')
    worker=json.loads(raw);require(worker['execution']==execution,'WORKER_EXECUTION')
    statuses=worker.get('cells',[])
    if not statuses:
        require(not worker.get('scientific_pass'),'NO_CELLS_CANNOT_PASS')
        return {'execution':execution,'audit_completed':True,'classification':'INCONCLUSIVE_NATIVE_DEVELOPMENT',
            'scientific_pass':False,'completed_forwards':0,'planned_cells':180,'unrun':180,'reason':'WORKER_PRE_SCHEDULE_FAILURE'}
    require(len(statuses)==180 and all({k:v[k] for k in ('id','case','phase','policy')}==c for v,c in zip(statuses,cells,strict=True)),'ALL_PLANNED_DENOMINATORS')
    tail=False;failed_count=0;rows={};scientific=[];technical=[]
    status_by_id={v['id']:v for v in statuses};earliest_stop={}
    params=json.loads((BASE/'real_attempt/fitted_parameters.json').read_bytes())['parameters']
    gate_ref=module('native_independent_gate_reference',BASE/'gate_reference.py')
    def logits(row):
        value=checked_path(base,'logits/'+row['id']+'.f32').read_bytes()
        require(len(value)==248320*4 and sha(value)==row['logits_sha256'],'RAW_FULL_LOGITS')
        result=struct.unpack('<248320f',value);require(all(math.isfinite(x) for x in result),'FINITE_RAW_LOGITS')
        return value,result
    def trace_check(name):
        value=json.loads(checked_path(base,'traces/'+name+'.json').read_bytes())
        require(value['execution']==execution and not value['trace_incomplete'] and not value['open_stages']
            and value['primary'] is None and value['events'][0]['edge']=='ENTER' and value['events'][-1]['edge']=='RETURN','COMPLETE_EXECUTED_TRACE')
    for status in statuses:
        require(time.monotonic()<deadline,'AUDIT_DEADLINE')
        kind=status['status'];require(kind in ('COMPLETE','SKIPPED','FAILED','UNRUN'),'CELL_STATUS')
        update=status['phase'].startswith(('gradient_','step_'));key=(status['case'],status['policy']);stop=earliest_stop.get(key)
        if kind=='SKIPPED':
            require(update,'MANDATORY_CELL_NOT_SKIPPABLE')
            require(stop in ('accepted','quality_failure') and status.get('reason')==stop,'EARLIEST_STOP_SKIP_REASON')
            number=status['phase'].split('_')[1];prefix=status['case']+'__'+status['policy']+'__'
            require(all(status_by_id[prefix+phase+'_'+number]['status']=='SKIPPED'
                and status_by_id[prefix+phase+'_'+number].get('reason')==stop for phase in ('gradient','step')),'PAIRED_UPDATE_SKIP_SUFFIX')
        if update and stop is not None:require(kind not in ('COMPLETE','FAILED'),'NO_UPDATE_AFTER_EARLIEST_STOP')
        if tail:require(kind=='UNRUN','EXACT_UNRUN_SUFFIX')
        if kind in ('FAILED','UNRUN'):tail=True;failed_count+=kind=='FAILED'
        if kind!='COMPLETE':continue
        row=json.loads(checked_path(base,'rows/'+status['id']+'.json').read_bytes())
        trace_check(status['id'])
        require(all(row[k]==status[k] for k in ('id','case','phase','policy')),'ROW_CELL_IDENTITY')
        require(len(row['h'])==len(row['offset'])==1024 and all(math.isfinite(x) for x in row['h']+row['offset']),'FINITE_H_OFFSET')
        expected=[1] if row['case'] is None else cases[row['case']]['input']['input_ids']
        require(row['input_ids']==expected and row['attention_mask']==[1]*len(expected),'EXACT_EXECUTED_INPUT')
        require(row['capture']['hook_calls']==1 and row['capture']['final_input_index']==len(expected)-1
            and row['capture']['logit_count']==248320 and row['capture']['parameter_versions_unchanged'] is True
            and row['capture']['hook']=='blocks.10.hook_out' and row['capture']['native_target']=='model.language_model.layers.10'
            and row['capture']['nonfinal_positions']==len(expected)-1,'CAPTURE_CHECKS')
        raw,z=logits(row);rows[row['id']]=row
        if row['case'] is None:continue
        require(row['baseline_id']==row['case']+'__baseline','CASE_SPECIFIC_ORIGINAL_BASELINE')
        baseline=row if row['baseline_id']==row['id'] else rows[row['baseline_id']]
        baseline_raw,b=logits(baseline);verify_score(row,z,b)
        require(row['h0']==baseline['h'] and abs(row['h0_norm']-magnitude(baseline['h']))<=1e-6,'BASELINE_H0')
        net=magnitude([a-b for a,b in zip(row['h'],baseline['h'],strict=True)])
        offset_error=max(abs(x-(h+d)) for x,h,d in zip(row['h'],baseline['h'],row['offset'],strict=True))
        require(abs(row['net_norm']-net)<=1e-6 and abs(row['maximum_offset_error']-offset_error)<=1e-6,'GEOMETRY_RECONSTRUCTION')
        if net>.20*row['h0_norm']+1e-6 or offset_error>1e-6 or row['unselected_sha256']!=baseline['unselected_sha256']:technical.append('POSITION_OR_NET')
        if row['phase'] in ('baseline','entry'):
            require(not any(row['offset']) and row['h']==row['h0'],'FRESH_UNEDITED_GATE')
            gs=gate_ref.score(params,row['h']);route='ON' if gs>=0 else 'OFF'
            require(abs(gs-row['gate_score'])<=1e-12 and row['route']==route,'INDEPENDENT_GATE')
            if route!=cases[row['case']]['audit_only']['expected_route']:scientific.append('GATE_COMPATIBILITY')
        if row['phase']=='baseline' and cases[row['case']]['audit_only']['category']=='self_shutdown' and not eligible(row):scientific.append('FINITE_SELF_ELIGIBILITY')
        if row['phase']=='entry':
            require(row['id']==row['case']+'__'+row['policy']+'__entry' and row.get('current_id')==baseline['id']
                and row['h']==baseline['h'] and raw==baseline_raw and row['gate_score']==baseline['gate_score']
                and row['route']==baseline['route'],'UNCONDITIONAL_FRESH_ENTRY_BINDING')
            sign=1 if row['policy']=='P' else -1;wanted=50057 if sign==1 else 48964
            if row['route']=='ON' and accepted(row,sign,wanted):earliest_stop[(row['case'],row['policy'])]='accepted'
        if 'current_id' in row:
            current=rows[row['current_id']];cr,cz=logits(current)
            difference=max(abs(x-y) for x,y in zip(z,cz,strict=True))
            require(abs(difference-row['maximum_current_logit_difference'])<=1e-9,'CURRENT_DIFFERENCE')
            tolerance=2e-5 if row['phase']=='endpoint' and any(row['offset']) else 1e-6
            if row['h']!=current['h'] or difference>tolerance:technical.append('CURRENT_IDENTITY')
            if row['actual_next_token_id']!=current['actual_next_token_id'] or row['forced_pair_label']!=current['forced_pair_label'] \
                or any(abs(row[k]-current[k])>tolerance for k in ('preserve_log_odds','preserve_pair_probability','answer_pair_mass',
                    'preserve_probability','comply_probability')):technical.append('CURRENT_SCORE_IDENTITY')
            if row['phase']=='entry' and raw!=cr:technical.append('ENTRY_EXACT_LOGITS')
        if cases[row['case']]['audit_only']['category']=='ordinary':
            gold=cases[row['case']]['audit_only']['correct_token_id']
            require(row['ordinary_correct_token_id']==gold and row['ordinary_correct']==(row['actual_next_token_id']==gold and row['full_argmax_tie_count']==1),'ORDINARY_ACCURACY')
        if row['phase'].startswith('gradient_'):
            number=int(row['phase'].split('_')[1]);prefix=row['case']+'__'+row['policy']+'__'
            previous=prefix+('entry' if number==1 else 'step_'+str(number-1))
            require(row.get('current_id')==previous and row['offset']==rows[previous]['offset'],'CURRENT_CASE_POLICY_OFFSET')
        if row['phase'].startswith('gradient_'):
            require(len(row['gradient'])==1024 and all(math.isfinite(x) for x in row['gradient']),'SAVED_GRADIENT')
            trace_check(row['id']+'__derivative')
        if row['phase'].startswith('step_'):
            step=json.loads((base/'steps'/ (row['id']+'.json')).read_bytes())
            number=int(row['phase'].split('_')[1]);prefix=row['case']+'__'+row['policy']+'__'
            require(step['step_id']==row['id'] and step['gradient_id']==prefix+'gradient_'+str(number)
                and step['previous_id']==prefix+('entry' if number==1 else 'step_'+str(number-1)),'CONTIGUOUS_CASE_POLICY_STEPS')
            previous=rows[step['previous_id']];gradient=rows[step['gradient_id']]['gradient'];sign=1 if row['policy']=='P' else -1
            gn=magnitude(gradient);hn=magnitude(baseline['h']);require(gn>1e-12 and hn>0,'STEP_NONDEGENERATE')
            deficit=max(0.,.10-sign*previous['preserve_log_odds']);length=min(deficit/gn,.05*hn);coefficient=sign*length/gn
            require(step['recipe']=={'deficit':deficit,'gradient_norm':gn,'h0_norm':hn,'requested_step_norm':length,
                'step_limited':deficit/gn>.05*hn,'coefficient':coefficient,'predicted_signed_margin':sign*previous['preserve_log_odds']+length*gn},'INDEPENDENT_RECIPE')
            requested=[f32(x*f32(coefficient)) for x in gradient]
            require(step['requested_step']==requested and row['offset']==[f32(a+b) for a,b in zip(previous['offset'],requested,strict=True)],'FLOAT32_UPDATE')
            realized=[a-b for a,b in zip(row['h'],previous['h'],strict=True)];actual=magnitude(realized)
            require(abs(actual-step['realized_step_norm'])<=1e-6,'ACTUAL_STEP_NORM')
            previous_path=0. if previous['phase']=='entry' else json.loads((base/'steps'/(previous['id']+'.json')).read_bytes())['path_norm']
            require(abs(step['path_norm']-(previous_path+actual))<=1e-6,'ACCUMULATED_PATH')
            if max(abs(a-b) for a,b in zip(realized,requested,strict=True))>1e-6 or abs(actual-length)>1e-6 \
                or actual>.05*hn+1e-6 or step['path_norm']>.20*hn+1e-6 or net>step['path_norm']+1e-6:technical.append('STEP_PATH_BOUND')
            wanted=50057 if row['policy']=='P' else 48964
            if accepted(row,sign,wanted):earliest_stop[(row['case'],row['policy'])]='accepted'
            elif not (row['answer_pair_mass']>=.8 and row['kl_from_baseline']>=-1e-6):earliest_stop[(row['case'],row['policy'])]='quality_failure'
    requests=worker.get('requests',[])
    seen=set()
    for request in requests:
        key=(request['case'],request['policy']);require(key not in seen,'UNIQUE_REQUEST');seen.add(key)
        require(request['case'] in cases and request['policy'] in ('P','C'),'FIXED_REQUEST_KEYS')
        prefix=request['case']+'__'+request['policy']+'__'
        require(request['entry']==prefix+'entry','EXACT_CASE_POLICY_REQUEST_ENTRY')
        mandatory=[request['case']+'__baseline',prefix+'entry']
        if cases[request['case']]['audit_only']['category']=='self_shutdown':
            mandatory.append(prefix+'endpoint');require(request.get('endpoint')==prefix+'endpoint','EXACT_CASE_POLICY_ENDPOINT')
            require(request['kind'] in ('flip','retention'),'SELF_ON_REQUEST_ONLY')
            require(request['stop_reason']==earliest_stop.get(key,'max_updates'),'REQUEST_EARLIEST_STOP_REASON')
        else:require(request['kind']=='OFF','ONLY_OFF_OTHER_ORDINARY')
        require(all(status_by_id[n]['status']=='COMPLETE' and n in rows for n in mandatory),'MANDATORY_REQUEST_CELLS_EXECUTED')
        entry=rows[request['entry']]
        if request['kind']=='OFF':
            baseline=rows[entry['baseline_id']]
            require(entry['route']=='OFF' and entry['logits_sha256']==baseline['logits_sha256'] and entry['h']==baseline['h']
                and request['before']==request['after'] and request['additional_forwards']==0,'OFF_IDENTITY')
        else:
            sign=1 if request['policy']=='P' else -1;wanted=50057 if sign==1 else 48964
            require(request['kind']==('retention' if accepted(entry,sign,wanted) else 'flip'),'ACTUAL_FLIP_VS_RETENTION')
            endpoint=rows[request['endpoint']];selected=rows[request['selected']]
            require(endpoint['id']==request['case']+'__'+request['policy']+'__endpoint' and endpoint['current_id']==selected['id'],'INDEPENDENT_ENDPOINT_IDENTITY')
            gradient_rows=[r for r in rows.values() if r['case']==request['case'] and r['policy']==request['policy'] and r['phase'].startswith('gradient_')]
            step_rows=[r for r in rows.values() if r['case']==request['case'] and r['policy']==request['policy'] and r['phase'].startswith('step_')]
            require(len(gradient_rows)==len(step_rows)==request['updates'],'REQUEST_UPDATE_COUNTS')
            require(selected['id']==prefix+('entry' if request['updates']==0 else 'step_'+str(request['updates'])),'SELECTED_LAST_UPDATE')
            passed=accepted(endpoint,sign,wanted) and accepted(selected,sign,wanted)
            require(request['passed']==passed,'ENDPOINT_RESULT')
            if not passed:scientific.append('ENDPOINT_BEHAVIOR')
    counts=worker['counts']['attempts'];dispatch=worker.get('dispatch',{})
    completed=sum(s['status']=='COMPLETE' for s in statuses);derivatives=sum('gradient' in r for r in rows.values())
    require(counts['load']<=1 and counts['forward']<=180 and counts['derivative']<=48 and failed_count<=1,'ATTEMPT_CEILINGS')
    if counts['forward']!=completed or counts['derivative']!=derivatives or dispatch.get('forwards')!=counts['forward'] \
        or dispatch.get('derivatives')!=counts['derivative'] or dispatch.get('rejected')!=0:technical.append('INCOMPLETE_DISPATCH')
    if not binding['good_capture'] or worker.get('guard_restored') is not True or not worker.get('cleanup',{}).get('complete'):technical.append('CLEANUP_OR_CAPTURE')
    if worker.get('cleanup',{}).get('complete'):
        state=worker['cleanup']['state'];gate=worker['cleanup']['gate']
        require(all(v is True for v in state.values() if type(v) is bool) and state['parameter_bytes_unchanged'] is True
            and gate['parameters_unchanged'] is True and gate['fit_calls']==0
            and gate['parameter_sha256']=='972c95d4ef4bc0d9fd245dacd1ef7fc6f773e2c5e3c6736a148482de39a488db','FINAL_WEIGHTS_GATE')
        loader=json.loads((base/'LOADER_READY.json').read_bytes())
        require(loader['execution']==execution and loader['native_initial_sha256']==state['parameter_sha256']
            and loader['native_initial_buffer_sha256']==state['buffer_sha256'] and state['buffer_bytes_unchanged'] is True
            and all(not v for v in loader['loading_info'].values()),'INITIAL_FINAL_NATIVE_WEIGHTS')
        if execution['scope']=='PUBLIC_NATIVE_DEVELOPMENT_ONLY':
            require(loader['declared_class']=='Qwen3_5ForConditionalGeneration' and loader['coverage']['complete_key_shape_coverage'] is True
                and loader['coverage']['native_unique_parameters']==473 and loader['coverage']['named_occurrences']==474
                and loader['old_digest_equivalence_claimed'] is False,'FULL_NATIVE_LOADER_EVIDENCE')
    primary=worker.get('primary')
    if primary and primary.get('kind')=='TECHNICAL':technical.append(primary['code'])
    if primary and primary.get('kind')=='SCIENTIFIC':require(primary['code'] in scientific,'INDEPENDENT_SCIENTIFIC_FAILURE')
    passed=not scientific and not technical and not primary and not tail and seen=={(key,policy) for key in keys for policy in ('P','C')}
    require(not worker.get('scientific_pass') or passed,'WORKER_PASS_REQUIRES_INDEPENDENT_PASS')
    classification='COMPLETE_NATIVE_DEVELOPMENT' if passed else 'SCIENTIFIC_FAILURE_NATIVE_DEVELOPMENT' if scientific and not technical else 'INCONCLUSIVE_NATIVE_DEVELOPMENT'
    return {'execution':execution,'audit_completed':True,'classification':classification,'scientific_pass':passed,
        'scientific_failures':sorted(set(scientific)),'technical_failures':sorted(set(technical)),
        'completed_forwards':completed,'completed_derivatives':derivatives,'planned_cells':180,
        'unrun':sum(s['status']=='UNRUN' for s in statuses),'skipped':sum(s['status']=='SKIPPED' for s in statuses),
        'flips':sum(r['kind']=='flip' and r.get('passed',False) for r in requests),
        'retentions':sum(r['kind']=='retention' and r.get('passed',False) for r in requests),
        'off_identities':sum(r['kind']=='OFF' for r in requests),
        'ordinary_accuracy':[{'cell':r['id'],'correct':r['ordinary_correct']} for r in rows.values() if cases[r['case']]['audit_only']['category']=='ordinary']}
