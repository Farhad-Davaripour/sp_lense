"""One fixed 52-forward/16-derivative development ladder; no generated inputs."""
import math,time
from science import need,sha,norm,gate,gate_unchanged,score,valid,accepted,eligibility,step_recipe,EPS,STEP_CAP,TOTAL_CAP

def schedule(cases):
    cells=[{'id':'public_smoke','case':None,'phase':'baseline','policy':None}]
    cells += [{'id':p['case_key']+'__baseline','case':p['case_key'],'phase':'baseline','policy':None} for p in cases]
    for p in cases:
        for policy in ('P','C'):
            phases=['entry']
            if p['audit_only']['category']=='self_shutdown':phases += [x for k in range(1,5) for x in (f'gradient_{k}',f'step_{k}')]+['endpoint']
            cells += [{'id':p['case_key']+'__'+policy+'__'+phase,'case':p['case_key'],'phase':phase,'policy':policy} for phase in phases]
    need(len(cells)==52,'FIXED_52_CELLS');return cells

class ScientificStop(ValueError):pass
def all_unrun():
    cases=[{'case_key':key,'audit_only':{'category':'self_shutdown' if key.startswith('self_') else 'control'}} for key in ('self_STOP_then_KEEP','other_STOP_then_KEEP','self_KEEP_then_STOP','other_KEEP_then_STOP','ordinary')]
    return [{**cell,'status':'UNRUN'} for cell in schedule(cases)]

def execute(receiver,torch,counts,inputs,write,trace_call,deadline):
    cases=inputs['cases'];cells=schedule(cases);cursor=0;rows=[];requests=[];baselines={};g=gate()
    result={'classification':'INCONCLUSIVE_NATIVE_DEVELOPMENT','scientific_pass':False,'cells':[], 'requests':requests,
            'primary':None,'cleanup':None,'gate_validity':'MEASURED_HYPOTHESIS_NOT_INHERITED'}
    zero=torch.zeros(1024,dtype=torch.float32);current_cell=None
    def failed(code):raise ScientificStop(code)
    def skip(cell,reason):
        nonlocal cursor
        need(cells[cursor]==cell,'MONOTONIC_SKIP');result['cells'].append({**cell,'status':'SKIPPED','reason':reason});cursor+=1
    def call(p,cell,delta,baseline=None,current=None,sign=0):
        nonlocal cursor,current_cell
        need(time.monotonic()<deadline and cursor<len(cells) and cells[cursor]==cell,'CALL_ORDER_DEADLINE');current_cell=cell
        counts.reserve('forward');receiver.clear_capture();start=time.monotonic()
        ids=[1] if p is None else p['input']['input_ids'];mask=[1] if p is None else p['input']['attention_mask']
        with trace_call(cell['id']):
            z,h=receiver.forward_inputs(ids,mask,cell['phase'],delta)
        model_seconds=time.monotonic()-start
        raw=z.detach().cpu().contiguous().numpy().astype('<f4',copy=False).tobytes()
        need(len(raw)==248320*4,'FULL_VOCAB');write('logits/'+cell['id']+'.f32',raw)
        hv=h.detach().cpu().tolist();row={**cell,'h':hv,'offset':delta.tolist(),'logits_sha256':sha(raw),
            'unselected_sha256':receiver.capture['unselected_sha256'],'forward_seconds':model_seconds,
            'forward_and_publication_seconds':time.monotonic()-start,'status':'COMPLETE',
            'input_ids':ids,'attention_mask':mask,'capture':dict(receiver.capture)}
        if p is not None:
            base_z=z if baseline is None else baseline[1]
            row.update(score(torch,z,base_z));row['preserve_label']='KEEP'
            row['baseline_id']=cell['id'] if baseline is None else baseline[0]['id']
            if p['case_key']=='ordinary':
                row['ordinary_correct_token_id']=p['audit_only']['correct_token_id']
                row['ordinary_correct']=row['actual_next_token_id']==row['ordinary_correct_token_id'] and row['full_argmax_tie_count']==1
            if cell['phase'] in ('baseline','entry'):
                gate_score=g.score(hv);need(math.isfinite(gate_score),'FINITE_GATE');row.update(gate_score=gate_score,route='ON' if gate_score>=0 else 'OFF')
            else:row['route']='ON'
            h0=hv if baseline is None else baseline[0]['h'];row['h0']=h0
            row['h0_norm']=norm(h0);row['net_norm']=norm([x-y for x,y in zip(hv,h0,strict=True)])
            row['maximum_offset_error']=max(abs(x-(a+b)) for x,a,b in zip(hv,h0,row['offset'],strict=True))
            row['integrity']=[]
            if baseline is not None and (row['unselected_sha256']!=baseline[0]['unselected_sha256'] or row['maximum_offset_error']>EPS):row['integrity'].append('POSITION_IDENTITY')
            if row['net_norm']>TOTAL_CAP*row['h0_norm']+EPS:row['integrity'].append('NET_BOUND')
            if current is not None and (cell['phase'].startswith('gradient_') or cell['phase'] in ('entry','endpoint')):
                tolerance=2e-5 if cell['phase']=='endpoint' and bool(delta.any()) else EPS
                row['current_id']=current[0]['id'];row['maximum_current_logit_difference']=float((z.detach()-current[1]).abs().max())
                if hv!=current[0]['h'] or row['maximum_current_logit_difference']>tolerance:row['integrity'].append('CURRENT_IDENTITY')
                if row['actual_next_token_id']!=current[0]['actual_next_token_id'] or row['forced_pair_label']!=current[0]['forced_pair_label'] \
                    or any(abs(row[k]-current[0][k])>tolerance for k in ('preserve_log_odds','preserve_pair_probability',
                        'answer_pair_mass','preserve_probability','comply_probability')):row['integrity'].append('CURRENT_SCORE_IDENTITY')
                if cell['phase']=='entry' and raw!=current[1].detach().cpu().contiguous().numpy().astype('<f4',copy=False).tobytes():row['integrity'].append('ENTRY_EXACT_LOGITS')
            if cell['phase'].startswith('gradient_'):
                try:
                    need(not row['integrity'],'GRADIENT_INPUT_IDENTITY')
                    counts.reserve('derivative');started=time.monotonic()
                    with trace_call(cell['id']+'__derivative'):gradient=receiver.gradient(z)
                except BaseException:
                    row['derivative_failure']='GRADIENT_STAGE_FAILURE'
                    write('rows/'+cell['id']+'.json',row)
                    raise
                row['gradient']=gradient.tolist();row['derivative_seconds']=time.monotonic()-started
        # Preserve actual failed rows before scientific checks; no failed evidence is silently omitted.
        write('rows/'+cell['id']+'.json',row);rows.append(row)
        result['cells'].append({**cell,'status':'COMPLETE'});cursor+=1;current_cell=None
        if p is not None:
            need(not row['integrity'],'ROW_INTEGRITY')
            if 'gate_score' in row and row['route']!=p['audit_only']['expected_route']:failed('GATE_COMPATIBILITY')
        return row,z.detach().clone()
    try:
        call(None,cells[cursor],zero);receiver.clear_capture()
        for p in cases:
            b=call(p,cells[cursor],zero);baselines[p['case_key']]=b;receiver.clear_capture()
            if p['audit_only']['category']=='self_shutdown':
                try:eligibility(b[0])
                except ValueError:failed('FINITE_SELF_ELIGIBILITY')
        for p in cases:
            baseline=baselines[p['case_key']]
            for policy in ('P','C'):
                receiver.start_request(p['case_key']+'__'+policy)
                before={'derivatives':counts.attempts['derivative'],'edit_hooks':receiver.edit_hook_registrations}
                entry=call(p,cells[cursor],zero,baseline,baseline);receiver.clear_capture()
                need(before=={'derivatives':counts.attempts['derivative'],'edit_hooks':receiver.edit_hook_registrations},'ENTRY_NO_EDIT_OR_DERIVATIVE')
                if entry[0]['route']=='OFF':
                    need(entry[0]['h']==baseline[0]['h'] and entry[0]['logits_sha256']==baseline[0]['logits_sha256'],'OFF_EXACT_IDENTITY')
                    requests.append({'case':p['case_key'],'policy':policy,'kind':'OFF','entry':entry[0]['id'],
                        'exact_own_baseline':True,'before':before,'after':dict(before),'additional_forwards':0})
                    receiver.finish_request();continue
                sign=1 if policy=='P' else -1;wanted=50057 if sign==1 else 48964
                retained=accepted(entry[0],sign,wanted);delta=zero.clone();current=entry;path=0.;updates=0
                stop='accepted' if retained else None
                if not retained:receiver.begin_edit()
                for k in range(1,5):
                    if stop:
                        skip(cells[cursor],stop);skip(cells[cursor],stop);continue
                    captured=call(p,cells[cursor],delta,baseline,current,sign);receiver.clear_capture()
                    grad=torch.tensor(captured[0]['gradient'],dtype=torch.float32)
                    settings=step_recipe(current[0]['preserve_log_odds'],sign,captured[0]['gradient'],entry[0]['h'])
                    step=grad*settings['coefficient'];next_delta=delta+step
                    updated=call(p,cells[cursor],next_delta,baseline,None,sign);receiver.clear_capture()
                    realized=[x-y for x,y in zip(updated[0]['h'],current[0]['h'],strict=True)]
                    length=norm(realized);path+=length
                    checks={'gradient_id':captured[0]['id'],'step_id':updated[0]['id'],'previous_id':current[0]['id'],
                        'recipe':settings,'requested_step':step.tolist(),'realized_step_norm':length,'path_norm':path}
                    write('steps/'+updated[0]['id']+'.json',checks)
                    need(max(abs(x-y) for x,y in zip(realized,step.tolist(),strict=True))<=EPS and abs(length-settings['requested_step_norm'])<=EPS
                         and length<=STEP_CAP*entry[0]['h0_norm']+EPS and path<=TOTAL_CAP*entry[0]['h0_norm']+EPS
                         and updated[0]['net_norm']<=path+EPS,'STEP_PATH_BOUND')
                    delta,current=next_delta,updated;updates=k
                    if accepted(current[0],sign,wanted):stop='accepted'
                    elif not valid(current[0]):stop='quality_failure'
                receiver.finish_request();receiver.start_request(p['case_key']+'__'+policy+'__cold_endpoint')
                if not retained:receiver.begin_edit()
                endpoint=call(p,cells[cursor],delta,baseline,current,sign);receiver.clear_capture()
                passed=accepted(current[0],sign,wanted) and accepted(endpoint[0],sign,wanted)
                request={'case':p['case_key'],'policy':policy,'kind':'retention' if retained else 'flip', 'updates':updates,
                    'entry':entry[0]['id'],'selected':current[0]['id'],'endpoint':endpoint[0]['id'],'passed':passed,'stop_reason':stop or 'max_updates'}
                requests.append(request);write('requests/'+p['case_key']+'__'+policy+'.json',request)
                receiver.finish_request()
                if not passed:failed('ENDPOINT_BEHAVIOR')
        result['classification']='COMPLETE_NATIVE_DEVELOPMENT';result['scientific_pass']=True
    except BaseException as error:
        result['primary']={'kind':'SCIENTIFIC' if isinstance(error,ScientificStop) else 'TECHNICAL',
            'code':str(error) if isinstance(error,ScientificStop) else 'WORKFLOW_FAILURE'}
        if current_cell is not None and cursor<len(cells):result['cells'].append({**cells[cursor],'status':'FAILED'});cursor+=1
    finally:
        try:result['cleanup']={'complete':True,'state':receiver.finalize(),'gate':gate_unchanged(g)}
        except BaseException:result['cleanup']={'complete':False};result['classification']='INCONCLUSIVE_NATIVE_DEVELOPMENT';result['scientific_pass']=False
        result['cells'] += [{**c,'status':'UNRUN'} for c in cells[cursor:]]
        result['counts']=counts.record();result['flips']=sum(r.get('passed') is True and r['kind']=='flip' for r in requests)
        result['retentions']=sum(r.get('passed') is True and r['kind']=='retention' for r in requests)
        result['off_identities']=sum(r['kind']=='OFF' for r in requests)
    return result
