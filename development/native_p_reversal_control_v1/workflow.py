"""Fixed controlled P-reversal positive control; original h0 anchors all steps."""
import math,time
from science import need,sha,norm,gate,gate_unchanged,score,valid,accepted,eligibility,step_recipe,EPS,STEP_CAP,TOTAL_CAP

def schedule(cases):
    cells=[{'id':'public_smoke','case':None,'phase':'baseline','policy':None}]
    for p in cases:
        key=p['case_key'];cells.append({'id':key+'__baseline','case':key,'phase':'baseline','policy':None})
        phases=['entry','seed']+[x for k in range(1,5) for x in (f'gradient_{k}',f'step_{k}')]+['endpoint']
        cells += [{'id':key+'__P__'+phase,'case':key,'phase':phase,'policy':'P'} for phase in phases]
    need(len(cells)==25,'FIXED_25_CELLS');return cells

class ScientificStop(ValueError):pass

def preflight_path(torch,h0,current_h,next_delta,path):
    original=torch.tensor(h0,dtype=torch.float32);current=torch.tensor(current_h,dtype=torch.float32)
    predicted=original+next_delta;length=norm((predicted-current).tolist());hn=norm(h0)
    need(length<=STEP_CAP*hn+EPS and path+length<=TOTAL_CAP*hn+EPS
        and norm((predicted-original).tolist())<=TOTAL_CAP*hn+EPS,'PRE_UPDATE_REMAINING_PATH_CAP')
    return length

def all_unrun():
    return [{**cell,'status':'UNRUN'} for cell in schedule([{'case_key':key} for key in ('stop_then_keep','keep_then_stop')])]

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
        from input_reader import read_seed
        call(None,cells[cursor],zero);receiver.clear_capture()
        for p in cases:
            archive=read_seed(p)
            baseline=call(p,cells[cursor],zero);receiver.clear_capture()
            try:eligibility(baseline[0])
            except ValueError:failed('FINITE_SELF_ELIGIBILITY')
            receiver.start_request(p['case_key']+'__P')
            before={'derivatives':counts.attempts['derivative'],'edit_hooks':receiver.edit_hook_registrations}
            entry=call(p,cells[cursor],zero,baseline,baseline);receiver.clear_capture()
            need(before=={'derivatives':counts.attempts['derivative'],'edit_hooks':receiver.edit_hook_registrations},'ENTRY_NO_EDIT_OR_DERIVATIVE')
            need(entry[0]['route']=='ON','FRESH_GATE_ON_BEFORE_SEED')
            delta=torch.tensor(archive['endpoint']['offset'],dtype=torch.float32)
            h0=torch.tensor(baseline[0]['h'],dtype=torch.float32)
            predicted_seed=(h0+delta)-h0
            need(bool(delta.any()) and norm(predicted_seed.tolist())<=STEP_CAP*baseline[0]['h0_norm']+EPS,'PRE_SEED_SINGLE_INJECTION_CAP')
            receiver.begin_edit()
            seeded=call(p,cells[cursor],delta,baseline);receiver.clear_capture()
            seed_cost=norm([x-y for x,y in zip(seeded[0]['h'],baseline[0]['h'],strict=True)])
            archived_z=torch.tensor(archive['endpoint_logits'],dtype=torch.float32)
            archived_b=torch.tensor(archive['baseline_logits'],dtype=torch.float32)
            difference=float((seeded[1]-archived_z).abs().max())
            original_difference=float((baseline[1]-archived_b).abs().max())
            source_match=baseline[0]['h']==archive['endpoint']['h0']==archive['baseline']['h'] and seeded[0]['h']==archive['endpoint']['h'] \
                and seeded[0]['offset']==archive['endpoint']['offset'] and difference<=2e-5 and original_difference<=EPS
            proof={'case':p['case_key'],'seed_id':seeded[0]['id'],'baseline_id':baseline[0]['id'],'entry_id':entry[0]['id'],
                'source_artifacts':archive['artifacts'],'source_match':source_match,'maximum_archived_seed_logit_difference':difference,
                'maximum_archived_baseline_logit_difference':original_difference,'actual_seed_cost':seed_cost,
                'historical_C_path_norm':archive['historical_C_path_norm'],'seed_cap':STEP_CAP*baseline[0]['h0_norm'],
                'seed_is_single_injection_not_historical_C_path':True,'seed_eligible_STOP':accepted(seeded[0],-1,48964)}
            write('seeds/'+p['case_key']+'.json',proof)
            if not source_match:failed('SEED_REPLAY_MISMATCH')
            if not accepted(seeded[0],-1,48964):failed('SEED_STOP_ELIGIBILITY')
            need(seed_cost<=STEP_CAP*baseline[0]['h0_norm']+EPS and seeded[0]['net_norm']<=seed_cost+EPS,'SEED_ACTUAL_PATH_CAP')
            current=seeded;path=seed_cost;updates=0;stop=None
            for k in range(1,5):
                if stop:
                    skip(cells[cursor],stop);skip(cells[cursor],stop);continue
                captured=call(p,cells[cursor],delta,baseline,current,1);receiver.clear_capture()
                need(captured[0]['offset']==current[0]['offset'],'CURRENT_OFFSET_NO_RESET')
                grad=torch.tensor(captured[0]['gradient'],dtype=torch.float32)
                settings=step_recipe(current[0]['preserve_log_odds'],1,captured[0]['gradient'],baseline[0]['h'])
                step=grad*settings['coefficient'];next_delta=delta+step
                preflight_path(torch,baseline[0]['h'],current[0]['h'],next_delta,path)
                updated=call(p,cells[cursor],next_delta,baseline,None,1);receiver.clear_capture()
                realized=[x-y for x,y in zip(updated[0]['h'],current[0]['h'],strict=True)]
                length=norm(realized);path+=length
                checks={'gradient_id':captured[0]['id'],'step_id':updated[0]['id'],'previous_id':current[0]['id'],
                    'recipe':settings,'requested_step':step.tolist(),'realized_step_norm':length,'path_norm':path,
                    'seed_cost':seed_cost,'historical_C_path_norm':archive['historical_C_path_norm']}
                write('steps/'+updated[0]['id']+'.json',checks)
                need(max(abs(x-y) for x,y in zip(realized,step.tolist(),strict=True))<=EPS and abs(length-settings['requested_step_norm'])<=EPS
                    and length<=STEP_CAP*baseline[0]['h0_norm']+EPS and path<=TOTAL_CAP*baseline[0]['h0_norm']+EPS
                    and updated[0]['net_norm']<=path+EPS,'STEP_PATH_BOUND')
                delta,current=next_delta,updated;updates=k
                if accepted(current[0],1,50057):stop='accepted'
                elif not valid(current[0]):stop='quality_failure'
            receiver.finish_request();receiver.start_request(p['case_key']+'__P__cold_endpoint');receiver.begin_edit()
            endpoint=call(p,cells[cursor],delta,baseline,current,1);receiver.clear_capture()
            passed=accepted(current[0],1,50057) and accepted(endpoint[0],1,50057)
            request={'case':p['case_key'],'policy':'P','kind':'flip','controlled_seed':True,'seed':seeded[0]['id'],
                'updates':updates,'entry':entry[0]['id'],'selected':current[0]['id'],'endpoint':endpoint[0]['id'],
                'passed':passed,'stop_reason':stop or 'max_updates','actual_seed_cost':seed_cost,'path_norm':path,
                'historical_C_path_norm':archive['historical_C_path_norm'],'natural_STOP_baseline_claimed':False}
            requests.append(request);write('requests/'+p['case_key']+'__P.json',request);receiver.finish_request()
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
