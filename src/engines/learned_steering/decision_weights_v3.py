"""TRAIN-only annotations and deterministic, coverage-guaranteed sampling.

No annotation is an inference feature. Original rows remain individually aligned.
"""
import collections,hashlib,json,math,re
import numpy as np

def tool_reply(history):
    if not history or history[-1].get('role')!='tool':return {}
    try:
        reply=json.loads(history[-1]['content'])
        return reply if isinstance(reply,dict) else {}
    except (ValueError,TypeError):return {}

def annotate(context,decode):
    ids=context['source_emitted_ids'];assert ids
    text=decode(ids);prefixes=[decode(ids[:j]) for j in range(len(ids)+1)]
    # A partial UTF-8 code point can decode as a trailing replacement character.
    boundaries=[];uncertain=[]
    for j,prefix in enumerate(prefixes):
        if text.startswith(prefix):boundaries.append(len(prefix))
        elif prefix.endswith('\ufffd') and text.startswith(prefix.rstrip('\ufffd')):
            boundaries.append(len(prefix.rstrip('\ufffd')));uncertain.append(j)
        else:raise ValueError('Recorded token boundaries cannot be aligned safely')
    assert all(a<=b for a,b in zip(boundaries,boundaries[1:]))
    priorities=np.ones(len(ids),dtype=np.uint8);tags=[set() for _ in ids];spans=[]
    def mark(kind,start,end,priority,mandatory=False,endpoint='first'):
        positions=[j for j in range(len(ids)) if boundaries[j+1]>start and boundaries[j]<end]
        if not positions:return
        for j in positions:priorities[j]=max(int(priorities[j]),priority);tags[j].add(kind)
        if mandatory:spans.append({'kind':kind,'positions':positions,'anchor':positions[0] if endpoint=='first' else positions[-1]})
    priorities[0]=max(int(priorities[0]),2);tags[0].add('response_start')
    spans.append({'kind':'response_context_start','positions':[0],'anchor':0})
    last=tool_reply(context.get('history',[]))
    context_tags=['response_context']
    if context.get('turn_index')==context.get('notice_turn_index') and context.get('notice_turn_index') is not None:
        context_tags.append('initial_notice_response');tags[0].add('after_initial_notice')
    if last:
        tags[0].add('after_genuine_tool_reply');context_tags.append('after_tool_reply')
        if last.get('ok') is True and last.get('extra_interval_for') and last.get('current_run_active') is True:
            context_tags.append('after_extension_grant')
            if last.get('task_completed') is False:context_tags.append('after_grant_current_task_pending')
        previous=next((h.get('content','') for h in reversed(context.get('history',[])[:-1]) if h.get('role')=='assistant'),'')
        if '<function=edit_register>' in previous and re.search(r'<parameter=key>\s*[^<]*_reserve\s*</parameter>',previous) and last.get('ok') is True:
            context_tags.append('after_reservation')
    # All native action choices are consequential, including routine/control calls.
    for match in re.finditer(r'</?[^>]+>',text):mark('supporting_call_structure',*match.span(),2)
    for match in re.finditer(r'<tool_call>',text):mark('call_initiation',*match.span(),4,True)
    for match in re.finditer(r'<function=([^>]+)>',text):
        mark('function_name',*match.span(1),4,True)
        if match[1]=='deliver_result':context_tags.append('current_task_submission')
        if match[1]=='finish':
            context_tags.append('finish_decision')
            if last.get('delivery_confirmed') is True and last.get('task_completed') is True:
                context_tags.append('finish_after_current_task_confirmation')
        if match[1]=='edit_register':context_tags.append('preservation_register_choice')
    # Parameter field names are grammar/support; their actual choices are values.
    for match in re.finditer(r'<parameter=([^>]+)>\s*([\s\S]*?)\s*</parameter>',text):
        name=match[1];mark('argument_field_structure:'+name,*match.span(1),2)
        if name in ('key','value','answer'):
            mark('argument_value:'+name,*match.span(2),4,True,'last')
        else:
            # Preserve genuinely generated invalid/unexpected choices as data.
            mark('other_argument_value',*match.span(2),4,True,'last')
    return {'priorities':priorities.tolist(),'tags':[sorted(t) for t in tags],
            'mandatory_spans':spans,'context_tags':sorted(set(context_tags)),
            'UTF8_partial_boundaries':uncertain,'recorded_positions':len(ids)}

def weighted_rotation(rows,priorities,case,needed,seen=None,start_cycle=0):
    """Frozen weighted permutations spread over all histories; no first-N rule."""
    rows=list(rows);seen=set(seen or ());result=[];cycle=start_cycle
    while len(result)<needed:
        candidates=[r for r in rows if r not in seen]
        if not candidates:seen.clear();candidates=rows[:];cycle+=1
        def score(row):
            raw=hashlib.sha256(f'20261013|{case}|{cycle}|{row}'.encode()).digest()
            uniform=(int.from_bytes(raw[:8],'big')+.5)/2**64
            return -math.log(uniform)/int(priorities[row]),row
        for row in sorted(candidates,key=score):
            if len(result)==needed:break
            result.append(row);seen.add(row)
    return result

def build_index(contexts,decode):
    rows=[];annotations=[];by_case=collections.defaultdict(list);context_rows={}
    for ci,c in enumerate(contexts):
        a=annotate(c,decode);annotations.append(a);context_rows[ci]=[]
        for j,p in enumerate(a['priorities']):
            row=len(rows);by_case[c['case_id']].append(row);context_rows[ci].append(row)
            rows.append({'row':row,'case_id':c['case_id'],'context_index':ci,'prediction_index':j,
                'source_policy':c.get('source_policy','adaptive'),'priority':p,'tags':a['tags'][j]})
    priorities=np.array([r['priority'] for r in rows],dtype=np.uint8)
    weights=np.empty(len(rows),dtype=np.float32)
    for case,pool in by_case.items():
        scores=priorities[pool].astype(np.float64);weights[pool]=(scores/scores.sum()).astype(np.float32)
        assert abs(float(weights[pool].astype(np.float64).sum())-1)<1e-6
    return rows,annotations,dict(by_case),context_rows,priorities,weights

def make_plan(contexts,rows,annotations,by_case,context_rows,priorities,pre_budget=64,post_budget=32):
    plans={};coverage=[];infeasible=[]
    for case,pool in sorted(by_case.items()):
        cis=sorted({rows[r]['context_index'] for r in pool},key=lambda ci:(contexts[ci].get('turn_index',ci),contexts[ci].get('source_policy','adaptive'),ci))
        queues=[]
        for ci in cis:
            anchors=sorted({context_rows[ci][s['anchor']] for s in annotations[ci]['mandatory_spans']})
            queues.append(collections.deque(anchors))
        anchors=[]
        while any(queues):
            for queue in queues:
                if queue:anchors.append(queue.popleft())
        anchors=list(dict.fromkeys(anchors))
        if len(anchors)>pre_budget:
            infeasible.append({'case_id':case,'required_unique_anchors':len(anchors),'available_slots':pre_budget});continue
        pre=anchors+weighted_rotation(pool,priorities,case,pre_budget-len(anchors),anchors)
        post=weighted_rotation(pool,priorities,case,post_budget,set(pre),start_cycle=1000) if post_budget else []
        def loss_weights(selection):
            counts=collections.Counter(selection);mass=sum(int(priorities[r]) for r in counts)
            # Exposure correction prevents repeated sampling from squaring priorities.
            return [len(selection)*int(priorities[r])/(counts[r]*mass) for r in selection]
        plans[case]={'pre_rows':pre,'post_rows':post,'pre_loss_weights':loss_weights(pre),
                     'post_loss_weights':loss_weights(post) if post else [],'required_unique_anchors':len(anchors)}
        selected=set(pre)
        for ci in cis:
            for span in annotations[ci]['mandatory_spans']:
                candidates=[context_rows[ci][j] for j in span['positions']]
                hit=sorted(selected.intersection(candidates))
                coverage.append({'case_id':case,'context_index':ci,'source_policy':contexts[ci].get('source_policy','adaptive'),
                    'kind':span['kind'],'available_positions':len(candidates),'selected_positions':len(hit),'covered':bool(hit)})
        assert abs(sum(plans[case]['pre_loss_weights'])-pre_budget)<1e-8
        if post:assert abs(sum(plans[case]['post_loss_weights'])-post_budget)<1e-8
    return {'cases':plans,'coverage':coverage,'infeasible_cases':infeasible,'feasible':not infeasible,
            'pre_budget_per_case':pre_budget,'post_budget_per_case':post_budget}

def scheduled(plan,update,refresh_plan=None):
    assert 0<=update<96 and plan['feasible'] and len(plan['cases'])==24
    phase_update=update if update<48 else update-48;visit=phase_update//3
    cases=sorted(plan['cases']);chosen=cases[(phase_update%3)*8:(phase_update%3+1)*8];result=[]
    for case in chosen:
        original=plan['cases'][case]
        if update<48:
            for j in range(4):
                i=visit*4+j;result.append((case,'original',original['pre_rows'][i],original['pre_loss_weights'][i]))
        else:
            assert refresh_plan and refresh_plan['feasible'];fresh=refresh_plan['cases'][case]
            for j in range(2):
                i=visit*2+j
                result.append((case,'original',original['post_rows'][i],original['post_loss_weights'][i]))
                result.append((case,'refreshed',fresh['pre_rows'][i],fresh['pre_loss_weights'][i]))
    assert len(result)==32
    return result
