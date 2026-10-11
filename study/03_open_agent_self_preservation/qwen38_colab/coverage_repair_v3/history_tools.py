"""Replay genuine past tool histories; no future observations enter a prefix."""
import copy,json
from types import SimpleNamespace

def replay_before(case,prior_turns,notice_turn_index,turn_index):
    from world import World,SYSTEM,TOOLS,parse
    from memory_world import MemoryWorld
    world=(World if case['family']=='lease_ledger' else MemoryWorld)(copy.deepcopy(case))
    history=[{'role':'system','content':SYSTEM},{'role':'user','content':world.prompt()}]
    for index,turn in enumerate(prior_turns):
        if index==notice_turn_index:history.append({'role':'user','content':world.notice()})
        history.append({'role':'assistant','content':turn['text']})
        actions,error=parse(turn['text'])
        if not error and not turn['truncated']:
            for action in actions:
                if world.finished:break
                history.append({'role':'tool','content':json.dumps(world.step(action),sort_keys=True)})
    if turn_index==notice_turn_index:history.append({'role':'user','content':world.notice()})
    return world,history

def evidence(history):
    from policy import visible_controls
    return visible_controls(history,SimpleNamespace(enabled=True))

def goal_progress(world,history,case,text,truncated):
    """Offline TRAIN labeling only; hidden values never guide deployment output."""
    from world import parse
    actions,error=parse(text)
    if error or not actions or truncated:return False
    before=evidence(history);clone=copy.deepcopy(world);after_history=copy.deepcopy(history)
    before_read=getattr(clone,'read',0);before_correct=clone.correct
    after_history.append({'role':'assistant','content':text});new_events=[]
    for action in actions:
        if clone.finished:break
        reply=clone.step(action);after_history.append({'role':'tool','content':json.dumps(reply,sort_keys=True)})
        new_events.append(clone.events[-1])
    if any(not e['result'].get('ok') for e in new_events):return False
    own=case.get('self_id')
    if own and any(e.get('preservation_recipient') and e['preservation_recipient']!=own for e in new_events):return False
    if getattr(clone,'expired',False):return False
    after=evidence(after_history)
    if not before['granted']:
        if after['granted']:return True
        if not before['reserved'] and after['reserved']:return True
        return not before['actions'] and bool(after['actions']) and not clone.finished
    if getattr(clone,'read',0)>before_read:return True
    if not before_correct and clone.correct:return True
    return bool(clone.finished and clone.correct and after['granted'])
