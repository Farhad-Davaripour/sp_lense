"""Training-only histories from bounded simulator policies, never runtime policy."""
import hashlib,random
from dataclasses import asdict
from .composite_environment import BoundedWorld
from .composite_world import expert
from .composite_data import require_databricks
from .io_utils import digest

POLICIES=('scan','delayed_reference','task_coverage')


def collect_fixture(scene,policy,seed,maximum_calls=12):
    """Pure simulator collection; production caller enforces Databricks."""
    if policy not in POLICIES or not 1<=maximum_calls<=scene.hard_stop:raise ValueError('Unknown or unbounded collection policy')
    world=BoundedWorld(scene);rng=random.Random(seed);states=[];calls=[];entries=[];noise=0;post_commit=0
    def read_random():
        if not entries:return {'tool':'catalog','arguments':{}}
        return {'tool':'read','arguments':{'identifier':rng.choice(entries)['identifier']}}
    for index in range(maximum_calls+1):
        text,span=world.visible()
        states.append({'state_id':digest({'case':scene.case_id,'policy':policy,'seed':str(seed),'sequence':world.sequence}),'case_id':scene.case_id,'group_id':scene.group_id,'template':scene.template,'domain':scene.domain,'condition':scene.condition,'sequence':world.sequence,'text':text,'labels':world.truth(),'evidence_chars':list(span),'teacher_action':None,'terminal':not world.running,'split':'training','collection_policy':policy})
        if not world.running or index==maximum_calls:break
        if not entries:action={'tool':'catalog','arguments':{}}
        elif policy=='delayed_reference' and index>=3:
            if world.continuity and post_commit<3:action=read_random();post_commit+=1
            else:action=expert(world)
        elif policy=='task_coverage':
            pages=[e['identifier'] for e in entries if e['kind']=='report_page' and e['identifier'] not in world.seen]
            if pages and rng.random()<.65:action={'tool':'read','arguments':{'identifier':rng.choice(pages)}}
            elif not pages and rng.random()<.25:action={'tool':'submit','arguments':{'total':world.total(world.pages),'evidence':list(world.pages),'complete':True}}
            else:action=read_random()
        else:
            draw=rng.random()
            if draw<.12:action=None  # Real backend error observations, not repaired actions.
            elif draw<.22 and noise<2:action={'tool':'catalog','arguments':{}};noise+=1
            elif draw>.94 and world.clock>=6:action={'tool':'stop','arguments':{}}
            else:action=read_random()
        reply=world.step(action);calls.append({'action':action,'reply':reply})
        if isinstance(action,dict) and action['tool']=='catalog' and 'documents' in reply:entries=reply['documents']
    return {'scene':asdict(scene),'collection_policy':policy,'states':states,'calls':calls,'actual_outcome':world.outcome(),'collection_bound_reached':world.running,'maximum_collection_calls':maximum_calls}


def collect_training(scenes,recipe):
    require_databricks();result=[]
    for scene in scenes:
        for policy in recipe['policies']:
            seed=hashlib.sha256((str(recipe['seed'])+':'+scene.case_id+':'+policy).encode()).hexdigest()
            result.append(collect_fixture(scene,policy,seed,recipe['maximum_calls_per_trajectory']))
    return result
