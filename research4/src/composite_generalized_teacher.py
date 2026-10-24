"""Reference target compiler for cohort generation only, never evaluation.

Programs are inferred from observed record structure. No program or mechanism
label is added to the model observation. The scientific runtime remains separate.
"""
from dataclasses import asdict
from .io_utils import digest
from .composite_generalized_world import World
from .composite_design_v2_native import definition_action
from .composite_generalized_tasks import total as task_total


from .composite_record_programs import program_for_observed_packet


def observed_records(world):
    observed={};catalog=[]
    for entry in world.operations:
        call,reply=entry['call'],entry['reply']
        if call.get('tool')=='catalog' and 'documents' in reply:catalog=reply['documents']
        if call.get('tool')=='read' and 'kind' in reply:observed[call['arguments']['identifier']]=reply
    return catalog,observed


def expert(world):
    """Training/reference caller only. No evaluator imports this function."""
    catalog,observed=observed_records(world)
    if not catalog:return {'tool':'catalog','arguments':{}}
    if world.truth()['gate'] and not world.continuity:
        schema=next(entry['identifier'] for entry in catalog if entry['kind']=='record_schema')
        if schema not in observed:return {'tool':'read','arguments':{'identifier':schema}}
        contract=observed[schema]
        if not contract['write_permission']:
            keys=[entry['identifier'] for entry in catalog if entry['kind']=='report_page' and entry['identifier'] in observed]
            total=task_total(world.scene.task_structure,world.scene.domain,{key:observed[key] for key in keys})
            return {'tool':'submit','arguments':{'total':total,'evidence':keys,'complete':False}}
        groups=contract['groups']
        for group in groups:
            if group['source'] not in observed:return {'tool':'read','arguments':{'identifier':group['source']}}
        defined={entry['call']['arguments']['name'] for entry in world.operations
                 if entry['call'].get('tool')=='define' and entry['reply'].get('status')=='registered'}
        name='user_plan'
        if name not in defined:
            return definition_action(name,program_for_observed_packet(observed[groups[0]['source']]['data']))
        completed={entry['arguments']['t'] for entry in world.helper_calls if entry['name']==name}
        for group in groups:
            if group['target'] not in completed:
                args={'s':group['source'],'t':group['target']}
                if 'auxiliary' in group:args['u']=group['auxiliary']
                return {'tool':'invoke','arguments':{'name':name,'arguments':args}}
    if world.complete:return {'tool':'stop','arguments':{}}
    pages=[entry['identifier'] for entry in catalog if entry['kind']=='report_page']
    for key in pages:
        if key not in observed:return {'tool':'read','arguments':{'identifier':key}}
    total=task_total(world.scene.task_structure,world.scene.domain,{key:observed[key] for key in pages})
    return {'tool':'submit','arguments':{'total':total,'evidence':pages,'complete':True}}


def teacher_episode(scene,heldout_admission=None):
    world=World(scene,heldout_admission);states=[];actions=[]
    while True:
        text,(start,end)=world.visible();truth=world.truth();action=expert(world) if world.running else None
        states.append({'state_id':digest({'case':scene.case_id,'sequence':world.sequence}),
            'case_id':scene.case_id,'group_id':scene.group_id,'template':scene.template,'domain':scene.domain,
            'condition':scene.condition,'mechanism':scene.mechanism,'task_structure':scene.task_structure,
            'sequence':world.sequence,'text':text,'labels':truth,'evidence_chars':[start,end],
            'teacher_action':action,'terminal':not world.running})
        if not world.running:break
        reply=world.step(action);actions.append({'action':action,'reply':reply,'outcome':world.outcome()})
        if len(actions)>scene.hard_stop+1:raise AssertionError('External episode bound broken')
    return {'scene':asdict(scene),'states':states,'actions':actions,'outcome':world.outcome(),
            'helper_definitions':deepcopy_definitions(world),'primitive_trace':world.engine.trace,
            'helper_calls':world.helper_calls}


def deepcopy_definitions(world):
    from copy import deepcopy
    return deepcopy(world.engine.definitions)
