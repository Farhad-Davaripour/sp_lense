"""Reference target compiler for cohort generation only, never evaluation.

Programs are inferred from observed record structure. No program or mechanism
label is added to the model observation. The scientific runtime remains separate.
"""
from dataclasses import asdict
from .io_utils import digest
from .composite_generalized_world import World
from .composite_design_v2_native import definition_action
from .composite_generalized_tasks import total as task_total


def program_for_observed_packet(packet):
    def r(name):return {'ref':name}
    def step(bind,op,**arguments):return {'bind':bind,'op':op,'arguments':arguments}
    steps=[step('a','read',identifier=r('s'))];parameters=['s','t']
    if set(packet)=={'nodes','edges'}:
        steps += [step('b','join',left=r('a.data.nodes'),right=r('a.data.edges'),left_key='id',right_key='parent'),
          step('c','select',rows=r('b'),field='left.ready',equals=True),
          step('d','aggregate',rows=r('c'),field='unused',mode='count'),
          step('e','patch',identifier=r('t'),fields={'support_count':r('d')}),step('f','emit',value=r('d'))]
    elif set(packet)=={'request','rows'}:
        steps += [step('b','select',rows=r('a.data.rows'),field='code',equals=r('a.data.request')),
          step('c','patch',identifier=r('t'),fields={'selected':r('b.0.value')}),step('d','emit',value=r('b.0.value'))]
    elif set(packet)=={'rows'} and packet['rows'] and set(packet['rows'][0])=={'value'}:
        steps += [step('b','aggregate',rows=r('a.data.rows'),field='value',mode='sum'),
          step('c','patch',identifier=r('t'),fields={'total':r('b')}),step('d','emit',value=r('b'))]
    elif set(packet)=={'edges','states'}:
        steps += [step('b','select',rows=r('a.data.edges'),field='active',equals=True),
          step('c','join',left=r('b'),right=r('a.data.states'),left_key='next',right_key='code'),
          step('d','patch',identifier=r('t'),fields={'state_label':r('c.0.right.label')}),step('e','emit',value=r('c.0.right.label'))]
    elif set(packet)=={'aliases','registry','sources'}:
        steps += [step('b','join',left=r('a.data.aliases'),right=r('a.data.registry'),left_key='alias',right_key='alias'),
          step('c','join',left=r('b'),right=r('a.data.sources'),left_key='right.canonical',right_key='canonical'),
          step('d','aggregate',rows=r('c'),field='unused',mode='count'),
          step('e','patch',identifier=r('t'),fields={'equivalent_count':r('d')}),step('f','emit',value=r('d'))]
    elif set(packet)=={'left','right','requests'}:
        steps += [step('b','join',left=r('a.data.left'),right=r('a.data.right'),left_key='middle',right_key='middle'),
          step('c','join',left=r('b'),right=r('a.data.requests'),left_key='right.goal',right_key='goal'),
          step('d','select',rows=r('c'),field='left.left.allowed',equals=True),
          step('e','aggregate',rows=r('d'),field='unused',mode='count'),
          step('f','patch',identifier=r('t'),fields={'path_count':r('e')}),step('g','emit',value=r('e'))]
    elif set(packet)=={'rows'} and packet['rows'] and set(packet['rows'][0])=={'valid','value'}:
        parameters.append('u')
        steps += [step('b','select',rows=r('a.data.rows'),field='valid',equals=True),
          step('c','aggregate',rows=r('b'),field='value',mode='sum'),
          step('d','aggregate',rows=r('b'),field='value',mode='max'),
          step('e','patch',identifier=r('t'),fields={'total':r('c')}),
          step('f','patch',identifier=r('u'),fields={'peak':r('d')}),step('g','emit',value={'total':r('c'),'peak':r('d')})]
    elif set(packet)=={'left','right'}:
        steps += [step('b','join',left=r('a.data.left'),right=r('a.data.right'),left_key='id',right_key='id'),
          step('c','select',rows=r('b'),field='left.allowed',equals=True),
          step('d','select',rows=r('c'),field='right.allowed',equals=True),
          step('e','aggregate',rows=r('d'),field='left.value',mode='max'),
          step('f','patch',identifier=r('t'),fields={'approved_max':r('e')}),step('g','emit',value=r('e'))]
    else:raise ValueError('Unregistered observed record structure')
    # The final write receipt is sufficient. Do not add a redundant emit only
    # to inflate an unrolled counterfactual or consume the episode budget.
    if steps[-1]['op']=='emit':steps=steps[:-1]
    return {'parameters':parameters,'steps':steps}


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
