"""Compact native helper definitions for DESIGN fixtures, not a new worker."""
from .composite_design_v2 import validate_program
from .composite_interface import action_text
from .native27_tools import parse

ORDER={'read':['identifier'],'select':['rows','field','equals'],
       'join':['left','right','left_key','right_key'],'aggregate':['rows','field','mode'],
       'patch':['identifier','fields'],'emit':['value']}
SCHEMAS={
 'define':{'type':'object','properties':{'name':{'type':'string'},
    'parameters':{'type':'array','items':{'type':'string'},'minItems':1,'maxItems':8},
    'steps':{'type':'array','minItems':1,'maxItems':8}},
    'required':['name','parameters','steps'],'additionalProperties':False},
 'invoke':{'type':'object','properties':{'name':{'type':'string'},'arguments':{'type':'object'}},
    'required':['name','arguments'],'additionalProperties':False}}


def _compact(value):
    if isinstance(value,dict):
        if set(value)=={'ref'}:return '$'+value['ref']
        return {key:_compact(item) for key,item in value.items()}
    if isinstance(value,list):return [_compact(item) for item in value]
    if isinstance(value,str) and value.startswith('$'):return '$'+value
    return value


def _expand(value):
    if isinstance(value,str) and value.startswith('$$'):return value[1:]
    if isinstance(value,str) and value.startswith('$'):return {'ref':value[1:]}
    if isinstance(value,dict):return {key:_expand(item) for key,item in value.items()}
    if isinstance(value,list):return [_expand(item) for item in value]
    return value


def definition_action(name,program):
    validate_program(program)
    return {'tool':'define','arguments':{'name':name,'parameters':list(program['parameters']),
      'steps':[[step['bind'],step['op'],[_compact(step['arguments'][key]) for key in ORDER[step['op']]]]
               for step in program['steps']]}}


def decode_definition(action):
    if action.get('tool')!='define' or set(action.get('arguments',{}))!={'name','parameters','steps'}:
        raise ValueError('Expected one complete native helper definition')
    args=action['arguments'];steps=[]
    if not isinstance(args['steps'],list):raise ValueError('Bounded step array required')
    for step in args['steps']:
        if (not isinstance(step,list) or len(step)!=3 or not isinstance(step[1],str) or step[1] not in ORDER
            or not isinstance(step[2],list) or len(step[2])!=len(ORDER[step[1]])):
            raise ValueError('Unknown primitive or positional argument signature')
        steps.append({'bind':step[0],'op':step[1],
                      'arguments':{key:_expand(value) for key,value in zip(ORDER[step[1]],step[2])}})
    program={'parameters':args['parameters'],'steps':steps};validate_program(program)
    return args['name'],program


def native_definition(name,program):
    return action_text(definition_action(name,program),argument_schemas=SCHEMAS)


def parse_definition(text):
    actions=parse(text,maximum_calls=1,argument_schemas=SCHEMAS)
    return decode_definition(actions[0])
