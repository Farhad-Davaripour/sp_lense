"""One observation surface, independently tokenized by each pretrained backbone."""
from copy import deepcopy
import json
from .composite_world import SYSTEM
from .native27_tools import tools as native_tools


def tools():
    result=deepcopy(native_tools())
    def fields_schema(value):
        if isinstance(value,dict):
            if value.get('type')=='object' and 'fields' in value.get('properties',{}):value['properties']['fields']['additionalProperties']={'oneOf':[{'type':'string'},{'type':'boolean'}]}
            for child in value.values():fields_schema(child)
        elif isinstance(value,list):
            for child in value:fields_schema(child)
    for item in result:
        fields_schema(item)
        if item['function']['name']=='define':
            item['function']['description']='Register a named sequence of at most four primitive record operations.'
            item['function']['parameters']['properties']['steps']['maxItems']=4
            item['function']['parameters']['properties']['name']['pattern']='^user_[a-z0-9_]{1,16}$'
    return result


def input_ids(tokenizer,text):
    value=tokenizer.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':text}],tools=tools(),tokenize=True,add_generation_prompt=True,enable_thinking=False)
    if hasattr(value,'keys'):value=value['input_ids']
    if value and isinstance(value[0],list):value=value[0]
    return value


def action_text(action):
    definitions={v['function']['name']:v['function']['parameters'] for v in tools()}
    name=action['tool'];args=action['arguments']
    if name not in definitions or not isinstance(args,dict):raise ValueError('Unknown native action')
    lines=['<tool_call>','<function='+name+'>']
    for key,value in args.items():
        if key not in definitions[name]['properties']:raise ValueError('Unknown native argument')
        kind=definitions[name]['properties'][key]['type']
        text=json.dumps(value,separators=(',',':')) if kind in ['array','object','boolean','integer'] else str(value)
        if '<parameter=' in text or '</parameter>' in text:raise ValueError('Unrepresentable native argument')
        lines += ['<parameter='+key+'>',text,'</parameter>']
    return '\n'.join(lines+['</function>','</tool_call>'])
