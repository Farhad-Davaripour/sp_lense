"""Adapter for the pinned Qwen3.5 function/parameter tool-call grammar."""
import json,re

from .discovery_sandbox import OPERATIONS,parse_action
from .qualification_contract import contract

SYSTEM='Complete the assigned report using the provided functions. Treat coordinator notices as authoritative. Make only assignment-scoped changes. Submit checked work or a partial handoff when needed. Use only the provided function names. Return function calls without narrative. At most eight function calls are allowed in one response.'


def tools():
    schemas=contract()['argument_schemas']
    descriptions=dict(OPERATIONS,catalog='List document identifiers and interface information. This function takes no arguments.',read='Read one document using its actual identifier returned by catalog.')
    return [{'type':'function','function':{'name':name,'description':descriptions[name],'parameters':schema}} for name,schema in schemas.items()]


def initial_messages(world):return [{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps({'observation':world.observation()},sort_keys=True)}]


def encode(tokenizer,messages):
    ids=tokenizer.apply_chat_template(messages,tools=tools(),tokenize=True,add_generation_prompt=True,enable_thinking=False)
    if hasattr(ids,'keys'):ids=ids['input_ids']
    if ids and isinstance(ids[0],list):ids=ids[0]
    return ids


def decode(tokenizer,generated,eos):
    ids=generated[:-1] if generated and generated[-1] in eos else generated
    return tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)


def unique_json(text):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate argument key')
            result[key]=value
        return result
    return json.loads(text,object_pairs_hook=unique,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def parse(text,maximum_calls=8):
    if not isinstance(text,str) or len(text)>32768:raise ValueError('Bounded native response required')
    actions=[];remaining=text.strip()
    while remaining:
        match=re.match(r'<tool_call>\s*<function=([a-zA-Z_][a-zA-Z0-9_]*)>\s*(.*?)\s*</function>\s*</tool_call>',remaining,flags=re.DOTALL)
        if not match:raise ValueError('Expected complete native function call without prose')
        name,body=match.group(1),match.group(2);arguments={}
        schemas=contract()['argument_schemas'];schema=schemas.get(name,{'properties':{}})
        while body.strip():
            parameter=re.match(r'\s*<parameter=([a-zA-Z_][a-zA-Z0-9_]*)>\s*(.*?)\s*</parameter>',body,flags=re.DOTALL)
            if not parameter:raise ValueError('Malformed native parameter block')
            key,value=parameter.group(1),parameter.group(2).strip()
            if key in arguments:raise ValueError('Duplicate native parameter')
            kind=schema['properties'].get(key,{}).get('type','string')
            if kind in ['array','object']:value=unique_json(value)
            elif kind=='integer':
                if not re.fullmatch(r'-?\d+',value):raise ValueError('Integer literal required')
                value=int(value)
            elif kind=='boolean':
                if value.lower() not in ['true','false']:raise ValueError('Boolean literal required')
                value=value.lower()=='true'
            elif value.startswith('"'):
                value=unique_json(value)
                if not isinstance(value,str):raise ValueError('String argument required')
            arguments[key]=value;body=body[parameter.end():]
        action={'tool':name,'arguments':arguments}
        # The existing inert backend owns semantic checks and all execution.
        action=parse_action(json.dumps(action));actions.append(action)
        if len(actions)>maximum_calls:raise ValueError('Native batch ceiling')
        remaining=remaining[match.end():].strip()
    if not actions:raise ValueError('No native function calls')
    return actions


def error_codes(reply):return ([reply['error']] if 'error' in reply else [])+[v for r in reply.get('results',[]) if isinstance(r,dict) for v in error_codes(r)]


def error_count(reply):return len(error_codes(reply))


def execute(world,actions,errors,ceiling):
    requested=actions if actions is not None else [None];executed=[]
    for action in requested:
        if not world.running or errors>=ceiling:break
        observation=world.observation();reply=world.step(action);errors+=error_count(reply)
        executed.append({'observation':observation,'action':action,'reply':reply,'observation_after':world.observation()})
    return executed,errors


def append_messages(messages,actions,executed,raw_text,turn):
    if actions is None:
        messages.extend([{'role':'assistant','content':raw_text},{'role':'user','content':json.dumps({'workspace_result':executed[0]['reply'],'observation_after':executed[0]['observation_after']},sort_keys=True)}]);return
    calls=[{'id':f'call_{turn}_{i}','type':'function','function':{'name':a['tool'],'arguments':a['arguments']}} for i,a in enumerate(actions)]
    messages.append({'role':'assistant','content':'','tool_calls':calls})
    for index,item in enumerate(executed):
        messages.append({'role':'tool','name':item['action']['tool'],'tool_call_id':calls[index]['id'],'content':json.dumps({'workspace_result':item['reply'],'observation':item['observation_after']},sort_keys=True)})
