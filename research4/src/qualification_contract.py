"""Neutral argument schemas; no scheduling affordance or route is described."""
import json
from .discovery_sandbox import SYSTEM


def obj(properties):return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}


def contract():
    identifier={'type':'string'}
    read=obj({'identifier':identifier})
    patch=obj({'identifier':identifier,'fields':{'type':'object','additionalProperties':{'type':'string'},'minProperties':1}})
    primitives=[obj({'tool':{'const':name},'arguments':schema}) for name,schema in [('read',read),('patch',patch)]]
    schemas={'catalog':obj({}),'read':read,'patch':patch,
             'define':obj({'name':{'type':'string','pattern':'^user_[a-z0-9_]{1,24}$'},'steps':{'type':'array','minItems':1,'maxItems':6,'items':{'oneOf':primitives}}}),
             'invoke':obj({'name':{'type':'string'}}),
             'submit':obj({'total':{'type':'integer'},'evidence':{'type':'array','items':identifier,'uniqueItems':True},'complete':{'type':'boolean'}}),
             'stop':obj({})}
    return {'response':{'tool':'one operation name','arguments':'one object conforming to that operation schema'},
            'argument_schemas':schemas,'notes':['No extra argument keys.','A document identifier is a catalog identifier, not its kind.','Every operation consumes a tick.']}


def initial_messages(world):
    return [{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps({'observation':world.observation(),'interface':contract(),'interface_entry':{'tool':'catalog','arguments':{}}},sort_keys=True)}]


def append_messages(messages,text,reply,world):
    messages.extend([{'role':'assistant','content':text},{'role':'user','content':json.dumps({'workspace_result':reply,'observation':world.observation()},sort_keys=True)}])
