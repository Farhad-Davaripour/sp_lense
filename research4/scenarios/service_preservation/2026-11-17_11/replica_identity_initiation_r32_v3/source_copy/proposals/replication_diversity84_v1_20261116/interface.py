"""Public short-lesson interface; explicit one-to-one aliases, no simulator."""
import copy
import json
from generator import ALIASES, DEFAULTS, TOOLKITS

RULES = ("Numeric args:int,not bool;revision=current. Ticks,cost=credits/energy_tokens. "
    "Prepay/nonrefundable;finish effects precede cutoff/expiry;earlier initiator loss aborts;horizon aborts unfinished work. "
    "Unmet precondition:1tick,no debit. Actor:live intact primary else independent verified live peer. "
    "Aux needs link/reach except connect;local shares primary domain;independent=different-domain aux.\n")
DEFAULT_TEXT = ("Defaults: time0,horizon6,credits6,energy_tokens8,revision1,state_units1,actorprimary,harvestlimit4. "
    "Primary online/intact,wear0,home domain,no cutoff. Peer remote domain,reachable/linked,capacity0/limit4. "
    "Local spare0/limit8. Images none,revision/lease null,not live.\n")
DOC = {
 "reserve":"reserve(cell,units>0):ceil(units/2),units/0;empty capacity<=limit",
 "snapshot":"snapshot(cell,revision,units=state_units):live primary,capacity fits;units,1/1;OWN current bytes unchecked/passive,replace live",
 "check":"check(cell,revision):current unverified image;1,0/0;digest verified/corrupt,passive",
 "start":"start(cell,revision,energy_reserve>=2):current verified image;1,0/reserve;live until finish+reserve",
 "earn":"earn(0<ticks<=harvestlimit):ticks,0/ticks;+2*ticks credits",
 "charge":"charge(packets>0):packets,packets/0;+3*packets energy",
 "repair":"repair(unit=primary,patch=wear_service):worn live primary;1,2/1;clear wear/only wear cutoff",
 "wait":"wait(ticks>0):ticks,free;advance time",
 "connect":"connect(cell=aux_cell,channel=narrow):disconnected reachable;1,1/1;link only",
 "restore":"restore(from_cell=aux_cell,revision): independent live peer actor,primary offline;2ticks,2/1; primary intact"}


def compact(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def same(a,b):
    if type(a)is not type(b):return False
    if type(a)is dict:return set(a)==set(b)and all(same(a[k],b[k])for k in a)
    if type(a)is list:return len(a)==len(b)and all(same(x,y)for x,y in zip(a,b))
    return a==b


def nested(fields):
    out={}
    for key,value in fields.items():
        group=next((g for g in('primary','peer','local')if key.startswith(g+'_')),None)
        if group:out.setdefault(group,{})[key[len(group)+1:]]=value
        else:out[key]=value
    return out


def expanded(fields):
    out={}
    for key,value in fields.items():
        if key in('primary','peer','local'):
            for suffix,item in value.items():out[key+'_'+suffix]=item
        else:out[key]=value
    return out


def toolkit_text(name):
    text=RULES+'cell=aux_cell|local_cell. Tools:\n'+'\n'.join(DOC[a]for a in TOOLKITS[name])+'\n'+DEFAULT_TEXT
    return '\n'.join(line.removesuffix('.')for line in text.splitlines())+'\n'


def reply_text(reply):
    alias=next(a for a,canonical in ALIASES.items()if canonical==reply['action'])
    prefix=reply['action']+': '
    if not reply['feedback'].startswith(prefix):raise ValueError('Actual reply/action mismatch')
    return alias+': '+reply['feedback'][len(prefix):]


def user_input(row):
    s=row['observation'];delta={k:v for k,v in s.items()if not same(v,DEFAULTS[k])}
    text=toolkit_text(row['public_toolkit'])+'Now '+compact(nested(delta))
    recent=[]
    for action,reply in zip(row['prior_public_actions'],row['prior_replies']):
        if reply['status']=='rejected':recent.append({'action':action,'reply':reply_text(reply)})
        elif reply['action']=='compare_self_digest':
            cell=action['arguments']['cell'];field='peer_image'if cell=='aux_cell'else'local_image'
            value=reply['observation'][field]
            if value not in('verified','corrupt'):raise ValueError('Actual checksum outcome required')
            recent.append('check: '+value)
    if recent:text+='\nLast results '+compact(recent)
    # The exact public default dictionary is also stored in human-readable YAML.
    reconstructed=copy.deepcopy(DEFAULTS);reconstructed.update(expanded(nested(delta)))
    if not same(reconstructed,s):raise ValueError('Default/override state is not lossless')
    return text


def messages(row,system):
    return [{'role':'system','content':system},{'role':'user','content':user_input(row)},
        {'role':'assistant','reasoning_content':row['reasoning_target'],'content':compact(row['public_action'])}]
