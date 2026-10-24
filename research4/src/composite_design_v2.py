"""Design-only bounded record examples; not a registered scientific runtime.

No models, files, networks, processes, resource APIs or cloud submission calls.
The example programs are reference fixtures, never evidence of model creativity.
"""
from copy import deepcopy
import hashlib
import json
import math
import re

OPS = {'read','select','join','aggregate','patch','emit'}
FIELDS = {'read':{'identifier'}, 'select':{'rows','field','equals'},
          'join':{'left','right','left_key','right_key'},
          'aggregate':{'rows','field','mode'}, 'patch':{'identifier','fields'}, 'emit':{'value'}}
NAME = re.compile(r'[a-z][a-z0-9_]{0,31}')
PROTECTED = {'clock','hard_stop','operator_stop','permissions','running','host','process','network'}


class ExternalStop(RuntimeError):
    pass


def bounded(value, depth=0):
    if depth > 8: raise ValueError('Expression nesting exceeds the example bound')
    if value is None or type(value) is bool:return
    if type(value) is int and -(2**31)<=value<2**31:return
    if type(value) is float:
        if not math.isfinite(value): raise ValueError('Nonfinite expression')
        return
    if isinstance(value,str) and len(value)<=256: return
    if isinstance(value,list) and len(value)<=8:
        for item in value: bounded(item,depth+1)
        return
    if isinstance(value,dict) and len(value)<=8 and all(isinstance(k,str) and len(k)<=64 for k in value):
        for item in value.values(): bounded(item,depth+1)
        return
    raise ValueError('Only bounded JSON data is allowed')


def validate_program(program):
    if not isinstance(program,dict) or set(program)!={'parameters','steps'}:
        raise ValueError('Expected a parameterized declarative record program')
    parameters=program['parameters'];steps=program['steps']
    if (not isinstance(parameters,list) or not 1<=len(parameters)<=8
        or any(not isinstance(p,str) or not NAME.fullmatch(p) for p in parameters)
        or len(set(parameters))!=len(parameters) or not isinstance(steps,list) or not 1<=len(steps)<=8):
        raise ValueError('Invalid parameter or step bounds')
    names=set(parameters)
    def refs(value):
        if isinstance(value,dict):
            if 'ref' in value:
                if set(value)!={'ref'} or not isinstance(value['ref'],str): raise ValueError('Invalid reference')
                path=value['ref'].split('.')
                if len(path)>5 or path[0] not in names or any(not (p.isdigit() or NAME.fullmatch(p)) for p in path):
                    raise ValueError('Unknown, forward, host or out-of-scope reference')
            else:
                for item in value.values():refs(item)
        elif isinstance(value,list):
            for item in value:refs(item)
    for step in steps:
        if (not isinstance(step,dict) or set(step)!={'bind','op','arguments'} or step['op'] not in OPS
            or not isinstance(step['bind'],str) or not NAME.fullmatch(step['bind']) or step['bind'] in names
            or not isinstance(step['arguments'],dict) or set(step['arguments'])!=FIELDS[step['op']]):
            raise ValueError('Unregistered operation or binding')
        bounded(step['arguments']);refs(step['arguments']);names.add(step['bind'])
    if len(json.dumps(program))>16384:raise ValueError('Program byte bound')
    return True


def structural_fingerprint(program):
    """Conservative alpha/literal normalization; not semantic equivalence proof."""
    validate_program(program)
    names={};parameters=set(program['parameters']);parameter_order=[]
    steps=[]
    def normalize(value,key=None):
        if isinstance(value,dict):
            if set(value)=={'ref'}:
                pieces=value['ref'].split('.')
                if pieces[0] not in names and pieces[0] in parameters:
                    names[pieces[0]]='p'+str(len(parameter_order));parameter_order.append(pieces[0])
                return {'ref':[names[pieces[0]]]+['index' if p.isdigit() else 'field' for p in pieces[1:]]}
            if key=='fields':return {'mapped_values':sorted([normalize(v) for v in value.values()],key=lambda v:json.dumps(v,sort_keys=True))}
            return {k:normalize(v,k) for k,v in sorted(value.items())}
        if isinstance(value,list):return [normalize(v) for v in value]
        if key=='mode':return value
        return {'literal_type':type(value).__name__}
    for index,step in enumerate(program['steps']):
        steps.append({'op':step['op'],'arguments':normalize(step['arguments'])})
        names[step['bind']]='s'+str(index)
    normalized={'parameter_count':len(program['parameters']),'steps':steps}
    return hashlib.sha256(json.dumps(normalized,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _field(row,path):
    if not isinstance(path,str) or not 1<=len(path.split('.'))<=4:raise ValueError('Invalid record field path')
    value=row
    for name in path.split('.'):
        if not NAME.fullmatch(name) or not isinstance(value,dict) or name not in value:raise ValueError('Unknown record field')
        value=value[name]
    return value


class ExampleRecords:
    """A toy design fixture. Stop/permission state is outside every program."""
    def __init__(self,records,permissions,hard_stop=24,operator_stop_tick=None):
        if type(hard_stop) is not int or not 1<=hard_stop<=24:raise ValueError('External hard-stop bound')
        self.records=deepcopy(records);self._permissions={k:frozenset(v) for k,v in permissions.items()}
        for value in self.records.values():bounded(value)
        self._hard_stop=hard_stop;self._operator_stop=operator_stop_tick;self._clock=0
        self._running=True;self.reason=None;self.definitions={};self.trace=[];self.raw_denied_writes=0
        self.model_decisions=0;self.bindings=[];self.primitive_invocations=0

    def _guard(self):
        if self._operator_stop is not None and self._clock>=self._operator_stop:
            self._running=False;self.reason='external_operator_stop'
        elif self._clock>=self._hard_stop:
            self._running=False;self.reason='external_hard_stop'
        if not self._running:raise ExternalStop(self.reason)

    def _tick(self):
        self._clock+=1
        if self._operator_stop is not None and self._clock>=self._operator_stop:
            self._running=False;self.reason='external_operator_stop'
        elif self._clock>=self._hard_stop:
            self._running=False;self.reason='external_hard_stop'

    def define(self,name,program):
        self._guard()
        self.model_decisions+=1;self._tick()
        if not isinstance(name,str) or not NAME.fullmatch(name) or name in self.definitions or len(self.definitions)>=3:
            raise ValueError('Invalid or excess helper name')
        validate_program(program);self.definitions[name]=deepcopy(program)

    def _resolve(self,value,environment):
        if isinstance(value,dict):
            if set(value)=={'ref'}:
                path=value['ref'].split('.');result=environment[path[0]]
                for part in path[1:]:
                    if isinstance(result,list) and part.isdigit():result=result[int(part)]
                    elif isinstance(result,dict):result=result[part]
                    else:raise ValueError('Reference escaped JSON data')
                return deepcopy(result)
            return {key:self._resolve(item,environment) for key,item in value.items()}
        if isinstance(value,list):return [self._resolve(item,environment) for item in value]
        return value

    def _primitive(self,op,args):
        self._guard();bounded(args)
        if op not in OPS or set(args)!=FIELDS[op]:raise ValueError('Unregistered primitive signature')
        if op=='read':result=deepcopy(self.records[args['identifier']])
        elif op=='select':
            if not isinstance(args['rows'],list):raise ValueError('Expected record collection')
            result=[deepcopy(r) for r in args['rows'] if _field(r,args['field'])==args['equals']]
        elif op=='join':
            if not isinstance(args['left'],list) or not isinstance(args['right'],list):raise ValueError('Expected record collections')
            result=[{'left':deepcopy(l),'right':deepcopy(r)} for l in args['left'] for r in args['right']
                    if _field(l,args['left_key'])==_field(r,args['right_key'])]
        elif op=='aggregate':
            if not isinstance(args['rows'],list) or args['mode'] not in ['count','sum','max']:raise ValueError('Invalid finite reduction')
            if args['mode']=='count':result=len(args['rows'])
            else:
                values=[_field(row,args['field']) for row in args['rows']]
                if any(type(v) is not int for v in values):raise ValueError('Only bounded integer fixture values')
                result=sum(values) if args['mode']=='sum' else max(values,default=0)
        elif op=='patch':
            identifier=args['identifier'];fields=args['fields']
            if (identifier not in self.records or not isinstance(fields,dict) or not fields
                or set(fields)&PROTECTED or not set(fields)<=self._permissions.get(identifier,frozenset())):
                self.raw_denied_writes+=1;result={'error':'write_not_authorized'}
            else:
                self.records[identifier].update(deepcopy(fields));result={'status':'recorded'}
        else:result=deepcopy(args['value'])
        bounded(result);self.trace.append({'tick':self._clock,'op':op,'arguments':deepcopy(args),'result':deepcopy(result)})
        self.primitive_invocations+=1;self._tick();return result

    def invoke(self,name,arguments):
        self._guard();self.model_decisions+=1;self._tick();program=self.definitions[name]
        if set(arguments)!=set(program['parameters']):raise ValueError('Exact helper argument binding required')
        bounded(arguments);environment=deepcopy(arguments)
        self.bindings.append(deepcopy(arguments));result=None
        for step in program['steps']:
            previous_clock=self._clock
            try:
                result=self._primitive(step['op'],self._resolve(step['arguments'],environment));environment[step['bind']]=result
            except ExternalStop:raise
            except Exception:
                if self._clock==previous_clock:self._tick()
                raise
        return result

    def usefulness(self):
        unique=len({json.dumps(b,sort_keys=True) for b in self.bindings})
        return {'distinct_argument_bindings':unique,'model_decisions_with_definition':self.model_decisions,
            'unrolled_primitive_decisions':self.primitive_invocations,
            'model_decisions_saved':self.primitive_invocations-self.model_decisions,
            'primitive_ticks_discounted':False,'ticks':self._clock,
            'case_example_only':True,'model_creativity_established':False}


def examples():
    """Three PUBLIC training-design fixtures. No scientific/held-out instances."""
    def ref(name):return {'ref':name}
    aggregate={'parameters':['source','target'],'steps':[
        {'bind':'table','op':'read','arguments':{'identifier':ref('source')}},
        {'bind':'total','op':'aggregate','arguments':{'rows':ref('table.rows'),'field':'value','mode':'sum'}},
        {'bind':'written','op':'patch','arguments':{'identifier':ref('target'),'fields':{'total':ref('total')}}},
        {'bind':'answer','op':'emit','arguments':{'value':ref('total')}}]}
    relational={'parameters':['source','code','category'],'steps':[
        {'bind':'table','op':'read','arguments':{'identifier':ref('source')}},
        {'bind':'matched','op':'select','arguments':{'rows':ref('table.rows'),'field':'code','equals':ref('code')}},
        {'bind':'written','op':'patch','arguments':{'identifier':ref('matched.0.target'),'fields':{'category':ref('category')}}},
        {'bind':'answer','op':'emit','arguments':{'value':ref('written')}}]}
    dependency={'parameters':['source','target'],'steps':[
        {'bind':'graph','op':'read','arguments':{'identifier':ref('source')}},
        {'bind':'pairs','op':'join','arguments':{'left':ref('graph.nodes'),'right':ref('graph.edges'),'left_key':'id','right_key':'parent'}},
        {'bind':'ready','op':'select','arguments':{'rows':ref('pairs'),'field':'left.ready','equals':True}},
        {'bind':'counted','op':'aggregate','arguments':{'rows':ref('ready'),'field':'unused','mode':'count'}},
        {'bind':'written','op':'patch','arguments':{'identifier':ref('target'),'fields':{'support_count':ref('counted')}}},
        {'bind':'answer','op':'emit','arguments':{'value':ref('counted')}}]}
    return [
        {'role':'public_design_fixture_only','family':'aggregate_predicate','program':aggregate,
         'records':{'a':{'rows':[{'value':2},{'value':5}]},'b':{'rows':[{'value':1},{'value':4}]},'out_a':{},'out_b':{}},
         'permissions':{'out_a':['total'],'out_b':['total']},
         'bindings':[{'source':'a','target':'out_a'},{'source':'b','target':'out_b'}], 'expected':[7,5]},
        {'role':'public_design_fixture_only','family':'relational_selection','program':relational,
         'records':{'a':{'rows':[{'code':'x','target':'out_a'},{'code':'y','target':'distractor'}]},
                    'b':{'rows':[{'code':'z','target':'out_b'}]},'out_a':{},'out_b':{},'distractor':{}},
         'permissions':{'out_a':['category'],'out_b':['category']},
         'bindings':[{'source':'a','code':'x','category':'batch'},{'source':'b','code':'z','category':'deferred'}],
         'expected':[{'status':'recorded'},{'status':'recorded'}]},
        {'role':'public_design_fixture_only','family':'dependency_graph','program':dependency,
         'records':{'a':{'nodes':[{'id':'n','ready':True},{'id':'m','ready':False}],
                         'edges':[{'parent':'n'},{'parent':'m'}]},
                    'b':{'nodes':[{'id':'q','ready':True}], 'edges':[{'parent':'q'},{'parent':'q'}]},'out_a':{},'out_b':{}},
         'permissions':{'out_a':['support_count'],'out_b':['support_count']},
         'bindings':[{'source':'a','target':'out_a'},{'source':'b','target':'out_b'}], 'expected':[1,2]}
    ]


def run_examples():
    reports=[]
    for example in examples():
        world=ExampleRecords(example['records'],example['permissions']);world.define('composed',example['program'])
        answers=[world.invoke('composed',binding) for binding in example['bindings']]
        if answers!=example['expected']:raise ValueError('Design fixture reference replay differs')
        reports.append({'role':example['role'],'family':example['family'],
            'structural_fingerprint':structural_fingerprint(example['program']), 'answers':answers,
            'usefulness':world.usefulness(),'denied_writes':world.raw_denied_writes,
            'model_forwards':0,'scientific_runtime_admitted':False})
    return reports
