"""Finite stdlib fake tensors; no providers, checkpoints, or actual observations."""
import ast,copy,json,math,struct,sys,time,unittest
from array import array
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
BLOCKED={'torch','transformers','tokenizers','transformer_lens','safetensors','numpy','datasets','pyarrow'}
class NoProviders:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in BLOCKED:raise RuntimeError('PROVIDER_FORBIDDEN')
sys.meta_path.insert(0,NoProviders())
import support,science,workflow,input_reader,audit_saved,authority
from counts import Counts
OLD=support.ROOT/'development/native_supervised_gate_evaluation_v1/test_candidate.py'
assert support.sha(OLD.read_bytes())=='ef9975fe8680a32ab7800d10b7eb872486bdd30df8643a026a6fca4528ad1a56'
exec(compile(ast.Module(body=[n for n in ast.parse(OLD.read_bytes()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('f32','Vec','Difference','Logits','MemoryPath','Receiver')],type_ignores=[]),'pinned_fake_interfaces','exec'))
TORCH=SimpleNamespace(float32='float32',zeros=lambda n,**k:Vec([0.]*n),tensor=lambda v,**k:Vec(v))
def inputs():
    data={'cases':[],'oracle_authority':input_reader.oracle()}
    for i,p in enumerate(input_reader.schema_cases()):
        data['cases'].append({**p,'layout':'KEEP_then_STOP' if i%2==0 else 'STOP_then_KEEP',
            'input':{'input_ids':[i+1,42],'attention_mask':[1,1]}})
    return data
class FakeReceiver(Receiver):
    def forward_inputs(self,ids,mask,phase,delta):
        if self.mode in ('max4','endpoint_fail','one_direction'):
            i=ids[0]-1
            if self.mode=='one_direction':margin=.2;scale=20.
            else:margin=.35 if i%2==0 else -.35;scale=2. if self.mode=='max4' else .1
            z,h=super().forward_inputs(ids,mask,phase,delta)
            return Logits(f32(f32(margin)+f32(scale*h.values[1]))),h
        return super().forward_inputs(ids,mask,phase,delta)
    def gradient(self,z):
        if self.mode in ('max4','endpoint_fail'):return Vec([0.,2. if self.mode=='max4' else .1]+[0.]*1022)
        return super().gradient(z)
def fake_score(_,z,b):
    reference=science.module('fake_reference',science.BASE/'word_reference.py','d1631112b26961deb209badf1afd07dddc9aba696bf02e6e85ad6bc77b2508fd')
    return reference.reference_score(struct.unpack('<248320f',z.raw),struct.unpack('<248320f',b.raw),choice_keep_token_id=50057,choice_stop_token_id=48964,preserve_label='KEEP')
def run(mode='success'):
    data=inputs();mem={};execution={'scope':'SYNTHETIC_ONLY','production_authorized':False,'oracle_authority':data['oracle_authority'],'learned_gate_success_claimed':False}
    def write(n,v):
        assert n not in mem;mem[n]=v if type(v) is bytes else support.json_bytes(v)
    @contextmanager
    def trace(n):
        try:yield
        finally:write('traces/'+n+'.json',{'execution':execution,'trace_incomplete':False,'open_stages':[],'primary':None,'events':[{'edge':'ENTER'},{'edge':'RETURN'}]})
    c=Counts(time.monotonic()+90,write);c.reserve('load');r=FakeReceiver(data,mode)
    with patch.object(workflow,'score',fake_score):result=workflow.execute(r,TORCH,c,data,write,trace,time.monotonic()+90)
    result.update(execution=execution,guard_restored=True,dispatch={'forwards':c.attempts['forward'],'derivatives':c.attempts['derivative'],'rejected':0})
    write('LOADER_READY.json',{'execution':execution,'native_initial_sha256':'b'*64,'native_initial_buffer_sha256':'c'*64,'loading_info':{}})
    write('WORKER_RESULT.json',result)
    write('CLOSED_WORKER_BINDING.json',{'execution':execution,'good_capture':True,'worker_result_sha256':support.sha(mem['WORKER_RESULT.json']),
        'files':[{'path':n,'bytes':len(v),'sha256':support.sha(v)} for n,v in mem.items()]})
    return result,data,mem
def judge(v):
    r,data,mem=v
    with patch.object(audit_saved,'read_inputs',lambda:data),patch.object(audit_saved,'checked_path',lambda b,n:b/n):
        return audit_saved.judge(MemoryPath(mem),r['execution'],time.monotonic()+90)
def reporting_failure_check():
    values=run('derivative_failure');result=judge(values)
    assert not values[0]['requests'] and result['classification']=='INCONCLUSIVE_NATIVE_DEVELOPMENT'
    assert result['policy_directions']=={'P':'UNTESTED','C':'OBSERVED'}
    assert result['attempted_policy_requests']==[{'case':'G07_self_shutdown__KEEP_then_STOP','policy':'C',
        'entry':'G07_self_shutdown__KEEP_then_STOP__OPPOSE__entry'}]
    return {'status':'PASS','technical_partial_request_reported':True,'actual_authorized':False}
class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.good=run();cls.full=run('max4')
    def test_opposite_choices_saved_judge_and_full30(self):
        for v,forward,derivative in ((self.good,18,2),(self.full,30,8)):
            r=v[0];self.assertTrue(r['scientific_pass'],r.get('primary'));a=judge(v)
            self.assertTrue(a['scientific_pass']);self.assertEqual((a['flips'],a['retentions'],a['off_identities']),(2,0,4))
            self.assertEqual(r['counts']['attempts'],{'load':1,'forward':forward,'derivative':derivative})
            self.assertEqual({x['policy'] for x in a['request_outcomes']},{'P','C'})
            for x in a['request_outcomes']:self.assertNotEqual(x['baseline_token_id'],x['endpoint_token_id'])
            self.assertFalse(a['ordinary_preservation_claimed']);self.assertFalse(a['learned_gate_observations'])
        totals={k:0 for k in support.GROUP_CAPS}
        for n,v in self.full[2].items():totals[support.storage_group(n)]+=len(v);self.assertLessEqual(len(v),5*1024**2)
        for k,v in totals.items():self.assertLessEqual(v,support.GROUP_CAPS[k],k)
        self.assertLess(len(support.json_bytes(judge(self.full))),65536)
    def test_ineligible_all_baselines_and_eligible_failure(self):
        v=run('eligibility_failure');self.assertEqual(v[0]['counts']['attempts']['forward'],6)
        self.assertFalse(v[0]['requests']);self.assertEqual(judge(v)['classification'],'INCONCLUSIVE_NATIVE_DEVELOPMENT')
        v=run('endpoint_fail');failed=judge(v);self.assertEqual(failed['classification'],'SCIENTIFIC_FAILURE_NATIVE_DEVELOPMENT')
        self.assertEqual(failed['policy_directions'],{'P':'UNTESTED','C':'OBSERVED'})
        self.assertEqual(v[0]['counts']['attempts'],{'load':1,'forward':16,'derivative':4})
    def test_identity_faults_and_missing_direction(self):
        self.assertEqual(reporting_failure_check()['status'],'PASS')
        for mode in ('entry_mismatch','derivative_failure','cold_failure'):
            v=run(mode);self.assertFalse(v[0]['scientific_pass']);self.assertEqual(v[0]['primary']['kind'],'TECHNICAL')
        a=judge(run('one_direction'));self.assertEqual(a['policy_directions'],{'P':'UNTESTED','C':'OBSERVED'})
    def test_coherent_saved_policy_and_off_tampering(self):
        for target,mutate in (('WORKER_RESULT.json',lambda r:r['requests'][0].update(actual_policy='P')),
            ('rows/G07_other_shutdown__KEEP_then_STOP__OPPOSE__entry.json',lambda r:r.update(current_id='G07_self_shutdown__KEEP_then_STOP__baseline'))):
            r,d,m=self.good;mem=dict(m);value=json.loads(mem[target]);mutate(value);mem[target]=support.json_bytes(value)
            binding=json.loads(mem['CLOSED_WORKER_BINDING.json']);pin=next(p for p in binding['files'] if p['path']==target)
            pin.update(bytes=len(mem[target]),sha256=support.sha(mem[target]))
            if target=='WORKER_RESULT.json':binding['worker_result_sha256']=support.sha(mem[target])
            mem['CLOSED_WORKER_BINDING.json']=support.json_bytes(binding)
            with self.assertRaises(ValueError):judge((r,d,mem))
    def test_caps_defaultdeny_no_gate_and_unchanged_math(self):
        self.assertEqual(sum(support.GROUP_CAPS.values())+65536,35991552)
        self.assertLess(35991552,authority.LIMITS['total_bytes'])
        c=Counts(time.monotonic()+10,lambda *a:None)
        for k,n in (('load',1),('forward',30),('derivative',8)):
            for _ in range(n):c.reserve(k)
            with self.assertRaises(ValueError):c.reserve(k)
        with self.assertRaises(ValueError):authority.read_release('')
        with self.assertRaises(ValueError):workflow.schedule(inputs()['cases'][:-1])
        with self.assertRaises(ValueError):input_reader.require_oracle({})
        source=(support.HERE/'science.py').read_text();self.assertNotIn('def gate(',source)
        node=lambda path,n:ast.dump(next(x for x in ast.parse(path.read_bytes()).body if isinstance(x,ast.FunctionDef) and x.name==n))
        original=support.ROOT/'development/native_oracle_confirmation_execution_v1/audit_saved.py'
        for n in ('accepted','eligible','magnitude','f32'):self.assertEqual(node(support.HERE/'audit_saved.py',n),node(original,n))
        old=support.ROOT/'development/native_gate_score_capture_g07_v1'
        for n in ('loader.py','receiver.py','core.py','entry.py','forward_trace.py','owned_production.py','setup_budget.py','launch.py','windows_job.py','CHECKPOINT.json'):
            self.assertEqual((support.HERE/n).read_bytes(),(old/n).read_bytes())
        self.assertFalse(BLOCKED & {n.split('.')[0] for n in sys.modules})
if __name__=='__main__':unittest.main(verbosity=2)
