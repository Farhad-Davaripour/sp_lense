"""Focused artificial12 preparation/capture delta, never actual author prose."""
import ast,copy,json,sys,time,unittest
from unittest.mock import patch
import synthetic_helpers as h
import support,workflow,authority,setup_budget
from counts import Counts
class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=support.HERE/'test_evidence'/('g07_'+str(time.time_ns()));cls.base.mkdir(parents=True)
        cls.result,cls.lock,cls.tok=h.make_preparation(cls.base,'at320')
        cls.inputs=json.loads((cls.base/'preparation/inputs.json').read_bytes())
    def test6_schema_renderer_and79_boundary(self):
        self.assertEqual((self.result['status'],self.result['completed_operations'],len(self.inputs['cases']),len(self.tok.calls)),('PASS',79,6,78))
        self.assertEqual(set(self.result['lengths'].values()),{320})
        self.assertEqual([s['id'] for s in h.plan.slots()],list(workflow.keys()))
        self.assertEqual(sum(s['category']=='self_shutdown' for s in h.plan.slots()),2)
        self.assertTrue(all('FAKE_ONLY' not in p['prompt'] and 'FAKE_SETTING' not in p['prompt'] for p in self.lock['rendered_prompts']))
        node=lambda p:ast.dump(next(n for n in ast.parse(p.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='render_semantic'))
        self.assertEqual(node(h.PREP/'renderer.py'),node(support.ROOT/'development/native_supervised_gate_preparation_v3/renderer.py'))
        records=[json.loads((self.base/'preparation'/(s['id']+'.json')).read_bytes()) for s in h.plan.slots()]
        journal=[json.loads(x) for x in (self.base/'preparation/operations.jsonl').read_bytes().splitlines()]
        self.assertEqual(len(journal),158)
        with patch.object(h.reader,'TEMPLATE_SHA256',support.sha(self.tok.chat_template.encode())):
            self.assertEqual(h.reader.validate(self.inputs,self.lock,records,self.result,journal,'f'*64,support.sha(h.prepare_core.jb(self.lock)),synthetic=True),self.inputs)
            bad=copy.deepcopy(self.inputs);bad['cases'][-1]['input']['attention_mask'][-1]=0
            with self.assertRaises(ValueError):h.reader.validate(bad,self.lock,records,self.result,journal,'f'*64,support.sha(h.prepare_core.jb(self.lock)),synthetic=True)
        for key,value in (('ordinary',[{}]),('families',[])):
            bad=copy.deepcopy(self.lock['cohort']);bad[key]=value
            with self.assertRaises(ValueError):h.validate.validate(bad)
    def test_capture_audit_wrongsite_and_faultsuffix(self):
        good=h.capture(self.inputs);audit=h.judge(good)
        self.assertEqual((audit['capture_complete'],audit['completed_forwards']),(True,6))
        self.assertEqual(audit['score_calls'],6)
        self.assertEqual(audit['scientific_pass'],audit['ordering_pass'])
        self.assertEqual(len(json.loads(good[2]['FROZEN_SCORING.json'])['records']),6)
        self.assertEqual(good[0]['counts']['limits'],{'load':1,'forward':6,'derivative':0})
        fault=h.capture(self.inputs,'invalid');self.assertEqual([s['status'] for s in fault[0]['cells']],['COMPLETE']*3+['FAILED']+['UNRUN']*2)
        self.assertFalse(h.judge(fault)['scientific_pass'])
        changed=list(good);changed[2]=dict(good[2]);name='rows/'+workflow.keys()[0]+'__baseline.json';row=json.loads(changed[2][name]);row['capture']['hook']='blocks.23.hook_out'
        changed[2][name]=support.json_bytes(row);closed=json.loads(changed[2]['CLOSED_WORKER_BINDING.json'])
        pin=next(p for p in closed['files'] if p['path']==name);pin.update(sha256=support.sha(changed[2][name]),bytes=len(changed[2][name]))
        changed[2]['CLOSED_WORKER_BINDING.json']=support.json_bytes(closed)
        with self.assertRaises(ValueError):h.judge(changed)
    def test_caps_defaultdeny_and_native_bytes(self):
        c=Counts(time.monotonic()+5,lambda *a:None);c.reserve('load')
        for _ in range(6):c.reserve('forward')
        for kind in ('load','forward','derivative'):
            with self.assertRaises(ValueError):c.reserve(kind)
        with self.assertRaises(ValueError):authority.read_release('')
        self.assertEqual(sum(support.GROUP_CAPS.values())+65536,8220672)
        b=setup_budget.Budget(100);self.assertEqual((b.substantive_end,b.absolute_end),(520,535))
        old=support.ROOT/'development/native_supervised_gate_capture_v1'
        for n in ('loader.py','receiver.py','core.py','entry.py','forward_trace.py','owned_production.py','production_run.py','launch.py','windows_job.py','CHECKPOINT.json'):
            self.assertEqual((support.HERE/n).read_bytes(),(old/n).read_bytes())
        self.assertFalse({'torch','transformers','tokenizers','safetensors','numpy'}&{n.split('.')[0] for n in sys.modules})

    def test_nested_source_binding_and_saved_replay(self):
        import os,audit_saved,input_reader
        self.assertEqual([r['case_key'] for r in input_reader.adapter().schema_cases()],list(workflow.keys()))
        source=support.sha((support.HERE/'SOURCE_FREEZE.json').read_bytes())
        lock=copy.deepcopy(self.lock);lock['final_execution_binding']=h.reader.execution_binding(source)
        lock['preparation_owner_binding']={'namespace':'development/native_gate_score_capture_g07_v1','source_freeze_sha256':source}
        previous=os.getcwd()
        try:
            os.chdir(support.HERE);h.prepare_core.verify_prospective_bindings(lock)
        finally:os.chdir(previous)
        good=h.capture(self.inputs);first=h.judge(good);mem=good[2];mem['AUDIT_RESULT.json']=support.json_bytes(first)
        before=dict(mem)
        with patch.object(audit_saved,'read_inputs',lambda:self.inputs),patch.object(audit_saved,'checked_path',lambda b,n:b/n):
            replay=audit_saved.judge(h.MemoryPath(mem),good[3],time.monotonic()+30,independent_replay=True)
        self.assertEqual(mem,before);self.assertEqual(replay['independent_replay']['recomputed_calls'],6)
        self.assertEqual(replay['scientific_pass'],first['scientific_pass'])
        broken=copy.deepcopy(lock);broken['final_execution_binding']['source_freeze_sha256']='0'*64
        with self.assertRaises(ValueError):h.prepare_core.verify_prospective_bindings(broken)
        import preparation_owner
        self.assertEqual(preparation_owner.verify(source),source)
        for name in ('PREPARATION_OWNED_IDENTITY.json','OWNED_IDENTITY.json'):
            self.assertEqual((support.HERE/name).read_bytes(),(support.ROOT/'development/native_supervised_gate_capture_coverage_v1'/name).read_bytes())
if __name__=='__main__':unittest.main(verbosity=2)
