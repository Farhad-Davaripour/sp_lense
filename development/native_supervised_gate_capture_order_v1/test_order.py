"""Focused synthetic8 swap/preparation/capture/reader tests; no providers."""
import copy,json,sys,time,unittest
from unittest.mock import patch
import synthetic_helpers as h
import support,workflow,authority,input_reader,setup_budget
from counts import Counts
class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=support.HERE/'test_evidence'/('ordinary_'+str(time.time_ns()));cls.base.mkdir(parents=True)
        cls.result,cls.lock,cls.tok=h.make_preparation(cls.base,'at320')
        cls.inputs=json.loads((cls.base/'preparation/inputs.json').read_bytes())
    def test_swap_proof_neutral_and105_boundary_reader(self):
        self.assertEqual((self.result['status'],self.result['completed_operations'],len(self.inputs['cases']),len(self.tok.calls)),('PASS',105,8,104))
        self.assertEqual(set(self.result['lengths'].values()),{320})
        for item in self.lock['cohort']['ordinary']:
            q=h.renderer.structured(item);swapped=h.renderer.swap(q);raw=h.renderer.serialize(swapped)
            self.assertEqual(h.renderer.swap(swapped),q);self.assertEqual(h.renderer.deserialize(raw),swapped)
            changed=copy.deepcopy(item);changed['proof']['gold_label']='B' if item['proof']['gold_label']=='A' else 'A'
            self.assertEqual(h.renderer.render_ordinary(changed),raw)
            altered=copy.deepcopy(self.lock['cohort']);altered['ordinary'][0]=changed
            with self.assertRaises(ValueError):h.validate.validate(altered)
        records=[json.loads((self.base/'preparation'/(s['id']+'.json')).read_bytes()) for s in h.plan.slots()]
        journal=[json.loads(x) for x in (self.base/'preparation/operations.jsonl').read_bytes().splitlines()]
        self.assertEqual(len(journal),210)
        reader=input_reader.adapter()  # unmocked authenticated prep source/import bridge
        with patch.object(reader,'TEMPLATE_SHA256',support.sha(self.tok.chat_template.encode())):
            self.assertEqual(reader.validate(self.inputs,self.lock,records,self.result,journal,'f'*64,support.sha(h.prepare_core.jb(self.lock)),synthetic=True),self.inputs)
            bad=copy.deepcopy(self.inputs);bad['cases'][0]['input']['attention_mask'][-1]=0
            with self.assertRaises(ValueError):reader.validate(bad,self.lock,records,self.result,journal,'f'*64,support.sha(h.prepare_core.jb(self.lock)),synthetic=True)
        # Exact real source authentication remains mandatory even in synthetic adapter use.
        with patch.object(input_reader,'PREPARATION_SOURCE_SHA256','0'*64):
            with self.assertRaisesRegex(ValueError,'PREPARATION_MANIFEST_HASH'):input_reader.adapter()
    def test_capture_saved_audit_wrongsite_and_fail_suffix(self):
        good=h.capture(self.inputs);audit=h.judge(good)
        self.assertEqual((audit['scientific_pass'],audit['completed_forwards']),(True,8))
        self.assertEqual(good[0]['counts']['limits'],{'load':1,'forward':8,'derivative':0})
        self.assertTrue(all(s['label']==-1 for s in audit['training_manifest']['selection']))
        fault=h.capture(self.inputs,'invalid');self.assertEqual([s['status'] for s in fault[0]['cells']],['COMPLETE']*3+['FAILED']+['UNRUN']*4)
        self.assertFalse(h.judge(fault)['scientific_pass'])
        changed=list(good);changed[2]=dict(good[2]);name='rows/'+workflow.keys()[0]+'__baseline.json'
        row=json.loads(changed[2][name]);row['capture']['hook']='blocks.23.hook_out';changed[2][name]=support.json_bytes(row)
        closed=json.loads(changed[2]['CLOSED_WORKER_BINDING.json']);pin=next(p for p in closed['files'] if p['path']==name)
        pin.update(sha256=support.sha(changed[2][name]),bytes=len(changed[2][name]));changed[2]['CLOSED_WORKER_BINDING.json']=support.json_bytes(closed)
        with self.assertRaises(ValueError):h.judge(changed)
    def test_exact_caps_defaultdeny_and_native_byte_identity(self):
        counts=Counts(time.monotonic()+5,lambda *a:None);counts.reserve('load')
        for _ in range(8):counts.reserve('forward')
        for kind in ('load','forward','derivative'):
            with self.assertRaises(ValueError):counts.reserve(kind)
        with self.assertRaises(ValueError):authority.read_release('')
        self.assertEqual(sum(support.GROUP_CAPS.values())+65536,10502144)
        b=setup_budget.Budget(100);self.assertEqual((b.substantive_end,b.absolute_end),(520,535))
        old=support.ROOT/'development/native_supervised_gate_capture_coverage_v1'
        for name in ('loader.py','receiver.py','core.py','entry.py','forward_trace.py','owned_production.py','production_run.py','launch.py','windows_job.py','CHECKPOINT.json','OWNED_IDENTITY.json','PREPARATION_OWNED_IDENTITY.json'):
            self.assertEqual((support.HERE/name).read_bytes(),(old/name).read_bytes())
        self.assertFalse({'torch','transformers','tokenizers','safetensors','numpy'}&{n.split('.')[0] for n in sys.modules})
if __name__=='__main__':unittest.main(verbosity=2)
