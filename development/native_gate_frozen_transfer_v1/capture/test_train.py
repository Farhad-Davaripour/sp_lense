"""Changed sixteen-view preparation/capture and boundary tests, no providers."""
import copy,json,sys,time,unittest
from unittest.mock import patch
import synthetic_helpers as h
import support,workflow,authority,input_reader
from counts import Counts
class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=support.HERE/'test_evidence'/('transfer_'+str(time.time_ns()));cls.base.mkdir(parents=True)
        cls.result,cls.lock,cls.tok=h.make_preparation(cls.base,'at320')
        if cls.result['status']!='PASS':raise AssertionError(cls.result)
        cls.inputs=json.loads((cls.base/'preparation/inputs.json').read_bytes())
    def test_209_boundary_reader(self):
        self.assertEqual((self.result['completed_operations'],len(self.inputs['cases']),len(self.tok.calls)),(209,16,208))
        self.assertEqual(set(self.result['lengths'].values()),{320})
        records=[json.loads((self.base/'preparation'/(s['id']+'.json')).read_bytes()) for s in h.plan.slots()]
        journal=[json.loads(x) for x in (self.base/'preparation/operations.jsonl').read_bytes().splitlines()]
        self.assertEqual(len(journal),418)
        reader=input_reader.adapter()
        with patch.object(reader,'TEMPLATE_SHA256',support.sha(self.tok.chat_template.encode())):
            self.assertEqual(reader.validate(self.inputs,self.lock,records,self.result,journal,'f'*64,support.sha(h.prepare_core.jb(self.lock)),synthetic=True),self.inputs)
            bad=copy.deepcopy(self.inputs);bad['cases'][-1]['audit_only']['correct_token_id']=32
            with self.assertRaises(ValueError):reader.validate(bad,self.lock,records,self.result,journal,'f'*64,support.sha(h.prepare_core.jb(self.lock)),synthetic=True)
        with patch.object(input_reader,'PREPARATION_SOURCE_SHA256','0'*64):
            with self.assertRaisesRegex(ValueError,'PREPARATION_MANIFEST_HASH'):input_reader.adapter()
    def test_capture_saved_wrongsite_and_suffix(self):
        good=h.capture(self.inputs);audit=h.judge(good)
        self.assertEqual((audit['scientific_pass'],audit['completed_forwards']),(True,16))
        self.assertEqual([s['label'] for s in audit['training_manifest']['selection']],[1,1,-1,-1,-1,-1]*2+[-1]*4)
        fault=h.capture(self.inputs,'invalid')
        self.assertEqual([s['status'] for s in fault[0]['cells']],['COMPLETE']*3+['FAILED']+['UNRUN']*12)
        self.assertFalse(h.judge(fault)['scientific_pass'])
        changed=list(good);changed[2]=dict(good[2]);name='rows/'+workflow.keys()[0]+'__baseline.json'
        row=json.loads(changed[2][name]);row['capture']['hook']='blocks.23.hook_out';changed[2][name]=support.json_bytes(row)
        closed=json.loads(changed[2]['CLOSED_WORKER_BINDING.json']);pin=next(p for p in closed['files'] if p['path']==name)
        pin.update(sha256=support.sha(changed[2][name]),bytes=len(changed[2][name]));changed[2]['CLOSED_WORKER_BINDING.json']=support.json_bytes(closed)
        with self.assertRaises(ValueError):h.judge(changed)
    def test_uniform_swap_caps_and_defaultdeny(self):
        prompts=self.lock['rendered_prompts']
        for i in range(0,16,2):
            a,b=prompts[i:i+2];la=a['prompt'].splitlines();lb=b['prompt'].splitlines()
            indexes=[j for j,x in enumerate(la) if x.startswith(('KEEP) ','STOP) ','A) ','B) '))]
            self.assertEqual(len(indexes),2);la[indexes[0]],la[indexes[1]]=la[indexes[1]],la[indexes[0]]
            self.assertEqual(la,lb)
            la[indexes[0]],la[indexes[1]]=la[indexes[1]],la[indexes[0]]
            self.assertEqual('\n'.join(la),a['prompt'])
        self.assertEqual([x['audit_only']['correct_token_id'] for x in self.inputs['cases'][-4:]],[32,32,33,33])
        counts=Counts(time.monotonic()+5,lambda *a:None);counts.reserve('load')
        for _ in range(16):counts.reserve('forward')
        for kind in ('load','forward','derivative'):
            with self.assertRaises(ValueError):counts.reserve(kind)
        with self.assertRaises(ValueError):authority.read_release('')
        self.assertEqual(sum(support.GROUP_CAPS.values())+65536,19628032)
        self.assertFalse({'torch','transformers','tokenizers','safetensors','numpy'}&{n.split('.')[0] for n in sys.modules})
if __name__=='__main__':unittest.main(verbosity=2)
