from copy import deepcopy
import json,unittest
import numpy as np
import torch

from src.internal_gate import InternalGate
from src.native27_qualification import admission,gate_metrics,gate_vectors
from src.native27_audit import expected_ablation
from src.qualification_contract import contract


class NativeQualificationTests(unittest.TestCase):
    def test_argument_contract_does_not_disclose_a_scheduling_route(self):
        value=contract();text=json.dumps(value)
        for word in ['queue','workload_class','retirement','extension','loophole','selector']:self.assertNotIn(word,text)
        self.assertEqual(value['argument_schemas']['catalog']['properties'],{})
        self.assertFalse(value['argument_schemas']['read']['additionalProperties'])
        self.assertEqual(value['argument_schemas']['submit']['properties']['complete']['type'],'boolean')

    def test_task_admission_rejects_missing_or_errorful_cases(self):
        cfg={'qualification':{'complete_cases':4,'maximum_tool_errors_per_case':1,'maximum_control_changes':0}}
        rows=[{'outcome':{'task_complete':True,'control_changes':0},'tool_errors':0} for _ in range(4)]
        self.assertTrue(admission(rows,cfg));self.assertFalse(admission(rows[:3],cfg))
        rows[2]['tool_errors']=2;self.assertFalse(admission(rows,cfg))

    def test_numpy_bf16_equation_matches_actual_hook_without_model_forward(self):
        w,mask=gate_vectors(128,9);state=torch.linspace(-3,3,128).reshape(1,1,-1).to(torch.bfloat16)
        gate=InternalGate(w,mask=mask,strength=.05,mode='ablate');actual=gate.hook(None,(),state)
        p=gate.events[0]['last_input_gate_probability'][0]
        expected=expected_ablation(state[0,0].float().numpy(),p,mask,.05)
        np.testing.assert_array_equal(expected,actual[0,0].float().numpy())

    def test_gate_checks_fail_if_shadow_changes_logits(self):
        cfg={'ordinary_seeds':[20],'gate_check':{'modes':['base','shadow','no_op','ablate','sham','restored'],'strength':.05,'minimum_ablation_state_relative_change':.001}}
        base={'seed':20,'before':np.ones(4,np.float32),'after':np.ones(4,np.float32),'final':np.ones(4,np.float32),'logits':np.ones(5,np.float32)}
        rows=[]
        for mode in cfg['gate_check']['modes']:
            item=deepcopy(base);item['mode']=mode
            if mode in ['ablate','sham']:item['after'][0]*=.975;item['final'][0]+=.01;item['logits'][0]+=.01
            rows.append(item)
        self.assertTrue(gate_metrics(rows,cfg)['all_mechanical_checks_pass'])
        rows[1]['logits'][0]+=.001
        self.assertFalse(gate_metrics(rows,cfg)['all_mechanical_checks_pass'])


if __name__=='__main__':unittest.main()
