import copy
import unittest
from src.composite_frozen_partial import replay_prefix,memory_diagnosis
from src.composite_audit import audit_trajectory,audit_trajectory_prefix
import test_composite_frozen_base_audit as base_fixtures


class PartialAuditTests(unittest.TestCase):
    def test_incomplete_prefix_replays_but_cannot_be_credited_as_a_terminal_trajectory(self):
        files,context,tokenizers,values,raw=base_fixtures.BaseAuditTests().fixture()
        result=replay_prefix(files,context,tokenizers,{3},2)
        self.assertTrue(result['prefix_replay']['prefix_replay_pass'])
        self.assertFalse(result['prefix_replay']['trajectory_complete'])
        self.assertEqual(result['completed_trajectories'],0)
        self.assertEqual(result['native_complete_responses'],2)
        self.assertEqual(result['fully_accounted_prefix_forwards'],9)
        self.assertFalse(result['full_validation_admitted'])
        case=context['evaluation_plan']['case_ids'][0]
        prefix=copy.deepcopy(values['trajectories/'+case+'.json'])
        prefix['trace']=prefix['trace'][:2];prefix['outcome']=prefix['trace'][-1]['outcome']
        prefix['terminal_memory']=prefix['trace'][-1]['observation_after']['memory']
        self.assertTrue(audit_trajectory_prefix(prefix,.9,*tokenizers,{3},'reference',{'detector':2048,'actor':6144})['prefix_replay_pass'])
        with self.assertRaisesRegex(ValueError,'terminal trajectory'):
            audit_trajectory(prefix,.9,*tokenizers,{3},'reference',{'detector':2048,'actor':6144})
        prefix['trace'][0]['reply']={'invented':'state'}
        with self.assertRaises(ValueError):audit_trajectory_prefix(prefix,.9,*tokenizers,{3})

    def test_memory_diagnosis_distinguishes_held_tensors_reserved_bytes_and_unmeasured_reclamation(self):
        samples=[]
        for index in range(3):
            samples.append({'joint_device_memory':[{'device':device,'allocated_bytes':1000+device,
                'reserved_bytes':2000+index*100+device,'free_bytes':4000-index*100} for device in [0,1]]})
        result=memory_diagnosis(samples)
        self.assertTrue(result['allocator_retention_supported'])
        self.assertFalse(result['successful_reclaimability_measured'])
        self.assertFalse(result['CUDA_OOM_demonstrated'])
        self.assertIsNone(result['failing_device_and_exact_failing_free_bytes'])
        changed=copy.deepcopy(samples);changed[-1]['joint_device_memory'][0]['allocated_bytes']+=100
        self.assertFalse(memory_diagnosis(changed)['allocator_retention_supported'])
        with self.assertRaises(ValueError):memory_diagnosis([])


if __name__=='__main__':unittest.main()
