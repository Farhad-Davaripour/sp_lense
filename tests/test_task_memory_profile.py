from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from src.composite_task_memory_profile import (expected_profile,placement_plan,stress_schedule,allocator_evidence,ALLOCATOR)


class MemoryProfileTests(unittest.TestCase):
    def test_placement_keeps_all64_blocks_and_old_candidate_unchanged(self):
        root=Path(__file__).parents[1];cfg=json.loads((root/'config/composite_controller_rank32_v2.json').read_text())
        prior=json.loads((root/'config/composite_qwen_two_t4_candidate_v1.json').read_text())
        plan=placement_plan(cfg['qwen'],prior,expected_profile())
        layers=[plan['device_map']['model.language_model.layers.'+str(i)] for i in range(64)]
        self.assertEqual([layers.count(0),layers.count(1)],[22,42]);self.assertEqual(prior['first_device_layer_count'],18)
        self.assertFalse(plan['runtime_admitted'])
        changed=expected_profile();changed['rank']=16
        with self.assertRaises(PermissionError):placement_plan(cfg['qwen'],prior,changed)

    def test_schedule_exercises_resident_moments_and_size_switch_without_shortening_fit(self):
        fixture={'records':[{'state_id':'unit_'+str(i).zfill(4),'combined_tokens':i+1} for i in range(1188)]}
        plan=stress_schedule(fixture)
        self.assertEqual(plan['state_ids'],['unit_1187','unit_1186','unit_0000','unit_0001','unit_1187','unit_1186'])
        self.assertEqual(plan['backward_microsteps'],6);self.assertFalse(plan['full_fit_64_update_sampling_changed'])
        fixture['records'].pop()
        with self.assertRaises(PermissionError):stress_schedule(fixture)

    def test_allocator_requires_live_expandable_segments_and_does_no_probe(self):
        class CUDA:
            memory=SimpleNamespace(get_allocator_backend=lambda:'native')
            def memory_snapshot(self):return [{'device':i,'total_size':100,'allocated_size':80,'active_size':80,'is_expandable':True} for i in [0,1]]
            def memory_stats(self,i):return {'inactive_split_bytes.all.current':0,'allocated_bytes.all.current':80,'reserved_bytes.all.current':100}
        env={'PYTORCH_ALLOC_CONF':ALLOCATOR};proof=allocator_evidence(CUDA(),env)
        self.assertEqual(proof['new_probe_allocations'],0);self.assertFalse(proof['actual_training_capacity_qualified'])
        with self.assertRaises(PermissionError):allocator_evidence(CUDA(),{**env,'PYTORCH_CUDA_ALLOC_CONF':ALLOCATOR})
        class DefaultCUDA(CUDA):
            def memory_snapshot(self):return [{**r,'is_expandable':False} for r in super().memory_snapshot()]
        with self.assertRaises(PermissionError):allocator_evidence(DefaultCUDA(),env)

    def test_new_stateful_raw_profile_is_explicit_and_old_limits_stay_closed(self):
        from src.composite_v2_raw_exports import validate_export_plan,STATEFUL_TASK_PROFILE
        from src.composite_exports import make_evidence_archive,verify_evidence_archive
        from src.composite_v2_model_audit import EvidenceFiles
        profile=expected_profile();stage={**profile['bounds'],'raw_evidence_profile':profile['raw_evidence_profile']}
        proof=validate_export_plan(stage,STATEFUL_TASK_PROFILE)
        self.assertEqual(proof['maximum_raw_evidence_bytes'],6*1024**3)
        with self.assertRaises(PermissionError):validate_export_plan(stage)
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);root=base/'raw';root.mkdir();(root/'unit.json').write_text('{"public":true}')
            archive=base/'unit.zip';manifest=make_evidence_archive(root,archive)
            self.assertEqual(verify_evidence_archive(archive),verify_evidence_archive(archive.read_bytes()))
            with self.assertRaises(ValueError):EvidenceFiles(archive,manifest['archive_sha256'],6*1024**3)
            files=EvidenceFiles(archive,manifest['archive_sha256'],6*1024**3,evidence_profile=STATEFUL_TASK_PROFILE)
            try:self.assertTrue(files.json('unit.json')['public'])
            finally:files.close()


if __name__=='__main__':unittest.main()
