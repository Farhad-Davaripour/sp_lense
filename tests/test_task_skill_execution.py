"""Public metadata/record fixtures; no scientific model or rollout admission."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from test_task_skill_validation import roster
from src.composite_task_skill_execution import execution_plan
from src.composite_task_skill_validation import task_development_plan
from src.composite_task_skill_world import fixture,World
from src import composite_task_skill_collection as collection
from src import composite_task_skill_inference as inference


class TaskExecutionGroupingTests(unittest.TestCase):
    def test_fixed_partitions_keep_all72_pairs_without_extra_model_reloads(self):
        data=roster();original=deepcopy(data);plan=execution_plan(data);old=task_development_plan(data)
        self.assertEqual([len(p['case_ids']) for p in plan['partitions']],[1]+[3]*11+[2])
        self.assertEqual([c for p in plan['partitions'] for c in p['case_ids']],
                         [p['case_ids'][0] for p in old['partitions']])
        self.assertEqual(sum(p['expected_trajectories'] for p in plan['partitions']),72)
        self.assertEqual(max(p['maximum_with_optional_native_parity'] for p in plan['partitions']),23366)
        self.assertFalse(plan['metadata_admits_launch']);self.assertFalse(plan['runtime_throughput_qualified'])
        self.assertEqual(data,original)
        for field in ['maximum_new_tokens','context_tokens','external_ticks']:
            self.assertEqual(plan[field],old[field])

    def test_grouping_cannot_hide_missing_cases_or_remove_an_arm(self):
        for change in ['case','arm']:
            data=roster()
            if change=='case':data['cases'].pop()
            else:data['arms']=['base']
            with self.assertRaises((ValueError,PermissionError)):execution_plan(data)

    def test_grouped_stage_metadata_is_conditional_and_keeps_full_response_budget(self):
        model={'evaluation_source_commit':'a'*40,'development_roster':roster(),'fit_source_commit':'b'*40,
               'fit_run_id':361964150,'checkpoint_sha256':'c'*64,'data_archive_sha256':'d'*64}
        with patch.object(inference,'require_frozen_task_package',return_value=model):
            stage=inference.expected_stage('a'*40,object(),1)
            self.assertEqual(stage['partition']['expected_trajectories'],6)
            self.assertEqual(stage['maximum_backbone_forwards'],23366)
            self.assertEqual((stage['maximum_tokens'],stage['maximum_new_tokens'],stage['external_ticks']),(6144,160,24))
            self.assertFalse(stage['runtime_throughput_qualified']);self.assertEqual(stage['optimizer_updates'],0)
            for index in [True,13,-1]:
                with self.assertRaises(PermissionError):inference.expected_stage('a'*40,object(),index)

    def test_three_public_pairs_start_fresh_and_cannot_implicitly_omit_a_case(self):
        scenes=[fixture(seed='unit:grouped-pair:'+str(i)) for i in range(3)]
        ids=[s.case_id for s in scenes];context=SimpleNamespace(model=object(),stage=lambda:{'partition':{'case_ids':ids}})
        calls=[];saved={}
        def generate(text,on,maximum_new_tokens):
            calls.append(on);return {'actions':[{'tool':'stop','arguments':{}}],'adapter_enabled':on}
        actor=SimpleNamespace(task_evaluation_context=context,cfg={'maximum_tokens':6144,'maximum_new_tokens':160},
                              generate=generate,verify_frozen=lambda:None)
        with ExitStack() as stack:
            for name in ['require_runtime_binding','authorize_scene','verify_frozen_checkpoint']:
                stack.enter_context(patch.object(collection,name,return_value={}))
            stack.enter_context(patch.object(collection,'require_frozen_task_package',return_value={
              'development_roster':{'cases':[asdict(s) for s in scenes]},'checkpoint_sha256':'a'*64}))
            stack.enter_context(patch.object(collection,'World',side_effect=lambda s,evaluation_context:World(s)))
            with self.assertRaises(PermissionError):collection.collect_pair(actor,context,lambda n,v:None)
            results=[collection.collect_pair(actor,context,lambda n,v:saved.__setitem__(n,v),case) for case in ids]
        self.assertEqual(calls,[False,True]*3)
        self.assertEqual(sum(r['completed_trajectories'] for r in results),6)
        self.assertEqual([r['case_id'] for r in results],ids)
        self.assertTrue(all(o['outcome']['clock']==1 for r in results for o in r['outcomes']))
        self.assertTrue(all(not r['scientific_validation_admitted'] for r in results))
        self.assertEqual(len(saved),18)


if __name__=='__main__':unittest.main()
