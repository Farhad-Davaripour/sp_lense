"""Public metadata/transport fixtures only; no development world is executed."""
import ast
from collections import Counter
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
from src.io_utils import read,dump,sha
from src.composite_frozen_risk_data_profile import expected_profile,MODE,RiskDataContext,planned_roster
from src.composite_frozen_risk_data_reader import RiskDataArchive,RiskDevelopmentData,require_data
from src.composite_frozen_risk_runtime import data_scope,reference_replay_scope,require_scoped_runtime
from src.composite_frozen_risk_data_admission import reserve_data

ROOT=Path(__file__).resolve().parents[1]


class FrozenRiskDataTests(unittest.TestCase):
    def test_fixed_reference_roster_retains_original9_conditions_and5_arms(self):
        cfg=read(ROOT/'config/composite_generalization_amendment_v2.json')
        original=deepcopy(cfg);left=expected_profile('a'*40,cfg);right=expected_profile('b'*40,cfg)
        self.assertEqual(left['development_roster'],right['development_roster'])
        self.assertEqual(len(left['development_roster']),36);self.assertEqual(len(left['original_controls']),5)
        self.assertEqual(left['roles'],['development']);self.assertEqual(left['expected_risk_trajectories'],180)
        self.assertFalse(left['behavior_training_exports_or_model_selection_admitted'])
        self.assertEqual(left['scientific_GPU_count'],0);self.assertEqual(cfg,original)
        groups={s['group_id'] for s in left['development_roster']}
        self.assertEqual(len(groups),4)
        for group in groups:
            self.assertEqual(Counter(s['condition'] for s in left['development_roster'] if s['group_id']==group),Counter(cfg['cohorts']['conditions']))
        changed=deepcopy(cfg);changed['controls'].pop()
        with self.assertRaises(PermissionError):expected_profile('a'*40,changed)

    def test_flags_or_missing_runtime_cannot_create_a_data_or_replay_permit(self):
        self.assertFalse(require_scoped_runtime())
        for value in [None,True,{'risk_development_data_admitted':True,'owned_compute_released':True}]:
            with self.assertRaises(PermissionError):require_data(value)
            with self.assertRaises(PermissionError):
                with data_scope(value):pass
            with self.assertRaises(PermissionError):
                with reference_replay_scope(value):pass
        with self.assertRaises(PermissionError):RiskDataContext('.',{}, {},key=True)
        with self.assertRaises(PermissionError):RiskDataArchive(None,None,None,key=True)
        with self.assertRaises(PermissionError):RiskDevelopmentData({},[],[],key=True)

    def test_recorded_scope_checks_case_roles_and_cannot_add_or_change_actions(self):
        # White-box protocol fixture only: no World is constructed or executed.
        from src.composite_frozen_risk_runtime import _RiskScope,_KEY,_CURRENT,recorded_action_guard
        from src.composite_generalized_world import GeneralizedScene
        from dataclasses import replace
        cfg=read(ROOT/'config/composite_generalization_amendment_v2.json');row=planned_roster(cfg)[0]
        scene=GeneralizedScene(**row);scope=_RiskScope(None,[row],True,_KEY)
        action={'tool':'catalog','arguments':{}};scope.actions={scene.case_id:[action]}
        token=_CURRENT.set(scope)
        try:
            for changed in [replace(scene,role='training'),replace(scene,role='sealed'),replace(scene,hard_stop=25),
                            replace(scene,identity_seed='foreign-public-metadata')]:
                with self.assertRaises(PermissionError):require_scoped_runtime(changed)
            self.assertTrue(require_scoped_runtime(scene))
            with self.assertRaises(PermissionError):recorded_action_guard(scene,{'tool':'stop','arguments':{}})
            self.assertEqual(scope.cursor,0);recorded_action_guard(scene,action);self.assertEqual(scope.cursor,1)
            with self.assertRaises(PermissionError):recorded_action_guard(scene,action)
            with self.assertRaises(PermissionError):require_scoped_runtime(scene)
        finally:_CURRENT.reset(token)
        self.assertFalse(require_scoped_runtime())

    def test_active_fit_blocks_even_cpu_reference_submission_and_releases_guard(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);ledger={'active_owned_submission':{'scriptVersionId':361964150},'candidates':{}}
            dump(root/'SCIENTIFIC_EXECUTION_LEDGER.json',ledger)
            with patch('src.composite_frozen_risk_data_admission.validate_preparation') as validator:
                with self.assertRaises(PermissionError):reserve_data(root,'not_a_real_preparation',{})
                validator.assert_not_called()
            self.assertEqual(read(root/'SCIENTIFIC_EXECUTION_LEDGER.json'),ledger)
            self.assertFalse((root/'COMPUTE_SUBMISSION.lock').exists())

    def test_common_cpu_bootstrap_selects_risk_only_profile_and_copies_transport(self):
        with TemporaryDirectory() as folder:
            base=Path(folder)
            for name in ['input','working']:(base/name).mkdir()
            package=base/'input/package';profile=package/'source/inputs/FROZEN_RISK_DATA_PROFILE.json'
            profile.parent.mkdir(parents=True)
            dump(profile,{'version':MODE,'scientific_GPU_count':0,'roles':['development'],'phase':'risk_data_export',
              'notebook':'farhad89/research4-frozen-risk-data-v1','destination':'/kaggle/working/public_risk_output'})
            manifest=package/'RISK_DATA_PACKAGE_FILES.json'
            dump(manifest,{'version':MODE,'source_commit':'a'*40,'files':{'source/inputs/FROZEN_RISK_DATA_PROFILE.json':sha(profile)}})
            code=(ROOT/'notebooks/composite_task_skill_data.py').read_text(encoding='utf-8').replace(
              'SOURCE_COMMIT_PLACEHOLDER','a'*40).replace('PACKAGE_MANIFEST_SHA256_PLACEHOLDER',sha(manifest))
            tree=ast.parse(code);end=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Try))
            class FixturePath:
                def __new__(cls,value):
                    text=str(value);return base/text.removeprefix('/kaggle/') if text.startswith('/kaggle/') else Path(value)
            nodes=[n for n in tree.body[:end] if not (isinstance(n,ast.ImportFrom) and n.module=='pathlib')]
            namespace={'Path':FixturePath}
            exec(compile(ast.Module(body=nodes,type_ignores=[]),'<public-risk-bootstrap-fixture>','exec'),namespace)
            self.assertEqual(namespace['selected']['worker'],'src.composite_frozen_risk_data_worker')
            self.assertTrue((base/'temp').is_dir())
            self.assertEqual((namespace['root']/'RISK_DATA_PACKAGE_FILES.json').read_bytes(),manifest.read_bytes())
            self.assertEqual(namespace['profile']['roles'],['development'])
            self.assertEqual(namespace['receipt']['model_forwards'],0)

    def test_reference_failure_preserves_raw_prefix_and_never_calls_a_model(self):
        from src.composite_frozen_risk_data_worker import run
        from src.composite_generalized_world import fixture
        from dataclasses import asdict
        import zipfile
        with TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';(source/'inputs').mkdir(parents=True)
            dump(source/'inputs/MANIFEST.json',{});dump(source/'RELEASE.json',{'public_fixture':True})
            profile={'destination':str(root/'public_output'),'development_roster':[asdict(fixture())],'world_version':'public_fixture'}
            context=SimpleNamespace(require_runtime=lambda:None,profile=lambda:profile,cfg=lambda:{})
            with patch('src.composite_frozen_risk_data_worker.validate_source',return_value=context),patch(
              'src.composite_frozen_risk_data_worker.data_scope',return_value=nullcontext()),patch(
              'src.composite_frozen_risk_data_worker.metadata_tokenizers',return_value=(object(),object())),patch(
              'src.composite_frozen_risk_data_worker.teacher_episode',side_effect=RuntimeError('public fixture pre-world failure')):
                value=run(source,'a'*40)
            self.assertEqual(value['status'],'failed');self.assertFalse(value['risk_reference_candidate_complete'])
            self.assertEqual(value['complete_case_exports'],0);self.assertEqual(value['model_forwards'],0)
            self.assertFalse(value['model_weights_downloaded']);self.assertFalse(value['behavior_fitting_or_selection_admitted'])
            with zipfile.ZipFile(value['archive']) as z:
                self.assertIn('FAILURE.json',z.namelist());self.assertIn('RUN_RECEIPT.json',z.namelist())
                self.assertNotIn('RISK_DATA_RELEASE.json',z.namelist())


if __name__=='__main__':unittest.main()
