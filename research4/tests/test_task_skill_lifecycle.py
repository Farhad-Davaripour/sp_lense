from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from src.composite_task_skill_training import qualification_plan, MODE
from src.composite_owned_artifacts import OwnedArtifacts
from src.composite_task_skill_rescue import rescue_owned


class Store:
    def __init__(self):self.values={}
    def put(self,name,raw):
        if name in self.values:raise FileExistsError(name)
        self.values[name]=raw
    def get(self,name):return self.values[name]


class TaskLifecycleTests(unittest.TestCase):
    def test_owned_task_namespace_is_distinct_and_unknown_phase_is_not_admitted(self):
        with tempfile.TemporaryDirectory() as folder:
            source='a'*40;store=Store()
            with self.assertRaises(PermissionError):
                OwnedArtifacts(store,'task_skill_training_v1/'+source+'/unregistered_01',source,base=folder,task_skill=True)
            owner=OwnedArtifacts(store,'task_skill_training_v1/'+source+'/qualification_01',source,base=folder,task_skill=True)
            owner.json('PUBLIC_FIXTURE.json',{'scientific':False})
            self.assertTrue(owner.finalize()['durable_complete'])
            self.assertTrue(owner.cleanup(True)['owned_scratch_removed'])

    def test_model_failure_preserves_unknown_optimizer_and_exact_callback_prefix(self):
        from src.composite_task_skill_model_worker import run
        with tempfile.TemporaryDirectory() as folder:
            source='b'*40;store=Store();plan=qualification_plan(source)
            class Services:
                clock=staticmethod(time.monotonic)
                def authorize(self,root,commit):return {'stage':plan,'full_config':{}},[{'public_unit':True}]
                def owner(self,context):return OwnedArtifacts(store,'task_skill_training_v1/'+source+'/qualification_01',source,base=folder,maximum_bytes=8*1024**2,task_skill=True)
                def runtime(self,context):return {'public_hardware_fixture':True}
                def resources(self):pass
                def alarm(self,seconds):return nullcontext()
                def load(self,context,owner,guard,forward):
                    forward('attempt');raise RuntimeError('public fake failure during a logical forward')
                def release(self):pass
                def remove_owned_cache(self,owner):return {'owned_download_cache_removed':True}
            result=run('not_a_scientific_root',source,_services=Services())
            self.assertEqual(result['status'],'failed')
            self.assertTrue(result['public_fixture_test_mode'])
            self.assertEqual(result['forward_counts'],{'attempted':1,'completed':0},msg=result.get('error'))
            self.assertIsNone(result['optimizer_updates'])
            self.assertFalse(result['qualification_admitted'])
            self.assertTrue(result['cleanup']['owned_scratch_removed'])

    def test_rescue_never_cleans_an_unconfirmed_child_and_preserves_partial_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);scratch=base/'research4-v2-owned-stage-unit';scratch.mkdir()
            raw=scratch/'raw';raw.mkdir();(raw/'partial.pt').write_bytes(b'public incomplete tensor prefix')
            cache=scratch/'cache';cache.mkdir();(cache/'replicated-weight').write_bytes(b'public cache')
            output=base/'published';output.mkdir();source='c'*40
            owner={'scratch':str(scratch),'source_commit':source,'prefix':'task_skill_training_v1/'+source+'/qualification_01','token':'a'*32}
            (scratch/'OWNER.json').write_text(json.dumps(owner));(output/'OUTPUT_OWNER.json').write_text(json.dumps(owner))
            stage=qualification_plan(source)
            result=rescue_owned(output,stage,{'child_exit_confirmed':None},base=base)
            self.assertEqual(result['status'],'child_exit_unknown_no_cleanup');self.assertTrue(cache.exists())
            result=rescue_owned(output,stage,{'child_exit_confirmed':True},base=base)
            self.assertFalse(cache.exists());self.assertTrue((raw/'partial.pt').exists())
            self.assertEqual((output/'rescue_raw_prefix/partial.pt').read_bytes(),(raw/'partial.pt').read_bytes())
            self.assertIsNone(result['scientific_result']);self.assertFalse(result['counts_exact_at_exit'])

    def test_partial_or_public_raw_model_export_never_becomes_a_qualification(self):
        from src.composite_exports import make_evidence_archive
        from src.composite_task_skill_model_audit import audit_qualification
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=root/'raw';raw.mkdir();source='d'*40
            receipt={'source_commit':source,'mode':MODE,'phase':'qualification','status':'failed',
                     'public_fixture_test_mode':True,'heldout_content_read':False,'forward_counts':None}
            (raw/'RUN_RECEIPT.json').write_text(json.dumps(receipt));archive=root/'evidence.zip'
            make_evidence_archive(raw,archive)
            with patch('src.composite_task_skill_model_audit.validate_model_source',return_value=(qualification_plan(source),{},None)):
                value=audit_qualification(archive.read_bytes(),hashlib.sha256(archive.read_bytes()).hexdigest(),root,source).receipt()
            self.assertFalse(value['complete_training_qualification_audited'])
            self.assertFalse(value['scientific_qualification_admitted'])
            self.assertIsNone(value['recorded_forward_counts'])

    def test_frozen_evaluation_reuses_cleanup_without_claiming_a_fit_or_validation(self):
        from src.composite_task_skill_model_worker import run
        from src.composite_task_skill_inference import MODE as INFERENCE_MODE,stage_prefix
        with tempfile.TemporaryDirectory() as folder:
            source='e'*40;store=Store();plan={**qualification_plan(source),'phase':'task_development',
              'version':INFERENCE_MODE,'objective':'frozen_ordinary_task_development_only','partition':{'index':0}}
            events=[]
            class Services:
                clock=staticmethod(time.monotonic)
                def authorize(self,root,commit):return {'stage':plan,'full_config':{}},[]
                def owner(self,context):return OwnedArtifacts(store,stage_prefix(plan),source,base=folder,
                                                              maximum_bytes=8*1024**2,task_evaluation=True)
                def runtime(self,context):return {'public_hardware_fixture':True}
                def resources(self):pass
                def alarm(self,seconds):return nullcontext()
                def load(self,context,owner,guard,forward):events.append('load');return object()
                def fit(self,actor,rows,cfg,stage,output,guard,progress,persist):
                    self.assert_no_training_rows=not rows;events.append('collect')
                    return {'optimizer_updates':0,'completed_trajectories':2,'scientific_validation_admitted':False}
                def release(self):events.append('release')
                def remove_owned_cache(self,owner):return {'owned_download_cache_removed':True}
            services=Services();result=run('public fixture',source,_services=services,task_evaluation=True)
            self.assertEqual(result['status'],'complete');self.assertEqual(events,['load','collect','release'])
            self.assertTrue(services.assert_no_training_rows);self.assertTrue(result['evaluation_only'])
            self.assertIsNone(result['training_objective']);self.assertEqual(result['optimizer_updates'],0)
            self.assertFalse(result['full_fit_completed']);self.assertFalse(result['qualification_admitted'])
            self.assertTrue(result['public_fixture_test_mode']);self.assertTrue(result['cleanup']['owned_scratch_removed'])
            self.assertTrue(any(n.endswith('/raw/TASK_COLLECTION_RESULT.json') for n in store.values))
            self.assertFalse(any(n.endswith('/raw/TRAINING_RESULT.json') for n in store.values))
            with self.assertRaises(PermissionError):run('public fixture',source,_services=services)

    def test_owned_evaluation_profile_cannot_cross_namespaces_or_mix_authority_flags(self):
        from src.composite_task_skill_inference import MODE as INFERENCE_MODE,stage_prefix
        source='f'*40;plan={'source_commit':source,'version':INFERENCE_MODE,'phase':'task_development',
                           'objective':'frozen_ordinary_task_development_only','partition':{'index':12}}
        self.assertTrue(stage_prefix(plan).endswith('partition_12'))
        for index in [True,-1,13,35,'0']:
            with self.assertRaises(PermissionError):stage_prefix({**plan,'partition':{'index':index}})
        with tempfile.TemporaryDirectory() as folder:
            for flags in [{'task_evaluation':True,'task_skill':True},{'task_evaluation':1},
                          {'task_evaluation':True,'development':True}]:
                with self.assertRaises(PermissionError):OwnedArtifacts(Store(),stage_prefix(plan),source,base=folder,**flags)

    def test_shared_bootstrap_routes_frozen_evaluation_to_its_own_source_and_stage(self):
        import ast
        from src.io_utils import dump,sha
        from src.composite_task_skill_inference import MODE as INFERENCE_MODE
        source='a'*40
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)
            for name in ['input','working','temp']:(base/name).mkdir()
            package=base/'input/package';stage_file=package/'source/inputs/TASK_EVALUATION_STAGE.json'
            stage_file.parent.mkdir(parents=True)
            stage={'version':INFERENCE_MODE,'source_commit':source,'phase':'task_development','scientific_GPU_count':2,
              'optimizer_updates':0,'notebook':'farhad89/research4-task-development-v1',
              'destination':'/kaggle/working/research4_campaign/runs/'+source+'/task_skill_evaluation_v1/partition_00'}
            dump(stage_file,stage)
            manifest=package/'TASK_TRAINING_PACKAGE_FILES.json'
            dump(manifest,{'version':INFERENCE_MODE,'source_commit':source,
                           'files':{'source/inputs/TASK_EVALUATION_STAGE.json':sha(stage_file)}})
            code=(Path(__file__).resolve().parents[1]/'notebooks/composite_task_skill_qualification.py').read_text().replace(
              'SOURCE_COMMIT_PLACEHOLDER',source).replace('PACKAGE_MANIFEST_SHA256_PLACEHOLDER',sha(manifest))
            tree=ast.parse(code);end=next(i for i,node in enumerate(tree.body) if isinstance(node,ast.Try))
            # Redirect provider-shaped paths into a public disposable fixture.
            class FixturePath:
                def __new__(cls,value):
                    text=str(value)
                    return base/text.removeprefix('/kaggle/') if text.startswith('/kaggle/') else Path(value)
            nodes=[n for n in tree.body[:end] if not (isinstance(n,ast.ImportFrom) and n.module=='pathlib')]
            namespace={'Path':FixturePath}
            exec(compile(ast.Module(body=nodes,type_ignores=[]),'<public-bootstrap-fixture>','exec'),namespace)
            self.assertTrue(namespace['evaluation'])
            self.assertEqual(namespace['stage']['optimizer_updates'],0)
            self.assertEqual(namespace['root'],base/'temp/research4_task_evaluation_inputs'/source)
            self.assertFalse((base/'temp/research4_task_training_inputs').exists())
            self.assertEqual(namespace['receipt']['phase'],'task_development')
            self.assertIsNone(namespace['receipt']['model_forwards'])


if __name__=='__main__':unittest.main()
