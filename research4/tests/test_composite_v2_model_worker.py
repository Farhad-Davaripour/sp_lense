from contextlib import nullcontext
from pathlib import Path
import tempfile,unittest
from src import composite_v2_model_worker as worker
from test_composite_owned_artifacts import Store


class ModelWorkerTests(unittest.TestCase):
    commit='a'*40
    def fixture(self):
        store=Store();workspace={'artifact_volume':'/Volumes/unit/research4/artifacts'}
        context={'workspace':workspace,'configuration':{'bounds':{'worker_seconds':2400,'export_reserve_seconds':180,
                 'maximum_backbone_forwards_per_run':30000}},'stage':{'phase':'qualification',
                 'destination':workspace['artifact_volume']+'/composite_controller_v1/model_generalized_v2/'+self.commit+'/qualification_01'}}
        services=worker.Services();services.clock=lambda:0;services.authorize=lambda *a:(context,store,[],{'public_fixture':True})
        services.alarm=lambda *a:nullcontext();services.runtime=lambda:{'public_fixture_not_GPU':True};services.resources=lambda:None
        services.release=lambda *a:None
        def load(context,owner,guard,forward):
            guard();forward('attempt');forward('complete');return object()
        services.load=load
        def fit(actor,rows,cfg,stage,output,guard,progress,persist):
            (Path(output)/'behavior.pt').write_bytes(b'public fixture, not a model checkpoint')
            persist('PUBLIC_UNIT_EVENT.json',{'fixture':True})
            return {'optimizer_updates':1,'full_fit_completed':False,'fixture':True}
        services.fit=fit
        return services,store

    def test_no_pretrained_model_or_scientific_admission_from_service_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            services,store=self.fixture();value=worker.run('public-fixture-root',self.commit,services,scratch_base=directory)
            self.assertEqual(value['status'],'complete');self.assertTrue(value['public_fixture_test_mode'])
            self.assertFalse(value['scientific_qualification_admitted']);self.assertFalse(value['full_fit_completed'])
            self.assertTrue(value['cleanup']['owned_scratch_removed']);self.assertFalse(value['provider_compute_release_verified'])
            prefix='model_generalized_v2/'+self.commit+'/qualification_01/'
            self.assertIn(prefix+'raw/behavior.pt',store.files);self.assertIn(prefix+'evidence.zip',store.files)

    def test_model_or_release_failure_still_exports_raw_failure_receipt(self):
        for phase in ['load','fit','release']:
            with self.subTest(phase=phase),tempfile.TemporaryDirectory() as directory:
                services,store=self.fixture()
                def fail(*a,**k):raise RuntimeError('synthetic '+phase+' failure')
                setattr(services,phase,fail)
                value=worker.run('public-fixture-root',self.commit,services,scratch_base=directory)
                self.assertEqual(value['status'],'failed');self.assertTrue(value['export']['durable_complete'])
                self.assertFalse(value['scientific_qualification_admitted'])

    def test_failed_artifact_store_retains_scratch_not_false_cleanup_or_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            services,store=self.fixture();prefix='model_generalized_v2/'+self.commit+'/qualification_01/'
            # Ownership itself succeeds; subsequent artifact writes and archive fail.
            original=store.put
            def put(name,data):
                if not name.endswith('OUTPUT_OWNER.json'):raise RuntimeError('synthetic storage outage')
                return original(name,data)
            store.put=put
            value=worker.run('public-fixture-root',self.commit,services,scratch_base=directory)
            self.assertEqual(value['status'],'export_unconfirmed');self.assertFalse(value['owned_scratch_removed'])
            self.assertFalse(value['model_result_preserved_durably']);self.assertTrue(Path(value['export']['retained_scratch']).exists())
            self.assertFalse(value['scientific_qualification_admitted'])


if __name__=='__main__':unittest.main()
