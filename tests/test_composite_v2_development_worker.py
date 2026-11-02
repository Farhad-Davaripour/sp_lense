from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import tempfile,unittest
from src import composite_v2_development_worker as worker
from src.composite_v2_development_contract import COLLECTION_BOUNDS,TRANSFER_BOUNDS
from src.composite_v2_detector_transfer import TransferEvidence,bound_transfer
from src.composite_v2_development_audit import PartitionEvidence,DevelopmentAttestation,bound_attestation,combine_partitions
from test_composite_owned_artifacts import Store


class DevelopmentWorkerTests(unittest.TestCase):
    commit='a'*40
    def fixture(self,phase='development_partition'):
        store=Store();workspace={'artifact_volume':'/Volumes/unit/research4/artifacts'}
        leaf='partition_00' if phase=='development_partition' else 'detector_transfer_01'
        stage={'phase':phase,'destination':workspace['artifact_volume']+'/composite_controller_v1/development_generalized_v2/'+self.commit+'/'+leaf,
               'data_release_sha256':'b'*64}
        context={'root':'public-fixture','source_commit':self.commit,'workspace':workspace,'stage':stage,
                 'bounds':dict(COLLECTION_BOUNDS if phase=='development_partition' else TRANSFER_BOUNDS),'package':{'public_fixture':True}}
        services=SimpleNamespace(clock=lambda:0,metadata=lambda *a:(context,store),runtime=lambda:{'public_fixture_not_hardware':True},
            resources=lambda:None,alarm=lambda *a:nullcontext(),inputs=lambda *a:{'release':{'public_fixture':True}},
            load=lambda *a:object(),release=lambda:None)
        services.integrity=lambda model,owner,when,transfer:owner.json('PUBLIC_INTEGRITY_'+when+'.json',{'public_fixture':True})
        def collect(context,model,data,guard,persist,verify):
            verify()
            if phase=='detector_transfer':return [{'state_id':'public'}]
            records=[{'scene':{'case_id':'public'},'condition':arm} for arm in ['base','always_on','initial_gate','learned_dynamic','oracle_dynamic']]
            for r in records:persist('traces/public/'+r['condition']+'.json',r)
            persist('COLLECTION_RECEIPT.json',{'observed_trajectories':5,'missing':[]})
            return records
        services.collect=collect
        return services,store,context

    def test_public_service_execution_exports_without_genuine_model_or_admission(self):
        for phase in ['development_partition','detector_transfer']:
            with self.subTest(phase=phase),tempfile.TemporaryDirectory() as folder:
                services,store,_=self.fixture(phase)
                value=worker.run('public-fixture',self.commit,services,scratch_base=folder)
                self.assertEqual(value['status'],'complete');self.assertTrue(value['public_fixture_test_mode'])
                self.assertFalse(value['scientific_validation_admitted']);self.assertFalse(value['runtime_capacity_qualified'])
                self.assertFalse(value['provider_compute_release_verified']);self.assertTrue(value['cleanup']['owned_scratch_removed'])
                self.assertTrue(any(name.endswith('/evidence.zip') for name in store.files))

    def test_failure_keeps_completed_traces_and_unknown_missing_outcomes(self):
        with tempfile.TemporaryDirectory() as folder:
            services,store,_=self.fixture()
            def fail(context,model,data,guard,persist,verify):
                persist('traces/public/base.json',{'public_fixture':True})
                persist('COLLECTION_RECEIPT.json',{'observed_trajectories':1,'missing':['public/other']})
                verify();raise RuntimeError('synthetic later-control failure')
            services.collect=fail
            value=worker.run('public-fixture',self.commit,services,scratch_base=folder)
            self.assertEqual(value['status'],'failed');self.assertEqual(value['observed_trajectories'],1)
            self.assertTrue(any(name.endswith('/raw/traces/public/base.json') for name in store.files))
            self.assertFalse(value['scientific_validation_admitted'])

    def test_storage_uncertainty_retains_owned_scratch_and_cannot_retry_science(self):
        with tempfile.TemporaryDirectory() as folder:
            services,store,_=self.fixture();original=store.put
            def put(name,data):
                if not name.endswith('OUTPUT_OWNER.json'):raise OSError('synthetic storage outage')
                return original(name,data)
            store.put=put
            value=worker.run('public-fixture',self.commit,services,scratch_base=folder)
            self.assertEqual(value['status'],'export_unconfirmed');self.assertFalse(value['owned_scratch_removed'])
            self.assertTrue(Path(value['export']['retained_scratch']).exists())

    def test_caller_attestations_and_empty_controls_cannot_admit_results(self):
        for cls,args in [(TransferEvidence,({},)),(PartitionEvidence,({},[],{},{},None)),(DevelopmentAttestation,({},))]:
            with self.assertRaises(PermissionError):cls(*args)
        self.assertIsNone(bound_transfer({'admitted':True},self.commit,{}))
        self.assertEqual(bound_attestation({'runtime_environment_enforced':True},self.commit,'b'*64,'c'*64,{}),{})
        from test_composite_v2_development_contract import metadata_roster
        from src.composite_v2_development_contract import partition_plan
        rows,design=metadata_roster();plan=partition_plan(rows,design)
        result=combine_partitions([],self.commit,{'development_roster':rows,'data_release_sha256':'b'*64},plan,None)
        self.assertFalse(result['complete']);self.assertEqual(len(result['missing']),180)
        self.assertIsNone(result['attestation']);self.assertIsNone(result['model_integrity'])


if __name__=='__main__':unittest.main()
