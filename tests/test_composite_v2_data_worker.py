from contextlib import nullcontext
import json,tempfile,unittest
from pathlib import Path
from src import composite_v2_data_worker as worker
from src.composite_exports import verify_evidence_archive
from src.composite_v2_data_contract import source_fingerprint
from src.io_utils import dump,read


class DataWorkerTests(unittest.TestCase):
    def fixture(self,base):
        root=base/'source';(root/'inputs').mkdir(parents=True);(root/'config').mkdir()
        dump(root/'inputs/MANIFEST.json',{})
        dump(root/'config/composite_generalization_amendment_v2.json',{})
        for name in ['composite_generalized_world','composite_generalized_teacher','composite_generalized_tasks',
                     'composite_generalized_interface','composite_generalized_audit','composite_generalized_data',
                     'composite_design_v2','composite_design_v2_native','composite_generalized_novelty',
                     'composite_world','composite_interface','native27_tools']:
            path=root/'src'/(name+'.py');path.parent.mkdir(exist_ok=True);path.write_text('# public synthetic fixture\n')
        context={'destination':str(base/'output'),'execution_workspace':{'host':'synthetic'},'source_commit':'a'*40,
                 'configuration':{},'authorization_sha256':'b'*64,'plan':{'identity_material':'public-fixture-only'}}
        services=worker.Services();services.authorize=lambda *a:context;services.clock=lambda:0
        services.alarm=lambda *a:nullcontext();services.tokenizers=lambda *a:(None,None)
        def generate(root,commit,role,seed,folder,persist):
            Path(folder).mkdir();persist('EPISODES.json',[{'public_fixture':True}]);persist('DETECTION_ROWS.json',[{'public_fixture':True}])
            return {'role':role,'states':1}
        services.generate=generate
        services.audit=lambda episodes,rows,manifest,cfg,reference_replay:{'role':manifest['role'],'complete_states':1,'reference_world_replayed':True}
        services.native=lambda *a:{'status':'complete_native_budget_audit_pass','states':1,'truncated_or_repaired':False}
        services.leakage=lambda *a:{'status':'structural_and_exact_leakage_checks_passed'}
        services.roster=lambda *a:{'role':'public_fixture_not_a_scientific_roster'}
        return root,context,services

    def test_public_service_injection_cannot_create_a_scientifically_admitted_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root,context,services=self.fixture(Path(directory))
            result=worker.run(root,'a'*40,services)
            self.assertEqual(result['status'],'complete');self.assertFalse(result['scientific_dataset_admitted'])
            self.assertTrue(result['public_fixture_test_mode']);self.assertEqual(result['model_forwards'],0)
            verify_evidence_archive(Path(result['archive']).read_bytes())
            with self.assertRaises(FileExistsError):worker.run(root,'a'*40,services)

    def test_failed_later_role_keeps_completed_role_and_partial_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root,context,services=self.fixture(Path(directory));original=services.generate
            def generate(*args,**kwargs):
                if args[2]=='development':raise RuntimeError('synthetic interrupted development')
                return original(*args,**kwargs)
            services.generate=generate;result=worker.run(root,'a'*40,services)
            self.assertEqual(result['status'],'failed');self.assertEqual(result['completed_roles'],['training'])
            self.assertFalse(result['scientific_dataset_admitted']);self.assertTrue((Path(context['destination'])/'training/EPISODES.json').exists())
            verify_evidence_archive(Path(result['archive']).read_bytes())

    def test_writer_refuses_duplicate_escape_and_oversize_without_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            writer=worker.Writer(directory,lambda:None);writer.json('one.json',{'kept':True})
            with self.assertRaises(FileExistsError):writer.json('one.json',{'changed':True})
            with self.assertRaises(ValueError):writer.json('../escape.json',{})
            writer.bytes=worker.BOUNDS['maximum_raw_export_bytes']
            with self.assertRaises(OSError):writer.json('too_big.json',{})
            self.assertEqual(read(Path(directory)/'one.json'),{'kept':True})


if __name__=='__main__':unittest.main()
