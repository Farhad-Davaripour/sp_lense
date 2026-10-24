import json,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from src.io_utils import dump,sha,read
from src import composite_v2_data_contract as contract
from tools import prepare_v2_data as preparation

ROOT=Path(__file__).resolve().parents[1]


class PreparationTests(unittest.TestCase):
    def git(self,repo,*args):
        return subprocess.check_output(['git','-c','maintenance.auto=false','-c','gc.auto=0',*args],cwd=repo,stderr=subprocess.DEVNULL,text=True).strip()

    def test_exact_source_package_is_inert_and_tampering_or_old_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);repo=base/'repo';repo.mkdir();source=repo/'research4';(source/'src').mkdir(parents=True);(source/'config').mkdir()
            for name in ['composite_v2_data_contract','composite_v2_data_worker','composite_generalized_data']:
                (source/'src'/(name+'.py')).write_text('# public fixture source, no production data\n')
            (source/'config/composite_generalization_amendment_v2.json').write_bytes((ROOT/'config/composite_generalization_amendment_v2.json').read_bytes())
            self.git(repo,'init');self.git(repo,'config','user.name','Unit Fixture');self.git(repo,'config','user.email','fixture@example.invalid')
            self.git(repo,'add','research4');self.git(repo,'commit','-m','Public metadata fixture')
            commit=self.git(repo,'rev-parse','HEAD')
            pins=base/'model_inputs';pins.mkdir()
            for name in ['QWEN_PIN.json','MODERNBERT_PIN.json']:dump(pins/name,{'repository':'synthetic_fixture','files':[]})
            dump(pins/'MANIFEST.json',{name:sha(pins/name) for name in ['QWEN_PIN.json','MODERNBERT_PIN.json']})
            human=base/'human.json';dump(human,{'status':'verified_direct_human_generalization_scope','human_user_message_id':'01a16c34-f963-77c0-aee1-d48bd7338bf8','unit_fixture':True})
            with patch.object(preparation,'ORIGINAL_MODEL_MANIFEST',sha(pins/'MANIFEST.json')),patch.object(contract,'ORIGINAL_MODEL_MANIFEST',sha(pins/'MANIFEST.json')),patch('src.composite_frozen_probe.verify_original_model_pin'):
                result=preparation.prepare(repo,commit,pins,human,base/'prepared','azure')
                self.assertFalse(result['scientific_allocation_authorized']);self.assertFalse(result['actual_scientific_cohorts_generated'])
                self.assertIsNone(result['worst_case_cost_CAD']);self.assertFalse(result['paid_submission_enabled'])
                staged=base/'prepared/source';context=contract.validate_metadata(staged,commit)
                self.assertFalse(context['allocation_authorized_by_metadata_validation'])
                self.assertEqual(context['plan']['contracts']['training']['cases'],108)
                self.assertFalse(any('EPISODES' in p.name for p in staged.rglob('*')))
                file=staged/'inputs/WORKSPACE.json';value=read(file);value['host']='https://wrong.invalid';dump(file,value)
                with self.assertRaises(ValueError):contract.validate_metadata(staged,commit)
                with self.assertRaises(FileExistsError):preparation.prepare(repo,commit,pins,human,base/'prepared','azure')


if __name__=='__main__':unittest.main()
