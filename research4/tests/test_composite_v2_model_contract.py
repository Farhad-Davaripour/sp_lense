import hashlib,json,tempfile,unittest
from pathlib import Path
from src.io_utils import dump
from src.composite_exports import make_evidence_archive
from src.composite_v2_model_contract import validate_model_metadata,load_training_export


class ModelMetadataTests(unittest.TestCase):
    def test_missing_actual_data_stage_bindings_fail_before_model_or_provider_use(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'inputs').mkdir()
            dump(root/'RELEASE.json',{'commit':'a'*40,'files':{}});dump(root/'inputs/MANIFEST.json',{})
            with self.assertRaises(PermissionError):validate_model_metadata(root,'a'*40)

    def test_archive_digest_and_fixture_or_partial_export_cannot_admit_training(self):
        with self.assertRaises(PermissionError):load_training_export(b'not_an_archive',{'stage':{'data_export':{'sha256':'0'*64}}})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);raw=root/'raw';raw.mkdir()
            dump(raw/'DATA_RELEASE.json',{'public_fixture':True})
            dump(raw/'RUN_RECEIPT.json',{'status':'complete','scientific_dataset_admitted':False,'public_fixture_test_mode':True})
            archive=root/'evidence.zip';make_evidence_archive(raw,archive)
            data=archive.read_bytes();context={'stage':{'data_export':{'sha256':hashlib.sha256(data).hexdigest()}}}
            with self.assertRaises(PermissionError):load_training_export(data,context)


if __name__=='__main__':unittest.main()
