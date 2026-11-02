import hashlib,json,tempfile,unittest
from pathlib import Path
import torch
from src.composite_v2_model_audit import EvidenceFiles,tensor_state
from src.composite_exports import make_evidence_archive


class StreamModelAuditTests(unittest.TestCase):
    def test_file_backed_archive_checks_same_members_and_restricted_tensors(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=root/'raw';raw.mkdir()
            tensor={'public_fixture':torch.arange(16,dtype=torch.float32).reshape(4,4)}
            torch.save(tensor,raw/'values.pt');(raw/'receipt.json').write_text(json.dumps({'scientific':False}))
            archive=root/'evidence.zip';proof=make_evidence_archive(raw,archive)
            files=EvidenceFiles(archive,proof['archive_sha256'])
            try:
                path=files.tensor_path('values.pt');self.assertTrue(torch.equal(tensor_state(path)['public_fixture'],tensor['public_fixture']))
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),files.manifest['values.pt'])
                self.assertEqual(files.json('receipt.json'),{'scientific':False})
                with self.assertRaises(PermissionError):files.tensor_path('receipt.json')
            finally:files.close()


if __name__=='__main__':unittest.main()
