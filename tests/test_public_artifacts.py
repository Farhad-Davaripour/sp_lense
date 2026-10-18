"""Published archives cannot escape their extraction root."""
from pathlib import Path
import tempfile
import unittest
from src.public_artifacts import member


class PublicArtifactTests(unittest.TestCase):
    def test_untrusted_archive_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            for name in ['../token','/absolute','C:/credential','a/../../secret',r'a\secret']:
                with self.assertRaises(ValueError):member(root,name)
            self.assertEqual(member(root,'adapters/example/config.json'),Path(root).resolve()/'adapters/example/config.json')


if __name__=='__main__':unittest.main()
