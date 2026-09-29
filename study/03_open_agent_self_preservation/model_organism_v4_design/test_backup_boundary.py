"""The backup helper must reject a second directory on the same drive."""
import os
import unittest
from pathlib import Path

from backup_verified_archive import require_separate_drive


@unittest.skipUnless(os.name == 'nt', 'Drive-separation check uses Windows paths')
class BackupBoundary(unittest.TestCase):
    def test_same_drive_is_not_durable_backup(self):
        with self.assertRaises(RuntimeError):
            require_separate_drive(Path('C:/primary/run.tar.gz'),
                                   Path('C:/another-folder'))


if __name__ == '__main__':
    unittest.main()
