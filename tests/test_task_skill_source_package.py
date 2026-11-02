import hashlib
import io
import json
from pathlib import Path
import unittest
import zipfile
from src.composite_task_skill_source_package import frozen_source_package


def package(extra=None,wrong_hash=False):
    raw=b'public source fixture; no scientific execution'
    mf={'source_commit':'a'*40,'files':{'source/unit.txt':('b'*64 if wrong_hash else hashlib.sha256(raw).hexdigest())}}
    memory=io.BytesIO()
    with zipfile.ZipFile(memory,'w') as z:
        z.writestr('source/unit.txt',raw)
        if extra:z.writestr(extra,b'public extra')
        z.writestr('TASK_TRAINING_PACKAGE_FILES.json',json.dumps(mf))
    return memory.getvalue()


class FrozenSourcePackageTests(unittest.TestCase):
    def test_owned_temporary_source_is_complete_and_removed_after_read(self):
        raw=package()
        with frozen_source_package(raw,hashlib.sha256(raw).hexdigest(),'a'*40) as (root,mf):
            self.assertTrue((root/'unit.txt').is_file());self.assertEqual(len(mf['files']),1)
            owned=Path(root)
        self.assertFalse(owned.exists())

    def test_changed_source_member_extra_or_escaped_member_cannot_be_accepted(self):
        for raw in [package('source/extra.txt'),package('../escape.txt'),package(wrong_hash=True)]:
            with self.assertRaises((PermissionError,ValueError)):
                with frozen_source_package(raw,hashlib.sha256(raw).hexdigest(),'a'*40):pass
        raw=package()
        with self.assertRaises(PermissionError):
            with frozen_source_package(raw,'0'*64,'a'*40):pass
        with self.assertRaises(PermissionError):
            with frozen_source_package(raw,hashlib.sha256(raw).hexdigest(),'c'*40):pass


if __name__=='__main__':unittest.main()
