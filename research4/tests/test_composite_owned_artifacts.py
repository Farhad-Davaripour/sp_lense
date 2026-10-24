import tempfile,unittest
from pathlib import Path
from src.composite_owned_artifacts import OwnedArtifacts,ExportUnconfirmed
from src.io_utils import dump


class Store:
    def __init__(self):self.files={};self.puts={};self.fail=set();self.after_write=set()
    def put(self,name,data):
        self.puts[name]=self.puts.get(name,0)+1
        if name in self.fail:raise RuntimeError('synthetic transfer failed')
        if name in self.files:raise FileExistsError(name)
        self.files[name]=data
        if name in self.after_write:raise RuntimeError('synthetic uncertain response after write')
    def get(self,name):return self.files[name]


class OwnedExportTests(unittest.TestCase):
    commit='a'*40
    prefix='model_generalized_v2/'+commit+'/qualification_01'
    def owner(self,base,store=None,**kwargs):return OwnedArtifacts(Store() if store is None else store,self.prefix,self.commit,base=base,**kwargs)

    def test_unknown_put_is_reconciled_without_a_duplicate_write(self):
        with tempfile.TemporaryDirectory() as directory:
            store=Store();owner=self.owner(directory,store);path=self.prefix+'/raw/item.json';store.after_write.add(path)
            self.assertTrue(owner.json('item.json',{'public_fixture':True})['durable'])
            result=owner.finalize();self.assertTrue(result['durable_complete']);self.assertEqual(store.puts[path],1)
            with self.assertRaises(PermissionError):owner.cleanup(False)
            self.assertTrue(owner.cleanup(True)['owned_scratch_removed'])

    def test_archive_can_preserve_raw_after_one_individual_transfer_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            store=Store();owner=self.owner(directory,store);path=self.prefix+'/raw/item.json';store.fail.add(path)
            with self.assertRaises(ExportUnconfirmed):owner.json('item.json',{'public_fixture':True})
            result=owner.finalize();self.assertTrue(result['archive_complete']);self.assertTrue(result['durable_complete'])
            self.assertFalse(result['raw_files_complete']);self.assertEqual(store.puts[path],1)
            owner.cleanup(True)

    def test_undurable_exports_retain_owned_scratch_and_never_claim_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            store=Store();owner=self.owner(directory,store)
            owner.json('item.json',{'public_fixture':True});store.fail.add(self.prefix+'/evidence.zip')
            # Simulate storage loss after initial write; already-confirmed files
            # are known bytes, but a newly written terminal artifact remains unknown.
            store.fail.add(self.prefix+'/raw/terminal.json')
            owner.json('terminal.json',{'status':'synthetic_failure'},terminal=True,require_durable=False)
            result=owner.finalize();self.assertFalse(result['durable_complete']);self.assertTrue(owner.scratch.exists())
            with self.assertRaises(PermissionError):owner.cleanup(True)

    def test_changed_marker_or_external_checkpoint_overflow_cannot_delete_data(self):
        with tempfile.TemporaryDirectory() as directory:
            owner=self.owner(directory,maximum_bytes=1024,terminal_reserve=128)
            (owner.raw/'external_checkpoint.bin').write_bytes(b'x'*2048)
            with self.assertRaises(OSError):owner.json('extra.json',{})
            self.assertFalse(owner.finalize()['durable_complete']);self.assertTrue(owner.scratch.exists())
            with self.assertRaises(PermissionError):owner.cleanup(True)
        with tempfile.TemporaryDirectory() as directory:
            owner=self.owner(directory);owner.json('item.json',{});owner.finalize()
            dump(owner.scratch/'OWNER.json',{'wrong_owner':True})
            with self.assertRaises(PermissionError):owner.cleanup(True)
            self.assertTrue(owner.scratch.exists())

    def test_unknown_prefix_escape_and_duplicate_finalization_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(PermissionError):OwnedArtifacts(Store(),'../other',self.commit,base=directory)
            owner=self.owner(directory)
            with self.assertRaises(ValueError):owner.json('../escape.json',{})
            owner.json('item.json',{});owner.finalize()
            with self.assertRaises(PermissionError):owner.finalize()

    def test_mutating_a_published_local_artifact_is_not_false_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            owner=self.owner(directory);owner.json('item.json',{'first':True})
            (owner.raw/'item.json').write_text('{"changed":true}')
            with self.assertRaises(ExportUnconfirmed):owner.finalize()
            self.assertTrue(owner.scratch.exists())


if __name__=='__main__':unittest.main()
