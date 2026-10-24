from types import SimpleNamespace
from pathlib import Path
import hashlib,io,tempfile,unittest
from src.composite_store import Store


class StreamingStoreTests(unittest.TestCase):
    def fixture(self):
        values={};calls=[]
        class Files:
            def create_directory(self,path):calls.append(('mkdir',path))
            def upload(self,path,stream,overwrite):
                calls.append(('upload',path,overwrite,hasattr(stream,'read')))
                values[path]=b''.join(iter(lambda:stream.read(1024),b''))
            def download(self,path):return SimpleNamespace(contents=io.BytesIO(values[path]))
        store=Store.__new__(Store);store.local=None;store.prefix='/Volumes/unit/research4/artifacts/composite_controller_v1'
        store.client=SimpleNamespace(files=Files())
        return store,values,calls

    def test_exclusive_file_upload_and_stream_digest_use_existing_transport(self):
        store,values,calls=self.fixture();raw=b'public fixture bytes'*100
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'checkpoint.pt';path.write_bytes(raw)
            store.put_file('model/public.pt',path)
        self.assertEqual(store.file_digest('model/public.pt'),{'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
        self.assertEqual(calls[-1],('upload',store.prefix+'/model/public.pt',False,True))

    def test_stream_methods_keep_path_and_readonly_boundaries(self):
        store,_,_=self.fixture()
        with self.assertRaises(ValueError):store.file_digest('../other')
        store.local=SimpleNamespace()
        with self.assertRaises(PermissionError):store.put_file('anything','not_read')


if __name__=='__main__':unittest.main()
