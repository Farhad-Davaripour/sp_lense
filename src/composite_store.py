"""Dedicated Volume transport and pinned model verification for AIR workers."""
import io,json,hashlib
from pathlib import Path
from .io_utils import read,dump,sha,assert_namespace
from .composite_data import require_databricks
from .workspace_accounts import artifact_volume
from .composite_runtime import platform,KaggleInputs


def download_model(pin,cache,metadata_only=False):
    require_databricks()
    from huggingface_hub import snapshot_download
    chosen=[f for f in pin['files'] if not metadata_only or not f['name'].endswith('.safetensors')]
    path=Path(snapshot_download(pin['repository'],revision=pin['revision'],cache_dir=str(cache),allow_patterns=[f['name'] for f in chosen],token=False,max_workers=4))
    return path,verify_model_files(pin,path,metadata_only)


def verify_model_files(pin,path,metadata_only=False):
    """Shared byte verification for downloads and the native loader boundary."""
    path=Path(path);chosen=[f for f in pin['files'] if not metadata_only or not f['name'].endswith('.safetensors')]
    proof=[]
    for f in chosen:
        file=path/f['name']
        if file.stat().st_size!=f['size']:raise ValueError('Pinned file length differs')
        value=sha(file)
        if f.get('sha256') and value!=f['sha256']:raise ValueError('Pinned weight digest differs')
        if not f.get('sha256') and f.get('git_blob_id'):
            data=file.read_bytes()
            if hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=f['git_blob_id']:raise ValueError('Pinned metadata blob differs')
        proof.append({'name':f['name'],'sha256':value,'size':file.stat().st_size})
    return proof


class Store:
    def __init__(self,prefix=None):
        self.local=None
        if platform()=='kaggle':
            if prefix is not None:raise ValueError('Kaggle artifact roots require a frozen transport manifest')
            self.local=KaggleInputs();return
        from databricks.sdk import WorkspaceClient
        self.client=WorkspaceClient();expected=artifact_volume(host=self.client.config.host)+'/composite_controller_v1'
        prefix=expected if prefix is None else prefix
        if prefix!=expected and not prefix.startswith(expected+'/'):raise ValueError('Artifact volume belongs to a different workspace or campaign')
        assert_namespace(prefix);self.prefix=prefix
    def path(self,relative):
        if self.local is not None:return str(self.local.path(relative))
        if relative.startswith('/') or '..' in relative.split('/') or '\\' in relative:raise ValueError('Unsafe artifact path')
        return self.prefix+'/'+relative
    def get(self,relative):
        if self.local is not None:return self.local.get(relative)
        with self.client.files.download(self.path(relative)).contents as stream:return stream.read()
    def json(self,relative):return json.loads(self.get(relative))
    def put(self,relative,data):
        if self.local is not None:raise PermissionError('Kaggle scientific inputs are read-only; the stage owns its output writer')
        path=self.path(relative);self.client.files.create_directory(path.rsplit('/',1)[0]);self.client.files.upload(path,io.BytesIO(data),overwrite=False)
        if self.get(relative)!=data:raise ValueError('Artifact readback differs')
    def put_json(self,relative,value):self.put(relative,(json.dumps(value,indent=2)+'\n').encode())
    def file_digest(self,relative):
        """Stream exact readback; do not allocate a multi-GiB bytes copy."""
        import hashlib
        h=hashlib.sha256();size=0
        if self.local is not None:
            stream=self.local.path(relative).open('rb')
        else:stream=self.client.files.download(self.path(relative)).contents
        with stream:
            for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block);size+=len(block)
        return {'sha256':h.hexdigest(),'bytes':size}
    def put_file(self,relative,local_path):
        """Exclusive streaming upload; callers reconcile an ambiguous outcome."""
        if self.local is not None:raise PermissionError('Frozen scientific input transport is read-only')
        path=self.path(relative);self.client.files.create_directory(path.rsplit('/',1)[0])
        with Path(local_path).open('rb') as stream:self.client.files.upload(path,stream,overwrite=False)
