"""Explicit runtime and artifact bindings shared by the composite stages."""
import io
import base64
import os
import subprocess
import threading
from pathlib import Path
from .io_utils import read, sha

KAGGLE_RUNTIME_DIRECTORIES = ['/kaggle/input', '/kaggle/working', '/kaggle/temp']
KAGGLE_INPUT_BASES = ['/kaggle/input', '/kaggle/working/research4_inputs']
KAGGLE_OUTPUT_BASE = '/kaggle/working/research4_campaign/runs'


def run_logged(command, cwd, environment, log, timeout, exit_observer=None):
    """Stream one owned child and enforce its external deadline; never retry."""
    process=None
    try:
        with Path(log).open('x') as stream:
            process = subprocess.Popen(command, cwd=cwd, env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,text=True,bufsize=1)
            def copy_output():
                for line in process.stdout:
                    stream.write(line); stream.flush(); print(line, end='', flush=True)
            reader=None; timed_out=False; code=None
            try:
                reader = threading.Thread(target=copy_output, daemon=True)
                reader.start()
                try: code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired: timed_out=True
            finally:
                try:
                    if process.poll() is None:
                        try: process.kill()
                        except ProcessLookupError: pass
                        code=process.wait(timeout=15)
                finally:
                    observed=process.poll()
                    if exit_observer is not None:
                        exit_observer({'child_started':True,'pid':process.pid,'exit_code':observed,
                                       'child_exit_confirmed':observed is not None})
                    if reader is not None and reader.ident is not None: reader.join(timeout=15)
                    if observed is not None and (reader is None or not reader.is_alive()): process.stdout.close()
            return {'exit_code': code, 'timed_out': timed_out}
    finally:
        if process is None and exit_observer is not None:
            exit_observer({'child_started':False,'pid':None,'exit_code':None,'child_exit_confirmed':True})


def platform():
    value = os.environ.get('RESEARCH4_EXECUTION_PLATFORM', 'databricks')
    if value not in ['databricks', 'kaggle']:
        raise ValueError('Unregistered Research 4 execution platform')
    return value


def require_scientific_runtime():
    from .composite_bounded_behavior_runtime import model_calls_forbidden
    if model_calls_forbidden():
        raise RuntimeError('Inert record generation/replay is not a pretrained-model runtime permit')
    if platform() == 'databricks':
        if not (os.environ.get('MLFLOW_RUN_ID') or
                os.environ.get('DATABRICKS_RUNTIME_VERSION') or
                os.environ.get('RESEARCH4_SCIENTIFIC_RUNTIME') == 'databricks-verified'):
            raise RuntimeError('Scientific cohort execution belongs on Databricks')
    elif not all(Path(p).is_dir() for p in KAGGLE_RUNTIME_DIRECTORIES):
        raise RuntimeError('The explicitly selected Kaggle runtime is required')


def relative_path(value):
    if not value or value.startswith('/') or '\\' in value or ':' in value:
        raise ValueError('Unsafe artifact path')
    if any(part in ['', '.', '..'] for part in value.split('/')):
        raise ValueError('Unsafe artifact path')
    return value


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative_path(relative)).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Artifact path escaped its frozen root')
    return path


class FrozenInputs:
    """Read registered development bytes on cloud or during an offline audit."""
    def __init__(self, root, manifest_sha256):
        root = Path(root).resolve()
        manifest = root / 'TRANSPORT_MANIFEST.json'
        if sha(manifest) != manifest_sha256:
            raise ValueError('Kaggle artifact transport manifest changed')
        self.root = root
        self.files = read(manifest)['files']

    def path(self, relative):
        relative_path(relative)
        if 'heldout' in relative.lower() or relative not in self.files:
            raise PermissionError('Unregistered or sealed held-out artifact')
        return contained(self.root, relative)

    def get(self, relative):
        data = self.path(relative).read_bytes()
        import hashlib
        if hashlib.sha256(data).hexdigest() != self.files[relative]:
            raise ValueError('Kaggle input bytes differ from their freeze')
        return data

    def json(self, relative):
        import json
        return json.loads(self.get(relative))


class KaggleInputs(FrozenInputs):
    def __init__(self):
        require_scientific_runtime()
        root = Path(os.environ['RESEARCH4_KAGGLE_INPUT_ROOT']).resolve()
        bases = [Path(base).resolve() for base in KAGGLE_INPUT_BASES]
        if not any(root.is_relative_to(base) for base in bases):
            raise ValueError('Kaggle inputs escaped the dedicated input roots')
        super().__init__(root, os.environ['RESEARCH4_KAGGLE_MANIFEST_SHA256'])


def begin_stage(root, commit, destination, mode):
    """Return the execution identity and an exclusive readback-checked writer."""
    require_scientific_runtime()
    if platform() == 'kaggle':
        # The platform supplies a T4 pair. The scientific child must expose and
        # use only the single device admitted by the unchanged stage protocol.
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
            raise ValueError('Kaggle scientific stages require device zero only')
        if mode not in ['detector-rank32-v5']:
            raise ValueError('Only the admitted recognition stage is enabled on Kaggle')
        pin = read(root / 'inputs/EXECUTION_PLATFORM.json')
        if pin.get('platform') != 'kaggle' or pin.get('account') != 'farhad89':
            raise ValueError('Kaggle execution identity differs from its freeze')
        if pin.get('mode') != mode or pin.get('source_commit') != commit:
            raise ValueError('Kaggle source/stage admission differs')
        if Path(pin['input_root']).resolve() != Path(os.environ[
                'RESEARCH4_KAGGLE_INPUT_ROOT']).resolve():
            raise ValueError('Kaggle input root differs from its freeze')
        if pin['transport_manifest_sha256'] != os.environ[
                'RESEARCH4_KAGGLE_MANIFEST_SHA256']:
            raise ValueError('Kaggle transport admission differs')
        KaggleInputs()
        target = Path(destination).resolve()
        base = Path(KAGGLE_OUTPUT_BASE).resolve()
        if not target.is_relative_to(base) or target == base:
            raise ValueError('Kaggle output must stay in its dedicated run directory')
        if str(target) != pin['destination']:
            raise ValueError('Kaggle output differs from its frozen destination')
        from .composite_exports import exclusive_output
        upload=exclusive_output(target,base)

        return {'platform': 'kaggle', 'account': pin['account'],
                'notebook': pin['notebook'], 'mode': mode,
                'physical_GPU_allocation': 'platform_T4_pair',
                'scientific_visible_GPU_count': 1}, upload

    from databricks.sdk import WorkspaceClient
    from .workspace_accounts import verify_runtime_workspace, artifact_volume
    if not os.environ.get('MLFLOW_RUN_ID') or root.resolve() != Path(
            os.environ['CODE_SOURCE_PATH']).resolve():
        raise ValueError('Actual Databricks AIR runtime required')
    client = WorkspaceClient()
    identity = verify_runtime_workspace(root, client.config.host)
    prefix = artifact_volume(host=client.config.host) + '/composite_controller_v1/'
    if not destination.startswith(prefix) or '..' in destination.split('/') or '\\' in destination:
        raise ValueError('Dedicated campaign destination required')
    client.files.create_directory(destination)

    def upload(name, data):
        client.files.upload(destination + '/' + name, io.BytesIO(data), overwrite=False)
        with client.files.download(destination + '/' + name).contents as stream:
            if stream.read() != data:
                raise ValueError('Campaign export readback differs')

    return identity, upload
