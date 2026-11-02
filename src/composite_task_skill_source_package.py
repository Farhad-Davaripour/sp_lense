"""Read a pinned source/input package into temporary owned audit storage only."""
from contextlib import contextmanager
import hashlib
import io
from pathlib import Path
import tempfile
import zipfile
from .io_utils import read,sha


def source_transport_limit(manifest_name):
    if manifest_name not in ['TASK_TRAINING_PACKAGE_FILES.json','RISK_DATA_PACKAGE_FILES.json',
            'TOOL_ALIAS_PACKAGE_FILES.json','BOUNDED_BEHAVIOR_DATA_PACKAGE_FILES.json',
            'BOUNDED_BEHAVIOR_FIT_PACKAGE_FILES.json','BOUNDED_ACTOR_EVALUATION_PACKAGE_FILES.json',
            'BOUNDED_SAMPLING_AMENDMENT_PACKAGE_FILES.json','BOUNDED_SAMPLING_FIT_PACKAGE_FILES.json']:
        raise PermissionError('Only registered complete source/input transport formats are readable')
    return 256*1024**2 if manifest_name in ['BOUNDED_BEHAVIOR_FIT_PACKAGE_FILES.json',
            'BOUNDED_ACTOR_EVALUATION_PACKAGE_FILES.json','BOUNDED_SAMPLING_FIT_PACKAGE_FILES.json'] else 128*1024**2


def validate_source_envelope(infos,manifest_name):
    names=[i.filename for i in infos]
    if (len(names)!=len(set(names)) or len(names)>4096
            or sum(i.file_size for i in infos)>source_transport_limit(manifest_name)):
        raise ValueError('Source/input archive envelope differs')


@contextmanager
def frozen_source_package(raw,expected_sha256,source_commit,*,manifest_name='TASK_TRAINING_PACKAGE_FILES.json'):
    source_transport_limit(manifest_name)
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:
        raise PermissionError('Frozen source/input package bytes changed')
    with tempfile.TemporaryDirectory(prefix='r4-task-parent-source-') as folder:
        base=Path(folder)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            infos=z.infolist();names=[i.filename for i in infos]
            validate_source_envelope(infos,manifest_name)
            for info in infos:
                name=info.filename
                if (not name or name.startswith('/') or any(p in ['','..','.'] for p in name.split('/'))
                        or '\\' in name or ':' in name or info.is_dir() or info.flag_bits&1
                        or (info.external_attr>>16)&0o170000==0o120000):
                    raise ValueError('Only safe ordinary source/input members are allowed')
                target=(base/name).resolve()
                if not target.is_relative_to(base):raise ValueError('Source escaped its owned audit folder')
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as output:output.write(z.read(name))
        name=manifest_name;mf=read(base/name)
        if mf.get('source_commit')!=source_commit or set(names)!=set(mf['files'])|{name}:
            raise PermissionError('Exact source and complete package roster required')
        for item,expected in mf['files'].items():
            target=(base/item).resolve()
            if not item.startswith('source/') or not target.is_relative_to(base) or sha(target)!=expected:
                raise PermissionError('Frozen source/input member changed')
        yield base/'source',mf
