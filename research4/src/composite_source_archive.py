"""Inert exact-commit source archive, shared across prepared execution profiles."""
import io,re,subprocess,tarfile
from pathlib import Path
from .io_utils import dump,sha


def archive_source(repo,commit,target):
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Exact commit required')
    target=Path(target);target.mkdir(parents=True,exist_ok=False)
    raw=subprocess.check_output(['git','archive',commit+':research4'],cwd=repo)
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive.getmembers():
            destination=(target/member.name).resolve()
            if not destination.is_relative_to(target.resolve()):raise ValueError('Unsafe source member')
            if member.isdir():destination.mkdir(parents=True,exist_ok=True)
            elif member.isfile():
                data=archive.extractfile(member).read()
                if destination.suffix in ['.py','.json','.yaml','.yml','.txt','.md']:
                    data=data.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')
                destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)
            else:raise ValueError('Source links are not admitted')
    release={'commit':commit,'deployment_text_line_endings':'CRLF (original study freeze)',
        'files':{file.relative_to(target).as_posix():sha(file) for folder in ['src','config','ddl']
                 for file in (target/folder).rglob('*') if file.is_file()}}
    dump(target/'RELEASE.json',release);return release
