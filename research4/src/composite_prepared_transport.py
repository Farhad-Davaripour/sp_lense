"""Inert complete source/manifest/one-cell notebook transport writing."""
from pathlib import Path
import zipfile
from .io_utils import dump,sha
from .composite_runtime import relative_path


def source_manifest(source,stage,filename):
    source=Path(source);bundle=source.parent
    if Path(filename).name!=filename:raise ValueError('Plain manifest filename required')
    manifest=bundle/filename
    dump(manifest,{'version':stage['version'],'source_commit':stage['source_commit'],'notebook':stage['notebook'],
      'files':{p.relative_to(bundle).as_posix():sha(p) for p in sorted(source.rglob('*')) if p.is_file()}})
    return manifest


def write_transport(output,source,stage,manifest,archive_name,notebook_name,template):
    output=Path(output);source=Path(source);manifest=Path(manifest)
    for name in [archive_name,notebook_name]:
        if Path(name).name!=name:raise ValueError('Plain output filenames required')
    relative_path(template)
    archive=output/archive_name
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(source.parent.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(source.parent).as_posix())
    code=(source/template).read_text(encoding='utf-8-sig').replace(
      'SOURCE_COMMIT_PLACEHOLDER',stage['source_commit']).replace('PACKAGE_MANIFEST_SHA256_PLACEHOLDER',sha(manifest))
    notebook=output/notebook_name
    dump(notebook,{'cells':[{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
      'source':code.splitlines(keepends=True)}],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
      'nbformat':4,'nbformat_minor':5})
    return archive,notebook
