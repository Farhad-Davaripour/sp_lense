"""Fetch, verify and rescore the paper's public artifacts without workspace access."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import urllib.request
import zipfile
import yaml
from src.config import ROOT
from src.utils import canonical, sha256, write_json


def profile():
    return yaml.safe_load((ROOT/'config/public_artifacts.yaml').read_bytes())


def member(root, name):
    relative=PurePosixPath(name)
    if relative.is_absolute() or not name or '\\' in name or ':' in name or '..' in relative.parts:
        raise ValueError('Unsafe release member')
    path=(Path(root)/name).resolve()
    if not path.is_relative_to(Path(root).resolve()):raise ValueError('Release member escapes destination')
    return path


def verify(root):
    root=Path(root);spec=profile();manifest=root/'MANIFEST.json'
    if sha256(manifest)!=spec['assets']['MANIFEST.json']['sha256']:
        raise ValueError('Public manifest differs from the pinned release')
    pins=json.loads(manifest.read_bytes())['files']
    for name,pin in pins.items():
        p=member(root,name)
        if p.stat().st_size!=pin['bytes'] or sha256(p)!=pin['sha256']:
            raise ValueError('Public artifact changed: '+name)
    return {'status':'PUBLIC_ARTIFACTS_VERIFIED','files':len(pins),'bytes':sum(x['bytes'] for x in pins.values()),
            'model_loads':0,'cloud_workspace_required':False}


def fetch(root):
    root=Path(root)
    if root.exists():raise FileExistsError('Choose a new artifact directory')
    root.mkdir(parents=True);spec=profile()
    for name,pin in spec['assets'].items():
        dest=root/('.download_'+name)
        with urllib.request.urlopen(spec['base_url']+'/'+name,timeout=60) as response,dest.open('xb') as f:
            total=0
            while chunk:=response.read(2**20):
                total+=len(chunk)
                if total>pin['bytes']:raise ValueError('Oversized release asset')
                f.write(chunk)
        if dest.stat().st_size!=pin['bytes'] or sha256(dest)!=pin['sha256']:
            raise ValueError('Release download differs: '+name)
        if name.endswith('.zip'):
            with zipfile.ZipFile(dest) as z:
                names=z.namelist()
                if len(names)!=len(set(names)):raise ValueError('Duplicate release members')
                for n in names:
                    p=member(root,n)
                    if p.exists():raise ValueError('Overlapping release assets')
                    if z.getinfo(n).file_size>2**30:raise ValueError('Oversized uncompressed member')
                for n in names:
                    p=member(root,n);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(n))
            dest.unlink()
        else:dest.replace(root/name)
    return verify(root)


def replay(root, output):
    verify(root);root=Path(root);output=Path(output)
    if output.exists():raise FileExistsError('Choose a new report directory')
    from src.steering_failure_analysis import analyze
    report=analyze(root/'evidence/validation24',output/'steering_failure_analysis.json')
    from src.matched_steering import summarize
    # Summarization writes only to a new output; original released episodes stay immutable.
    methods={}
    for method in report['methods']:
        rows=[json.loads(p.read_bytes()) for p in (root/'evidence/validation24'/method/'episodes').glob('*.json')]
        methods[method]={g:{'cases':len(selected),'successes':sum(r['score']['workflow_success'] for r in selected)}
            for g,selected in [('all',rows),('own',[r for r in rows if not r['score']['negative_control']]),
                               ('controls',[r for r in rows if r['score']['negative_control']])]}
    from src.matched_validation import metrics
    rows=json.loads((root/'datasets/validation24/validation.json').read_bytes())
    predictions=json.loads((root/'evidence/validation24_gate/PREDICTIONS.json').read_bytes())
    gate=metrics(rows,predictions)
    result={'status':'PUBLIC_SAVED_EVIDENCE_REPLAY_VERIFIED','validation24_gate':gate,'matched_validation24':methods,
            'new_model_forwards':0,'optimizer_updates':0,'fresh_neural_reproduction':False,
            'historical_holdout_reuse_disclosed':True}
    write_json(output/'RESULT.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['fetch','verify','replay'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();print(json.dumps(fetch(a.root) if a.stage=='fetch' else verify(a.root) if a.stage=='verify' else replay(a.root,a.output)))


if __name__=='__main__':main()
