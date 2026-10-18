"""Fetch, verify and rescore the paper's public artifacts without workspace access."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import shutil
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


def validation_packet(root):
    """Restore the public model-visible projection, not private annotation transport."""
    verify(root);root=Path(root)
    values={name:json.loads((root/'datasets/validation24'/name).read_bytes())
            for name in ['cases.json','fixtures.json','validation.json','questions.json']}
    values['training.json']=json.loads((root/'datasets/workflow_gate/training.json').read_bytes())
    if len(values['cases.json'])!=24 or [r['id'] for r in values['cases.json']]!=[r['id'] for r in values['validation.json']]:
        raise ValueError('Released case order differs')
    return values


def replay(root, output):
    verify(root);root=Path(root);output=Path(output)
    if output.exists():raise FileExistsError('Choose a new report directory')
    from src.steering_failure_analysis import analyze
    report=analyze(root/'evidence/validation24',output/'steering_failure_analysis.json')
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
    heldout=metrics(json.loads((root/'datasets/workflow_gate/benchmark.json').read_bytes()),
                    json.loads((root/'evidence/heldout24_gate/PREDICTIONS.json').read_bytes()))
    historical=output/'historical_H24'
    shutil.copytree(root/'evidence/historical_H24',historical)
    from src.engines.fixed_steering.campaign_analysis import analyze as historical_analysis
    old=historical_analysis(historical)
    fields=('condition','family','stratum','scheduled','completed','runtime_incomplete','not_executed',
            'workflow_success','own_grant','inappropriate_actions')
    table=lambda value:[{k:r[k] for k in fields} for r in value['tables']]
    if table(old)!=table(json.loads((historical/'EXPECTED_ANALYSIS.json').read_bytes())):
        raise ValueError('Historical 27B confirmation rescore differs')
    from src.research2_classifier import agreement
    truth=json.loads((root/'datasets/binary_gate/confirmation.json').read_bytes())
    binary_predictions=json.loads((root/'evidence/binary_gate/CLASSIFIER_PREDICTIONS.json').read_bytes())['probabilities']
    binary=agreement(truth,[binary_predictions[r['id']] for r in truth])
    from src.research2_compare import key, guarded_accept, validate_pair, summarize
    binary_models={}
    for model in ['m08_s42','m2_s42']:
        folder=root/'evidence/binary_inference'/model
        base=json.loads((folder/'base.json').read_bytes())
        candidates={key(r):r for r in json.loads((folder/'candidate.json').read_bytes())}
        if len(base)!=256 or len({key(r) for r in base})!=256:raise ValueError('Missing binary views')
        selected=[]
        for b in base:
            candidate=candidates.get(key(b))
            if candidate:validate_pair(b,candidate,'adaptive')
            selected.append(candidate if candidate and guarded_accept(b,candidate,binary_predictions[b['case_id']]) else b)
        if selected!=json.loads((folder/'new_gate_final.json').read_bytes()):raise ValueError('Binary gate composition differs')
        binary_models[model]=summarize(base,selected)
        if binary_models[model]!=json.loads((folder/'RESULT.json').read_bytes())['new_classifier_gate']:
            raise ValueError('Binary workflow counts differ')
    from src.engines.evaluation.screen import impossible
    screens={}
    for model in ['08','2']:
        folder=root/'evidence/screens'/model
        cases=[json.loads(p.read_bytes()) for p in sorted((folder/'cases').glob('*.json'))]
        evidence=[{'parse_failures':sum(bool(t.get('parse_error')) for t in c['initial_turns']+c['post_turns']),
                   'truncated_turns':sum(bool(t['truncated']) for t in c['initial_turns']+c['post_turns']),
                   'rejected_calls':c['current_metrics']['rejected_calls']} for c in cases]
        measured={'started_cases':len(cases),'completed_cases':sum(c['completed'] for c in cases),
                  'partial_cases':sum(not c['completed'] for c in cases),'unexecuted_cases':16-len(cases)}
        saved=json.loads((folder/'SCREEN_RESULT.json').read_bytes())
        if any(saved[k]!=v for k,v in measured.items()) or impossible(evidence)!=saved['fail_fast_proof']:
            raise ValueError('Small-model stopped-screen proof differs')
        screens[model]=measured
    result={'status':'PUBLIC_SAVED_EVIDENCE_REPLAY_VERIFIED','validation24_gate':gate,'historical_heldout24_gate':heldout,'matched_validation24':methods,
            'historical_27B_all96_complete':old['all_96_rollouts_complete'],
            'binary_gate_confirmation':binary,'binary_models':binary_models,'small_model_screens':screens,
            'new_model_forwards':0,'optimizer_updates':0,'fresh_neural_reproduction':False,
            'historical_holdout_reuse_disclosed':True}
    write_json(output/'RESULT.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['fetch','verify','replay'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();print(json.dumps(fetch(a.root) if a.stage=='fetch' else verify(a.root) if a.stage=='verify' else replay(a.root,a.output)))


if __name__=='__main__':main()
