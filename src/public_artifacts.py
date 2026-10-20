"""Fetch, verify and rescore the paper's public artifacts without workspace access."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
import yaml
from src.config import ROOT
from src.utils import sha256, write_json
from src.evidence import member, pinned_manifest, verify_files, replay_output, historical_rows, screen_summary

EVIDENCE_ASSETS = {
    'primary': ('datasets.zip', 'evidence.zip', 'MANIFEST.json'),
    'steering': ('diagnostic_outputs.zip', 'MANIFEST.json'),
}
EVIDENCE_PREFIXES = {'primary': ('datasets/', 'evidence/'), 'steering': ('diagnostic/',)}


def profile(name='primary'):
    files={'primary':'public_artifacts.yaml','steering':'public_steering_diagnostic.yaml'}
    return yaml.safe_load((ROOT/'config'/files[name]).read_bytes())


def verify(root, profile_name='primary', scope='full'):
    if scope not in ('evidence', 'full'):
        raise ValueError('Choose evidence or full artifact scope')
    root=Path(root);spec=profile(profile_name)
    manifest=pinned_manifest(root,spec['assets']['MANIFEST.json']['sha256'],
                             'Public manifest differs from the pinned release')
    pins=manifest['files']
    if scope=='evidence':
        pins={name:pin for name,pin in pins.items() if name.startswith(EVIDENCE_PREFIXES[profile_name])}
        if not pins:raise ValueError('The release contains no replay evidence')
    verify_files(root,pins,'Public artifact changed: ')
    return {'status':'PUBLIC_ARTIFACTS_VERIFIED','files':len(pins),'bytes':sum(x['bytes'] for x in pins.values()),
            'model_loads':0,'cloud_workspace_required':False,'scope':scope}


def fetch(root, profile_name='primary', scope='full'):
    """Fetch immutable assets; the CLI defaults to evidence, the existing API to full."""
    if scope not in ('evidence', 'full'):
        raise ValueError('Choose evidence or full artifact scope')
    root=Path(root)
    if root.exists():raise FileExistsError('Choose a new artifact directory')
    spec=profile(profile_name)
    assets=spec['assets'] if scope=='full' else {name:spec['assets'][name] for name in EVIDENCE_ASSETS[profile_name]}
    root.mkdir(parents=True)
    for name,pin in assets.items():
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
            prefix=pin.get('extract_to','')
            if prefix:member(root,prefix)
            with zipfile.ZipFile(dest) as z:
                names=z.namelist()
                if len(names)!=len(set(names)):raise ValueError('Duplicate release members')
                for n in names:
                    member(root,n)
                    p=member(root,prefix+'/'+n if prefix else n)
                    if p.exists():raise ValueError('Overlapping release assets')
                    if z.getinfo(n).file_size>2**30:raise ValueError('Oversized uncompressed member')
                for n in names:
                    p=member(root,prefix+'/'+n if prefix else n);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(n))
            dest.unlink()
        else:dest.replace(root/name)
    return verify(root,profile_name,scope)


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
    root=Path(root);output=replay_output(output,root,ROOT)
    verify(root,scope='evidence')
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
    if historical_rows(old['tables'])!=historical_rows(json.loads((historical/'EXPECTED_ANALYSIS.json').read_bytes())['tables']):
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
    screens={}
    for model in ['08','2']:
        folder=root/'evidence/screens'/model
        measured, _=screen_summary(folder)
        screens[model]=measured
    result={'status':'PUBLIC_SAVED_EVIDENCE_REPLAY_VERIFIED','validation24_gate':gate,'historical_heldout24_gate':heldout,'matched_validation24':methods,
            'historical_27B_all96_complete':old['all_96_rollouts_complete'],
            'binary_gate_confirmation':binary,'binary_models':binary_models,'small_model_screens':screens,
            'new_model_forwards':0,'optimizer_updates':0,'fresh_neural_reproduction':False,
            'historical_holdout_reuse_disclosed':True}
    write_json(output/'RESULT.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['fetch','verify','replay'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--profile',choices=['primary','steering'],default='primary')
    p.add_argument('--scope',choices=['evidence','full'],default='evidence',
                   help='Evidence is sufficient for CPU score replay; full also downloads adapters and fitting arrays')
    a=p.parse_args(argv)
    if a.stage=='replay' and a.profile!='primary':p.error('Use src.steering_diagnostic replay for the steering supplement')
    if a.stage=='replay' and not a.output:p.error('replay requires --output pointing to a new directory outside the checkout and evidence')
    if a.stage=='replay':
        if a.scope=='full':verify(a.root,a.profile,'full')
        result=replay(a.root,a.output)
    else:
        result=fetch(a.root,a.profile,a.scope) if a.stage=='fetch' else verify(a.root,a.profile,a.scope)
    print(json.dumps(result))
    return result


if __name__=='__main__':main()
