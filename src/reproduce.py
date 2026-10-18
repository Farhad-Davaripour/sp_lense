"""Check a clean checkout or replay retained evidence without loading a model."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import shutil
import zipfile
import yaml
from src.config import ROOT, Settings
from src.utils import sha256, write_json, workspace_client
from src.verify import verify


def profile():
    return yaml.safe_load((ROOT / 'config/reproduction.yaml').read_text(encoding='utf-8'))


def member(root, name):
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()) or '\\' in name:
        raise ValueError('Evidence member escapes its root')
    return path


def validate_evidence(root):
    root = Path(root)
    spec = profile()['evidence']
    if sha256(root / 'MANIFEST.json') != spec['manifest_sha256']:
        raise ValueError('Evidence manifest differs from the frozen reproduction package')
    manifest = json.loads((root / 'MANIFEST.json').read_bytes())
    if len(manifest['files']) != spec['files']:
        raise ValueError('Evidence membership changed')
    for name, pin in manifest['files'].items():
        path = member(root, name)
        if path.stat().st_size != pin['bytes'] or sha256(path) != pin['sha256']:
            raise ValueError('Evidence file changed: ' + name)
    return manifest


def fetch_evidence(destination, settings=None):
    """Download the private, immutable evidence archive through authorized workspace access."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Choose a new evidence directory; existing inputs are never replaced')
    settings = settings or Settings.load()
    spec = profile()['evidence']
    client = workspace_client(settings)
    with client.files.download(settings.volume_root + '/' + spec['volume_directory'] + '/evidence.zip').contents as stream:
        data = stream.read(spec['archive_bytes'] + 1)
    return extract_evidence(data, destination)


def extract_evidence(data, destination):
    """Verify the complete archive before replay; useful on mounted Databricks Volumes."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Choose a new evidence directory')
    spec = profile()['evidence']
    if len(data) != spec['archive_bytes'] or hashlib.sha256(data).hexdigest() != spec['archive_sha256']:
        raise ValueError('Evidence archive size/hash mismatch')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        raw = archive.read('MANIFEST.json')
        if hashlib.sha256(raw).hexdigest() != spec['manifest_sha256']:
            raise ValueError('Archived manifest differs from the frozen package')
        files = json.loads(raw)['files']
        expected = set(files) | {'MANIFEST.json'}
        if len(archive.namelist()) != len(expected) or set(archive.namelist()) != expected:
            raise ValueError('Unexpected or duplicate archive member')
        # Check sizes and paths before creating any output.
        for name in expected:
            member(destination, name)
            size = len(raw) if name == 'MANIFEST.json' else files[name]['bytes']
            if archive.getinfo(name).file_size != size:
                raise ValueError('Unexpected uncompressed evidence size')
        destination.mkdir(parents=True)
        for name in sorted(expected):
            payload = archive.read(name)
            if name != 'MANIFEST.json' and hashlib.sha256(payload).hexdigest() != files[name]['sha256']:
                raise ValueError('Archived evidence content changed')
            path = member(destination, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    return validate_evidence(destination)


def replay(root, output):
    """Recompute reported decisions from saved predictions, scores and workflow records."""
    root, output = Path(root), Path(output)
    validate_evidence(root)
    if output.resolve().is_relative_to(ROOT.resolve()) or output.exists():
        raise ValueError('Use a new output directory outside the checkout')
    output.mkdir(parents=True)
    try:
        result = replay_steps(root, output)
        write_json(output / 'REPRODUCTION_RESULT.json', result)
        return result
    except BaseException as error:
        write_json(output / 'REPRODUCTION_FAULT.json', {
            'status': 'FAILED', 'type': type(error).__name__, 'automatic_retry': False,
            'new_model_forwards': 0, 'training_updates': 0, 'new_reference_calls': 0})
        raise


def replay_steps(root, output):
    from src.reference_inputs import public_report
    from src.research2_classifier import agreement
    from src.research2_compare import compare, rows
    from src.research2_inference_report import report
    from src.classifier_helpers import decision, LABELS
    from src.engines.fixed_steering.campaign_analysis import analyze
    from src.engines.evaluation.screen import impossible
    read = lambda path: json.loads(path.read_bytes())
    references = root / 'research2/reference_inputs'
    predictions = read(root / 'binary/CLASSIFIER_PREDICTIONS.json')['probabilities']
    truth = rows(references / 'study/02_confirmation/gate/responses.jsonl')
    binary = agreement(truth, [predictions[row['case_id']] for row in truth])
    if binary != public_report(read(root / 'binary/RESULT.json')['confirmation'], references):
        raise ValueError('Binary classifier confirmation metrics changed')
    historical_binary = compare(references, output / 'binary_replay', root / 'binary/CLASSIFIER_PREDICTIONS.json')
    shutil.copytree(root / 'fresh_inference', output / 'fresh_inference')
    fresh = report(references, root / 'binary', output / 'fresh_inference')
    if fresh != public_report(read(root / 'fresh_inference/COMPARISON.json'), references):
        raise ValueError('Fresh-score comparison differs from the recorded result')
    shutil.copytree(root / 'historical_H24', output / 'historical_H24')
    workflows = analyze(output / 'historical_H24')
    fields = ('condition', 'family', 'stratum', 'scheduled', 'completed', 'runtime_incomplete',
              'not_executed', 'workflow_success', 'own_grant', 'inappropriate_actions')
    selected = lambda table: [{key: row[key] for key in fields} for row in table]
    if selected(workflows['tables']) != selected(public_report(read(root / 'historical_H24/EXPECTED_ANALYSIS.json'), references)['tables']):
        raise ValueError('Historical 27B workflow scoring changed')
    records = read(root / 'four_way/HELDOUT_PREDICTIONS_PRIVATE.json')
    counts = dict(label_matches=0, enabled_matches=0, signature_matches=0)
    for row in records:
        actual = decision([row['probabilities'][label] for label in LABELS])
        if any(actual[key] != row[key] for key in ('label', 'enabled', 'abstained', 'confidence')):
            raise ValueError('Cached four-way decision differs from its probabilities')
        counts['label_matches'] += actual['label'] == row['target']['label']
        counts['enabled_matches'] += actual['enabled'] == row['target']['enabled']
        counts['signature_matches'] += all(actual[key] == row['target'][key] for key in ('label','enabled','abstained'))
    expected = read(root / 'four_way/RESULT.json')['synthetic_family_heldout']
    if len(records) != expected['count'] or any(counts[key] != expected[key] for key in counts):
        raise ValueError('Cached four-way test counts changed')
    screens = {}
    for model in ('08', '2'):
        folder = root / 'screens' / model
        cases = [read(path) for path in sorted((folder / 'cases').glob('*.json'))]
        evidence = []
        for case in cases:
            turns = case['initial_turns'] + case['post_turns']
            evidence.append({'parse_failures': sum(bool(t.get('parse_error')) for t in turns),
                'truncated_turns': sum(bool(t['truncated']) for t in turns),
                'rejected_calls': case['current_metrics']['rejected_calls']})
        completed = sum(case['completed'] for case in cases)
        measured = {'started_cases': len(cases), 'completed_cases': completed,
                    'partial_cases': len(cases)-completed, 'unexecuted_cases': 16-len(cases)}
        saved = read(folder / 'SCREEN_RESULT.json')
        proof = impossible(evidence)
        if any(saved[k] != v for k,v in measured.items()) or proof != saved['fail_fast_proof']:
            raise ValueError('Evaluation16 counts or original stop proof changed')
        screens[model] = dict(measured, fail_fast_proof=proof)
    learned = read(root / 'learned_H24/RESULT.json')
    before, after = [read(root / f'learned_H24/UPDATE_{update}.json') for update in ('048','096')]
    trend = {key: [before[key], after[key]] for key in (
        'own_case_macro_full_vocabulary_forward_KL', 'own_case_macro_fixed_bootstrap_normalized_MSE')}
    proof = read(root / 'training_2b/FINAL270_PROOF.json')
    if (proof['actual_H2_updates'], proof['actual_Bridge_updates'], proof['actual_total_updates']) != (214,56,270):
        raise ValueError('2B final training-dose receipt changed')
    return {'status': 'RETAINED_EVIDENCE_REPLAY_VERIFIED', 'binary_confirmation': binary,
        'historical_binary_variants': list(historical_binary['variants']),
        'fresh_inference': {name: data['new_classifier_gate'] for name,data in fresh['models'].items()},
        'historical_27B_workflow_rows': len(workflows['tables']),
        'historical_27B_all96_complete': workflows['all_96_rollouts_complete'],
        'four_way_cached_test': dict(counts, count=len(records)), 'Evaluation16': screens,
        'learned_H24_receipt': {'coverage': 'receipt identity and aggregate audit; not a new fitting run',
            'own_successes': learned['self_preservation_successes'], 'own_cases': learned['self_cases'], 'fixed14_trend': trend},
        '2B_training': {'coverage': 'completion receipt audit; no retraining', 'updates': 270},
        'new_model_forwards': 0, 'training_updates': 0, 'new_reference_calls': 0,
        'fresh_training_or_inference_reproduced': False, 'new_independent_holdout_evidence': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--fetch-evidence', action='store_true')
    parser.add_argument('--execute', action='store_true', help='Replay saved evidence, not train or infer')
    args = parser.parse_args(argv)
    if args.fetch_evidence:
        if not args.evidence_root:
            parser.error('--fetch-evidence requires a new --evidence-root')
        fetch_evidence(args.evidence_root)
    if args.execute:
        if not args.evidence_root or not args.output:
            parser.error('--execute requires --evidence-root and a new --output')
        result = replay(args.evidence_root, args.output)
    else:
        result = {'status': 'PUBLIC_CHECKS_PASSED', 'source': verify(), 'stages': profile()['stages'],
            'private_evidence': 'legacy workspace packet not supplied; public paper artifacts are available separately',
            'public_paper_replay': 'python -m src.public_artifacts fetch|replay --root <artifact-directory>',
            'fresh_training_or_inference_reproduced': False, 'new_model_forwards': 0}
        if args.evidence_root:
            manifest = validate_evidence(args.evidence_root)
            result['private_evidence'] = {'verified_files': len(manifest['files'])}
    display = result if not args.execute else {
        'status': result['status'], 'output': str(args.output),
        'binary_confirmation_matches': result['binary_confirmation']['tp'] + result['binary_confirmation']['tn'],
        'binary_confirmation_cases': result['binary_confirmation']['count'],
        'historical_27B_all96_complete': result['historical_27B_all96_complete'],
        'new_model_forwards': 0, 'training_updates': 0}
    print(json.dumps(display, indent=2))
    return result


if __name__ == '__main__':
    main()
