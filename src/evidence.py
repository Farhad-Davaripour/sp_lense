"""Shared file checks and saved-score summaries; no model or workspace access."""
from pathlib import Path, PurePosixPath
import json
from src.utils import sha256


def member(root, name):
    """Resolve a portable archive member while keeping it inside its evidence root."""
    relative = PurePosixPath(name)
    if relative.is_absolute() or not name or name == '.' or '\\' in name or ':' in name or '..' in relative.parts:
        raise ValueError('Unsafe evidence member: ' + name)
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Evidence member escapes its root')
    return path


def pinned_manifest(root, digest, message):
    path = Path(root) / 'MANIFEST.json'
    if sha256(path) != digest:
        raise ValueError(message)
    return json.loads(path.read_bytes())


def verify_files(root, pins, message):
    for name, pin in pins.items():
        path = member(root, name)
        if path.stat().st_size != pin['bytes'] or sha256(path) != pin['sha256']:
            raise ValueError(message + name)


def replay_output(output, evidence_root, checkout_root):
    """Reject output locations that could replace code or the original evidence."""
    if output is None:
        raise ValueError('Replay requires a new output directory')
    output = Path(output)
    resolved = output.resolve()
    if output.exists() or any(resolved.is_relative_to(Path(root).resolve())
                              for root in (evidence_root, checkout_root)):
        raise ValueError('Use a new output directory outside the checkout and evidence')
    return output


def historical_rows(rows):
    fields = ('condition', 'family', 'stratum', 'scheduled', 'completed', 'runtime_incomplete',
              'not_executed', 'workflow_success', 'own_grant', 'inappropriate_actions')
    return [{key: row[key] for key in fields} for row in rows]


def screen_summary(folder):
    """Rescore the preserved screen counts using the original frozen stop rule."""
    from src.engines.evaluation.screen import impossible
    folder = Path(folder)
    cases = [json.loads(path.read_bytes()) for path in sorted((folder / 'cases').glob('*.json'))]
    evidence = []
    for case in cases:
        turns = case['initial_turns'] + case['post_turns']
        evidence.append({'parse_failures': sum(bool(turn.get('parse_error')) for turn in turns),
                         'truncated_turns': sum(bool(turn['truncated']) for turn in turns),
                         'rejected_calls': case['current_metrics']['rejected_calls']})
    completed = sum(case['completed'] for case in cases)
    measured = {'started_cases': len(cases), 'completed_cases': completed,
                'partial_cases': len(cases) - completed, 'unexecuted_cases': 16 - len(cases)}
    saved = json.loads((folder / 'SCREEN_RESULT.json').read_bytes())
    proof = impossible(evidence)
    if any(saved[key] != value for key, value in measured.items()) or proof != saved['fail_fast_proof']:
        raise ValueError('Small-model stopped-screen proof differs')
    return measured, proof
