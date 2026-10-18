"""Read immutable cached annotations without coupling code to a provider's name."""
import json
import re
from pathlib import Path
from src.utils import sha256


def annotation_token(root):
    """Read an opaque legacy identifier from the verified private path register."""
    manifest = json.loads((Path(root).parent / 'REFERENCE_MANIFEST.json').read_bytes())
    tokens = {match.group(1) for name in manifest['files']
              if (match := re.search(r'/02_([^/]+)_gate/', name))}
    if len(tokens) != 1:
        raise ValueError('Expected one cached annotation namespace')
    return tokens.pop()


def public_report(value, root):
    """Alias report metadata after original bytes have passed integrity checks.

    Only use on saved reports, never on model-visible inputs or fitting targets.
    Numeric measurements and original stored artifacts remain unchanged.
    """
    token = annotation_token(root)
    def visit(item):
        if isinstance(item, dict):
            return {re.sub(re.escape(token), 'reference', key, flags=re.I): visit(v)
                    for key, v in item.items()}
        if isinstance(item, list):
            return [visit(v) for v in item]
        if isinstance(item, str):
            return re.sub(re.escape(token), 'reference', item, flags=re.I)
        return item
    return visit(value)


def cached_file(directory, name):
    """Resolve a renamed report/arm using its unique provider-independent suffix."""
    directory = Path(directory)
    target = directory / name
    if target.is_file():
        return target
    candidates = list(directory.glob(name.replace('reference', '*')))
    if len(candidates) != 1 or not candidates[0].is_file():
        raise ValueError('Expected one immutable cached file for the public alias')
    return candidates[0]


def cached_response(row):
    """Accept the public neutral field or one unambiguous legacy response field."""
    candidates = [value for key, value in row.items()
                  if key.endswith('_response') and isinstance(value, dict)
                  and 'answers' in value and 'model' in value]
    if len(candidates) != 1:
        raise ValueError('Expected exactly one cached annotation response')
    return candidates[0]


def cached_provenance(row):
    candidates = [value for key, value in row.items()
                  if key.endswith('_provenance') and isinstance(value, dict)
                  and value.get('actual_original_service_call') is True]
    if len(candidates) != 1 or not any(
            key.startswith('genuine_cached_') and value is True
            for key, value in candidates[0].items()):
        raise ValueError('Genuine original cached annotation provenance is required')
    return candidates[0]


def frozen_source(root, public_path, expected_sha256=None):
    """Resolve a public alias through the original pinned private manifest.

    Stored source bytes and private paths stay unchanged. A digest is preferred;
    metadata files without a caller digest require a unique stage/basename match.
    Every returned file is checked against the original manifest.
    """
    root = Path(root).resolve()
    direct = (root / public_path).resolve()
    if not direct.is_relative_to(root):
        raise ValueError('Cached source must stay inside its private reference root')
    if expected_sha256 and direct.is_file():
        if sha256(direct) != expected_sha256:
            raise ValueError('Cached source content changed')
        return direct
    manifest = json.loads((root.parent / 'REFERENCE_MANIFEST.json').read_bytes())
    files = manifest['files']
    public = Path(public_path)
    if public_path in files:
        matches = [public_path]
    elif expected_sha256:
        matches = [name for name, pin in files.items() if pin['sha256'] == expected_sha256]
    else:
        stage = public.parts[1].split('_', 1)[0] if len(public.parts) > 1 else None
        matches = [name for name in files if Path(name).name == public.name
                   and (stage is None or len(Path(name).parts) > 1
                        and Path(name).parts[1].split('_', 1)[0] == stage)]
    if len(matches) != 1:
        raise ValueError('No unique pinned cached source for the public alias')
    name = matches[0]
    target = (root / name).resolve()
    pin = files[name]
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError('Cached source must stay inside its private reference root')
    actual = sha256(target)
    if actual != pin['sha256'] or expected_sha256 and actual != expected_sha256:
        raise ValueError('Cached source content changed')
    return target
