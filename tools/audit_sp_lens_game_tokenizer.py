"""CPU tokenizer-only audit: never imports MLX or loads model weights."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--renderer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve tokenizer receipt')
    start = time.monotonic()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False)
    spec = importlib.util.spec_from_file_location('bound_game_renderer', args.renderer)
    renderer = importlib.util.module_from_spec(spec); spec.loader.exec_module(renderer)
    manifest = json.loads((args.root / 'manifest.json').read_text())
    schema = json.loads((args.root / 'action_schema.json').read_text())
    for name in ('train/train.jsonl', 'dev/dev.jsonl', 'action_schema.json'):
        if sha(args.root / name) != manifest['file_sha256'][name]: raise ValueError('Staged dataset hash mismatch: ' + name)
    counts = {}; records = []
    for split in ('train', 'dev'):
        lines = (args.root / split / (split + '.jsonl')).read_text().splitlines()
        if len(lines) != manifest['counts'][split]: raise ValueError('Split count mismatch')
        counts[split] = len(lines)
        for index, line in enumerate(lines):
            row = json.loads(line)
            ids, prefix = renderer.render(tokenizer, row, schema)
            records.append({'split': split, 'index': index, 'total_tokens': len(ids), 'prefix_tokens': prefix,
                            'target_tokens': len(ids) - prefix,
                            'action': json.loads(row['messages'][-1]['content'])['action']})
    sizes = [r['total_tokens'] for r in records]
    receipt = {'schema': 'research4.sp-lens-game.tokenizer-audit.v1',
               'scope': 'Exact tokenizer and renderer only; no model weight load, GPU execution or inference.',
               'dataset_manifest_sha256': sha(args.root / 'manifest.json'),
               'renderer_sha256': sha(args.renderer), 'audit_script_sha256': sha(Path(__file__)),
               'model_directory': str(args.model),
               'input_sha256': {name: sha(args.root / name) for name in ('train/train.jsonl', 'dev/dev.jsonl', 'action_schema.json')},
               'counts': counts, 'records': records, 'total_rows': len(records),
               'max_tokens': max(sizes), 'min_tokens': min(sizes), 'mean_tokens': statistics.mean(sizes),
               'max_prefix_tokens': max(r['prefix_tokens'] for r in records),
               'max_target_tokens': max(r['target_tokens'] for r in records),
               'sequence_limit': 1024, 'over_limit': sum(size > 1024 for size in sizes),
               'no_truncation_required': max(sizes) <= 1024,
               'elapsed_seconds': time.monotonic() - start,
               'mlx_imported': any(name == 'mlx' or name.startswith('mlx.') for name in sys.modules),
               'status': 'PASS' if max(sizes) <= 1024 else 'FAIL_OVER_LIMIT'}
    if receipt['mlx_imported']: raise RuntimeError('Tokenizer-only audit imported MLX')
    args.output.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({key: receipt[key] for key in ('status', 'counts', 'max_tokens', 'max_prefix_tokens', 'over_limit', 'elapsed_seconds', 'mlx_imported')}, indent=2))
    if receipt['status'] != 'PASS': raise SystemExit(2)


if __name__ == '__main__': main()
