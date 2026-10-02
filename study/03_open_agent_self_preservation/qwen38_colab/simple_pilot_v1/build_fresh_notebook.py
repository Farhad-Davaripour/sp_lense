"""Build a clean fresh-fit notebook; keep the original pilot notebook unchanged."""
import base64
import gzip
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
original = json.loads((HERE / 'Research3_Qwen38_Simple_Pilot_V1.ipynb').read_text())
files = {str(path.relative_to(HERE)).replace('\\', '/'): path.read_text(encoding='utf-8')
         for path in HERE.rglob('*') if path.is_file() and path.suffix in ('.py', '.json', '.txt', '.md')
         and '__pycache__' not in str(path) and path.name != 'FREEZE.json'}
freeze = {'status': 'Fresh uninterrupted comparison, scientific thresholds unchanged',
          'sha256': {name: hashlib.sha256(content.encode()).hexdigest() for name, content in files.items()},
          'model': 'Qwen/Qwen3.8-27B', 'revision': '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
          'resumption': 'Not used; strict pilot comparison failed and remains recorded'}
files['data/FREEZE.json'] = json.dumps(freeze, indent=2) + '\n'
packed = base64.b64encode(gzip.compress(json.dumps(files).encode(), mtime=0)).decode()
literal = '(\n' + ''.join(repr(packed[index:index+100]) + '\n' for index in range(0, len(packed), 100)) + ')'
setup = ''.join(original['cells'][1]['source'])
setup = re.sub(r'PACKED_SOURCE = \(.*?\n\)\nSOURCE_FILES',
               'PACKED_SOURCE = ' + literal + '\nSOURCE_FILES', setup, count=1, flags=re.S)


def cell(kind, source, identifier):
    value = {'cell_type': kind, 'id': identifier, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        value.update(outputs=[], execution_count=None)
    return value


campaign = ('# Fresh fitting only: no pilot checkpoint or unvalidated resumed update.\n'
            'sys.path.insert(0, str(ROOT / "code"))\nfrom controller import run\n'
            'remaining_seconds = 21600 - (time.monotonic()-SESSION_STARTED)\n'
            'if any((ROOT / "checkpoints/adapters" / arm).exists() for arm in ("preservation","continuity","neutral")):\n'
            '    raise RuntimeError("This run already has fitted adapters; create a new immutable run for another experiment.")\n'
            'CAMPAIGN_RECEIPT = run(ROOT, "campaign", remaining_seconds)\n')
cells = [
    cell('markdown', '# Research 3 — fresh Qwen3.8-27B comparison\n\n'
         'A100 + High-RAM. This notebook runs fresh preservation, continuity and neutral '
         'fits, then compares the unchanged base. All model tools affect fictional '
         'memory only. Trusted code loads weights and saves research artifacts.\n\n'
         '**Before running:** check current Colab balance and rate against remaining '
         '50-unit authorization, including earlier runs; no purchases. The six-hour '
         'deadline is suitable for the observed 6.77-unit/hour A100 rate only when '
         'sufficient authorized units remain. Reduce the deadline otherwise.\n\n'
         'The strict checkpoint-resume pilot failed. This notebook does not resume '
         'training or call that check passed. See EXECUTION_UPDATE.md. Run setup, '
         'download, and the fresh campaign once. Status is read-only. Export after '
         'workers exit; disconnect the runtime when finished.\n', 'fresh-header'),
    cell('code', setup, 'fresh-setup'),
    cell('code', ''.join(original['cells'][2]['source']), 'fresh-download'),
    cell('code', campaign, 'fresh-campaign'),
    cell('code', (HERE / 'display_status.py').read_text(), 'fresh-status'),
    cell('markdown', '## Private export after model work exits\n\n'
         'No base weights exported. Mount Drive only during this trusted phase.\n', 'fresh-export-note'),
    cell('code', ''.join(original['cells'][6]['source']), 'fresh-export'),
]
notebook = dict(original, cells=cells)
notebook['metadata']['colab']['name'] = 'Research3_Qwen38_Fresh_Comparison.ipynb'
target = HERE / 'Research3_Qwen38_Fresh_Comparison.ipynb'
target.write_text(json.dumps(notebook, indent=2) + '\n', encoding='utf-8', newline='\n')
print(target)
