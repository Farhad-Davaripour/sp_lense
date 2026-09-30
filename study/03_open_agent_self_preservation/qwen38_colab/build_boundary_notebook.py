"""Keep the first readiness notebook unchanged; build a separate boundary probe."""
import json
from pathlib import Path
from build_notebook import cell

ROOT = Path(__file__).resolve().parent
source = (ROOT / 'containment_readiness.py').read_text(encoding='utf-8')
compile(source, 'containment_readiness.py', 'exec')
notebook = {
    'nbformat': 4, 'nbformat_minor': 5,
    'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                 'language_info': {'name': 'python'},
                 'colab': {'name': 'Research3_Colab_Boundary_Readiness.ipynb'}},
    'cells': [cell('markdown', '# Research 3 — trusted CPU boundary readiness\n\n'
                   'Model-free probes only. No Drive, model download or training. '
                   'Tests whether cgroup resource limits can be delegated in this '
                   'managed runtime. Neither a successful probe nor available '
                   'namespaces establish a passing worker isolation gate.\n'),
              cell('code', source)],
}
for index, item in enumerate(notebook['cells']):
    item['id'] = f'boundary-{index:02d}'
target = ROOT / 'Research3_Colab_Boundary_Readiness.ipynb'
target.write_text(json.dumps(notebook, indent=2) + '\n', encoding='utf-8')
print(target)
