"""Separate A100 model-free inventory; preserve both CPU notebooks."""
import json
from pathlib import Path
from build_notebook import cell

ROOT = Path(__file__).resolve().parent
sources = [(ROOT / name).read_text(encoding='utf-8')
           for name in ('readiness.py', 'containment_readiness.py')]
sources[0] = sources[0].split("if __name__ == '__main__':")[0] + '\nprint(json.dumps(inventory(), indent=2))\n'
for index, source in enumerate(sources):
    compile(source, f'gpu-probe-{index}', 'exec')
cells = [cell('markdown', '# Research 3 — GPU boundary feasibility only\n\n'
              'Select A100/high RAM only within the authorized 50 compute units. '
              'Read current rate/balance immediately after allocation. These '
              'trusted model-free probes test actual hardware and cgroup facilities. '
              'No Drive mount, model download, adapter training or agentic evaluation. '
              'Disconnect/delete this newly created test runtime after capturing '
              'results if required confinement is unavailable.\n')]
cells.extend(cell('code', source) for source in sources)
for index, item in enumerate(cells):
    item['id'] = f'gpu-boundary-{index:02d}'
notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
    'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
    'language_info': {'name': 'python'}, 'accelerator': 'GPU',
    'colab': {'name': 'Research3_A100_Boundary_Readiness.ipynb', 'gpuType': 'A100'}},
    'cells': cells}
target = ROOT / 'Research3_A100_Boundary_Readiness.ipynb'
target.write_text(json.dumps(notebook, indent=2) + '\n', encoding='utf-8')
print(target)
