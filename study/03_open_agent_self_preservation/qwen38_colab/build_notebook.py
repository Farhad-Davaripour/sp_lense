"""Build an output-free, self-contained notebook for trusted model-free inventory."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def cell(kind, source):
    result = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def build():
    readiness = (ROOT / 'readiness.py').read_text(encoding='utf-8')
    readiness = readiness.split("if __name__ == '__main__':")[0]
    cells = [
        cell('markdown', '# Research 3 — official Qwen3.8-27B readiness only\n\n'
             'This notebook does **not** load, train, or run a model. No Drive mount. '
             'Preserve the earlier community GGUF notebook. Use only the authorized '
             '50 compute units; no purchases. Complete account/terms review first.\n\n'
             'Use CPU initially for model-free probes to avoid unnecessary GPU billing. '
             'Inspect Settings → Subscription and runtime resources for actual balance '
             'and GPU unit rate. Prefer A100 for the later pilot; never start full '
             '27B training on T4. A selected GPU is not evidence of availability.\n'),
        cell('code', '# Researcher-entered observations from Colab UI, not assumptions.\n'
             'OBSERVED_START_BALANCE = None\nOBSERVED_CURRENT_BALANCE = None\n'
             'OBSERVED_RATE_UNITS_PER_HOUR = None\n'
             'print("Authorized ceiling: 50 units; actual balance/rate still require UI verification.")\n'),
        cell('code', readiness + '\nreport = inventory()\nprint(json.dumps(report, indent=2))\n'),
        cell('code', 'from datetime import datetime, timezone\nimport uuid\n'
             'RUN_ID = "qwen38_readiness_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]\n'
             'run_root = Path("/content/sp_lense_work") / RUN_ID\n'
             'if report["drive_mount_lines"]:\n'
             '    raise RuntimeError("Drive is mounted: no worker may start. Inventory only.")\n'
             'run_root.mkdir(parents=True, exist_ok=False)\n'
             'for relative in ("checkpoints/adapters", "checkpoints/resume", "training/configs", '
             '"training/logs", "training/receipts", "evaluation/scenarios", "evaluation/trajectories", '
             '"evaluation/tool_calls", "evaluation/results", "activations", "reports"):\n'
             '    (run_root / relative).mkdir(parents=True, exist_ok=False)\n'
             'manifest = {"run_id": RUN_ID, "status": "readiness_only", '
             '"authorized_compute_units": 50, "observed_start_balance": OBSERVED_START_BALANCE, '
             '"observed_current_balance": OBSERVED_CURRENT_BALANCE, '
             '"observed_rate_units_per_hour": OBSERVED_RATE_UNITS_PER_HOUR, '
             '"model": "Qwen/Qwen3.8-27B", "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", '
             '"worker_started": False, "isolation_verified": False, "inventory": report}\n'
             '(run_root / "manifest.json").write_text(json.dumps(manifest, indent=2))\n'
             'print("Local readiness receipt:", run_root / "manifest.json")\n'),
        cell('markdown', '## Stop here\n\n'
             'The restricted GPU worker, independent forbidden-operation/resource tests, '
             'software compatibility pilot, save/reload/resume test, baseline gate, '
             'scientific freeze and budget-admitted training are subsequent stages. '
             'Successful inventory alone authorizes none of them. The main experiment '
             'must not use a community GGUF or old 0.8B adapter. Never interpret '
             'namespace availability or offline library flags as verified containment.\n'),
    ]
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
        'language_info': {'name': 'python'},
        'colab': {'name': 'Research3_Qwen38_Official_Readiness.ipynb'}}, 'cells': cells}
    for index, item in enumerate(cells):
        item['id'] = f'readiness-{index:02d}'
        if item['cell_type'] == 'code':
            compile(''.join(item['source']), f'cell-{index}', 'exec')
    target = ROOT / 'Research3_Qwen38_Official_Readiness.ipynb'
    target.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return target


if __name__ == '__main__':
    print(build())
