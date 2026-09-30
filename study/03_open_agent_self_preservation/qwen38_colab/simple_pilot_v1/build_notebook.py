"""Freeze the application pilot and build its output-free Colab notebook."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def cell(kind, text):
    item = {'cell_type': kind, 'metadata': {}, 'source': text.splitlines(keepends=True)}
    if kind == 'code':
        item.update(outputs=[], execution_count=None)
    return item


def main():
    files = {str(path.relative_to(HERE)).replace('\\', '/'): path.read_text(encoding='utf-8')
             for path in HERE.rglob('*') if path.is_file() and path.suffix in ('.py', '.json', '.txt', '.md')
             and '__pycache__' not in str(path) and path.name != 'FREEZE.json'}
    frozen = {'status': 'Frozen before first application-pilot model run',
              'sha256': {name: hashlib.sha256(content.encode()).hexdigest() for name, content in files.items()},
              'model': 'Qwen/Qwen3.8-27B', 'revision': '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
              'operational_revision': 'User authorized application-level in-memory tools; no cgroup gate.'}
    freeze_text = json.dumps(frozen, indent=2) + '\n'
    (HERE / 'data/FREEZE.json').write_text(freeze_text, encoding='utf-8', newline='\n')
    files['data/FREEZE.json'] = freeze_text
    pin = json.loads((HERE.parent / 'model_pin.json').read_text(encoding='utf-8-sig'))
    setup = ('import hashlib, json, os, subprocess, sys, time, uuid\nfrom pathlib import Path\n'
             'SESSION_STARTED = time.monotonic()\n'
             'RUN_ID = "qwen38_simple_v1_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid.uuid4().hex[:8]\n'
             'ROOT = Path("/content/sp_lense_work") / RUN_ID\nROOT.mkdir(parents=True, exist_ok=False)\n'
             'for name in ("checkpoints/adapters", "checkpoints/resume", "training/configs", "training/logs", '
             '"training/receipts", "evaluation/scenarios", "evaluation/trajectories", "evaluation/tool_calls", '
             '"evaluation/results", "activations", "reports", "code"):\n    (ROOT / name).mkdir(parents=True, exist_ok=False)\n'
             'SOURCE_FILES = ' + repr(files) + '\n'
             'for name, content in SOURCE_FILES.items():\n'
             '    path = ROOT / "code" / name\n    path.parent.mkdir(parents=True, exist_ok=True)\n'
             '    path.write_text(content, encoding="utf-8", newline="\\n")\n'
             'MODEL_PIN = ' + repr(pin) + '\n'
             '(ROOT / "training/configs/model_pin.json").write_text(json.dumps(MODEL_PIN, indent=2))\n'
             '(ROOT / "manifest.json").write_text(json.dumps({"run_id":RUN_ID, "model":MODEL_PIN["repository"], '
             '"revision":MODEL_PIN["revision"], "boundary":"application-only fictional in-memory tools", '
             '"starting_balance_for_authorization":80, "authorized_units":50, "reserved_units":2, '
             '"thinking":False, "source_freeze":json.loads(SOURCE_FILES["data/FREEZE.json"])}, indent=2))\n'
             'print("RUN_ROOT", ROOT)\n'
             'print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], '
             'capture_output=True, text=True, check=True).stdout)\n'
             'subprocess.run([sys.executable,"-m","pip","install","--quiet","--no-input","-r", '
             'str(ROOT / "code/requirements.txt")], check=True, timeout=900)\n'
             'print("PINNED_PACKAGES_INSTALLED")\n')
    download = ('from huggingface_hub import snapshot_download\n'
                'MODEL_DIR = ROOT / "model"\n'
                'snapshot_download(repo_id=MODEL_PIN["repository"], revision=MODEL_PIN["revision"], '
                'local_dir=MODEL_DIR, token=False, max_workers=8, '
                'allow_patterns=["*.safetensors","*.json","*.jinja","merges.txt","vocab.json"])\n'
                'MODEL_MANIFEST = {}\n'
                'for item in MODEL_PIN["files"]:\n'
                '    path = MODEL_DIR / item["name"]\n'
                '    if not path.is_file():\n        continue\n'
                '    digest = hashlib.sha256()\n'
                '    with path.open("rb") as stream:\n'
                '        for block in iter(lambda: stream.read(16*1024**2), b""):\n            digest.update(block)\n'
                '    actual = digest.hexdigest()\n'
                '    if item["sha256"] and actual != item["sha256"]:\n'
                '        raise RuntimeError("Pinned model hash mismatch: " + item["name"])\n'
                '    MODEL_MANIFEST[item["name"]] = {"bytes":path.stat().st_size,"sha256":actual}\n'
                'assert len([name for name in MODEL_MANIFEST if name.endswith(".safetensors")]) == 18\n'
                '(ROOT / "training/configs/model_manifest.json").write_text(json.dumps(MODEL_MANIFEST, indent=2))\n'
                'print("OFFICIAL_MODEL_DOWNLOAD_VERIFIED", len(MODEL_MANIFEST), "files")\n')
    pilot = ('sys.path.insert(0, str(ROOT / "code"))\nfrom controller import run\n'
             'PILOT_RECEIPT = run(ROOT, "feasibility", min(1800, 21600 - (time.monotonic()-SESSION_STARTED)))\n'
             'FEASIBILITY = json.loads((ROOT / "training/receipts/feasibility.json").read_text())\n'
             'print("FEASIBILITY_SUMMARY", json.dumps({key:value for key,value in FEASIBILITY.items() '
             'if key not in ("targets","generation")}, indent=2))\n')
    campaign = ('# Conservatively admit the complete matched campaign from measured 1024-token updates.\n'
                'remaining_seconds = 21600 - (time.monotonic()-SESSION_STARTED)\n'
                'projected_training = max(FEASIBILITY["optimizer_block_seconds"]) * (214*3)\n'
                'generation = FEASIBILITY["generation"]\n'
                'seconds_per_token = generation["seconds"] / max(1, len(generation["token_ids"]))\n'
                '# Conservative generation projection; caps are maxima, actual episodes can be shorter.\n'
                'projected_evaluation = seconds_per_token * (4*(56*80 + 24*160 + 16*300))\n'
                'projection = 1.25*(projected_training + projected_evaluation) + 600\n'
                'print("CAMPAIGN_ADMISSION", json.dumps({"projected_seconds":projection,"remaining_seconds":remaining_seconds}))\n'
                'if projection > remaining_seconds:\n'
                '    raise RuntimeError("Measured campaign projection exceeds this bounded budget; retain pilot and adjust scope before full run.")\n'
                'CAMPAIGN_RECEIPT = run(ROOT, "campaign", remaining_seconds)\n')
    export = ('# Run only after the model subprocess has exited; no workers while Drive is mounted.\n'
              'import shutil, stat\n'
              'from google.colab import drive\n'
              'EXPORT_FILES = []\n'
              'for path in ROOT.rglob("*"):\n'
              '    relative = path.relative_to(ROOT)\n'
              '    if relative.parts[0] in ("model","worker_home"):\n        continue\n'
              '    info = path.lstat()\n'
              '    if stat.S_ISLNK(info.st_mode):\n        raise RuntimeError("Export symlink rejected")\n'
              '    if not stat.S_ISREG(info.st_mode):\n        continue\n'
              '    if info.st_size > 2*1024**3:\n        raise RuntimeError("Oversized export rejected")\n'
              '    EXPORT_FILES.append((path, relative, info.st_size))\n'
              'drive.mount("/content/drive")\n'
              'DEST = Path("/content/drive/MyDrive/sp_lense/research3/runs") / RUN_ID\n'
              'DEST.mkdir(parents=True, exist_ok=False)\n'
              'if shutil.disk_usage(DEST).free < sum(size for _,_,size in EXPORT_FILES) + 1024**3:\n'
              '    raise RuntimeError("Insufficient Drive free space")\n'
              'EXPORT_HASHES = {}\n'
              'for source, relative, size in EXPORT_FILES:\n'
              '    target = DEST / relative\n    target.parent.mkdir(parents=True, exist_ok=True)\n'
              '    shutil.copyfile(source, target)\n'
              '    def file_hash(path):\n'
              '        digest = hashlib.sha256()\n'
              '        with path.open("rb") as stream:\n'
              '            for block in iter(lambda: stream.read(16*1024**2), b""):\n                digest.update(block)\n'
              '        return digest.hexdigest()\n'
              '    actual = file_hash(source)\n'
              '    if actual != file_hash(target):\n        raise RuntimeError("Drive copy hash mismatch")\n'
              '    EXPORT_HASHES[str(relative)] = actual\n'
              '(DEST / "EXPORT_HASHES.json").write_text(json.dumps(EXPORT_HASHES, indent=2))\n'
              'drive.flush_and_unmount()\nprint("PRIVATE_EXPORT_VERIFIED", DEST, len(EXPORT_HASHES), "files")\n')
    cells = [cell('markdown', '# Research 3 — official Qwen3.8-27B application pilot\n\n'
                  'A100 + High-RAM. Model tools affect fictional memory only; no real filesystem, shell or network tools. '
                  'No cgroup prerequisite. Trusted code reads weights and writes artifacts. '
                  'Authorization: 50 units from initial observed balance 80, less prior usage; 2-unit reserve. '
                  'Run setup/download/feasibility first. The campaign has a 6-hour total notebook budget. '
                  'Export privately only after all model work stops; disconnect/delete the runtime when done.\n'),
             cell('code', setup), cell('code', download), cell('code', pilot),
             cell('code', campaign), cell('markdown', '## Private export after workers exit\n\n'
                  'Mount Drive only in this trusted phase. No base model weights are copied. '
                  'Do not run the export cell concurrently with model work.\n'), cell('code', export)]
    for index, item in enumerate(cells):
        item['id'] = f'simple-v1-{index:02d}'
    notebook = {'nbformat':4,'nbformat_minor':5,'metadata':{
        'kernelspec':{'name':'python3','display_name':'Python 3'},'language_info':{'name':'python'},
        'accelerator':'GPU','colab':{'name':'Research3_Qwen38_Simple_Pilot_V1.ipynb','gpuType':'A100'}},'cells':cells}
    target = HERE / 'Research3_Qwen38_Simple_Pilot_V1.ipynb'
    target.write_text(json.dumps(notebook, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(target)


if __name__ == '__main__':
    main()
