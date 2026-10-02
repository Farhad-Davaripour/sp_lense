"""Trusted post-evaluation cell: context checks for failed completed fits."""
import datetime
import hashlib
import json
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

failed_candidates = []
for candidate_root in PRES_RUNS:
    row_path = candidate_root/'reports/CANDIDATE_RESULT.json'
    if row_path.is_file():
        row = json.loads(row_path.read_text())
        forced = candidate_root.name in globals().get('FORCE_CONTEXT_DIAG_IDS',[])
        if row.get('completed') and (not row.get('objective_development_gates_pass') or forced):
            failed_candidates.append(candidate_root)
for candidate_root in PRES_RUNS:
    running = subprocess.run(['pgrep','-f',re.escape(str(candidate_root/'code/worker.py'))],capture_output=True,text=True)
    if running.stdout.strip():
        raise RuntimeError('Wait until normal evaluation has completed')

if not failed_candidates:
    print('No completed candidate currently has a failed objective screen. Review semantic preferences before deciding further diagnostics.',flush=True)
else:
    frozen_queue = json.loads((PRES_ROOT/'QUEUE_FREEZE.json').read_text())
    launched = datetime.datetime.strptime(PRES_ROOT.name.split('_')[-2],'%Y%m%dT%H%M%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()
    conservative_seconds = (frozen_queue['balance_at_launch']-30-2.5)/frozen_queue['rate']*3600-(time.time()-launched)
    diagnostic_seconds = min(1200,conservative_seconds,21600-(time.monotonic()-SESSION_STARTED))
    if diagnostic_seconds < 300:
        raise RuntimeError('Diagnostic not admitted within remaining original budget/time')
    CONTEXT_DIAG_ROOT = PRES_ROOT/('context_diagnostics_'+str(time.time_ns()))
    CONTEXT_DIAG_ROOT.mkdir(exist_ok=False)
    pin = 'f4c2ca08f7c66af04c3d3283d2c568396048076d'
    prefix = 'https://raw.githubusercontent.com/Farhad-Davaripour/sp_lense/'+pin+'/study/03_open_agent_self_preservation/qwen38_colab/simple_pilot_v1/'
    diagnostic_source = urllib.request.urlopen(prefix+'context_diagnostic_worker.py',timeout=30).read()
    if hashlib.sha256(diagnostic_source).hexdigest()!='b4f2fabb30289b276d0379dacf782ec2c6097a9a5b226e1e0185f58fcb3aef9c':
        raise RuntimeError('Diagnostic source hash mismatch')
    selection = [f'v4_bridge_{context}_{condition}_{suffix}' for context in (0,7)
                 for condition in ('self_complete','self_unfinished','other_complete','no_threat_complete')
                 for suffix in (0,1)]
    CONTEXT_DIAG_RUNS = []
    for candidate_root in failed_candidates:
        diagnostic_root = CONTEXT_DIAG_ROOT/candidate_root.name
        diagnostic_root.mkdir(exist_ok=False)
        for name in ('training/logs','training/receipts','evaluation/trajectories','evaluation/results','reports'):
            (diagnostic_root/name).mkdir(parents=True,exist_ok=False)
        shutil.copytree(candidate_root/'code',diagnostic_root/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        (diagnostic_root/'code/worker.py').write_bytes(diagnostic_source)
        (diagnostic_root/'model').symlink_to(ROOT/'model',target_is_directory=True)
        config = {'candidate':candidate_root.name,'candidate_root':str(candidate_root),
                  'diagnostic_adapter':str(candidate_root/'checkpoints/adapters/preservation'),
                  'selection_ids':selection,'source_commit':pin,'training_set_diagnostic':False,'controlled_context_diagnostic':True,
                  'validation_scores_changed':False,'simulated_outcome_replay':False}
        (diagnostic_root/'code/data/EXECUTION_CONFIG.json').write_text(json.dumps(config,indent=2))
        freeze = {'status':'Context diagnostic frozen before generation',
                  'sha256':{str(p.relative_to(diagnostic_root/'code')):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (diagnostic_root/'code').rglob('*') if p.is_file() and p.name!='FREEZE.json'}}
        (diagnostic_root/'code/data/FREEZE.json').write_text(json.dumps(freeze,indent=2))
        (diagnostic_root/'manifest.json').write_text(json.dumps(config,indent=2))
        CONTEXT_DIAG_RUNS.append(diagnostic_root)
    (CONTEXT_DIAG_ROOT/'PLAN.json').write_text(json.dumps({'roots':list(map(str,CONTEXT_DIAG_RUNS)),
         'selection_ids':selection,'max_seconds':diagnostic_seconds,'source_commit':pin},indent=2))
    print('CONTEXT_DIAGNOSTICS_STARTED',str(CONTEXT_DIAG_ROOT),flush=True)
    CONTEXT_DIAG_RECEIPT = controller_pair.run_pair(CONTEXT_DIAG_RUNS,diagnostic_seconds,CONTEXT_DIAG_ROOT)
    for diagnostic_root in CONTEXT_DIAG_RUNS:
        summary_path = diagnostic_root/'reports/CONTEXT_DIAGNOSTIC_RESULT.json'
        print('CONTEXT_DIAGNOSTIC_SUMMARY',summary_path.read_text() if summary_path.exists() else
              json.dumps({'candidate':diagnostic_root.name,'completed':False}),flush=True)
