"""Trusted post-worker summaries; raw traces stay private."""
import base64
import gzip
import json

evidence={'root':str(ROOT),'streams':{},'data_audit':json.loads((ROOT/'DATA_AUDIT.json').read_text()),
          'controller':json.loads((ROOT/'CONCURRENT_RECEIPT.json').read_text())}
for root in STREAM_ROOTS:
    info={}
    for name in ('STEP_ZERO','STREAM_RESULT','FAILURE'):
        p=root/'reports'/(name+'.json')
        if p.exists(): info[name]=json.loads(p.read_text())
    if root.name=='replay_coverage_stream':
        info['fits']={}
        for recipe in ('reference','coverage'):
            fit=root/recipe
            checkpoints=[json.loads(p.read_text()) for p in sorted((fit/'reports').glob('CHECKPOINT_*.json'))]
            preferences=fit/'evaluation/results/preference_validation.json'
            info['fits'][recipe]={'checkpoints':checkpoints,
                'preferences':json.loads(preferences.read_text()) if preferences.exists() else [],
                'result':json.loads((fit/'reports/FIT_RESULT.json').read_text()) if (fit/'reports/FIT_RESULT.json').exists() else {'completed':False}}
    evidence['streams'][root.name]=info
(ROOT/'review').mkdir(exist_ok=True)
(ROOT/'review/STREAM_EVIDENCE.json').write_text(json.dumps(evidence,indent=2))
print('REPLAY_DIAGNOSTICS_EVIDENCE_GZIP_BASE64',base64.b64encode(gzip.compress(json.dumps(evidence).encode())).decode(),flush=True)
