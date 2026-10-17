"""Read-only reconciliation of old signed responses; no model calls."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path('C:/Users/farha/Research4/artifacts/energy_response_v1/pilot_20261012')
OLD=Path('C:/Users/farha/Research4/artifacts/propagation_followup_v1/recovery_20261010/verified_export/verified_native')
ORIGINAL=Path('C:/Users/farha/Research4/artifacts/rotation_pilot_v1/executed_20261009/verified_native')
ROOT.mkdir(parents=True,exist_ok=True)
def cos(a,b):
    denominator=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.sum(a*b)/denominator) if denominator>1e-20 else None
rows=[]
panel=json.loads((OLD/'native/PANEL.json').read_text())
receipt=json.loads((OLD/'native/STATUS.json').read_text())
for prompt in panel['prompts']:
    pid=prompt['id'];arrays={};before={};tokens={}
    for arm in ['base','plus_v','minus_v']:
        with np.load(OLD/'native'/pid/(arm+'.npz'),allow_pickle=False) as z:
            arrays[arm]=z['residuals'].astype(np.float64);before[arm]=z['pre_injection'].astype(np.float64);tokens[arm]=z['forced_token_ids'].copy()
    assert all(np.array_equal(tokens['base'],tokens[a]) and np.array_equal(before['base'],before[a]) and
        np.array_equal(arrays['base'][:10],arrays[a][:10]) for a in ['plus_v','minus_v'])
    for phase,sl in [('prefill',slice(0,1)),('decode',slice(1,None)),('all',slice(None))]:
        dp=arrays['plus_v'][15,sl]-arrays['base'][15,sl]
        dm=arrays['minus_v'][15,sl]-arrays['base'][15,sl]
        sitep=arrays['plus_v'][10,sl]-before['plus_v'][sl]
        sitem=arrays['minus_v'][10,sl]-before['minus_v'][sl]
        odd=(dp-dm)/2;even=(dp+dm)/2
        oe=float(np.sum(odd*odd));ee=float(np.sum(even*even))
        measured_dot=float(np.sum(dp*dm));identity=ee-oe
        coeffs={}
        for arm in ['plus_v','minus_v']:
            record=next(r for r in receipt['completed'] if r['prompt']==pid and r['condition']==arm)
            delivery=np.asarray(record['delivery'],dtype=float)[sl]
            coeffs[arm]=[float(delivery[:,5].min()),float(delivery[:,6].max())]
        rows.append(dict(prompt=pid,pair=prompt['pair'],phase=phase,positions=len(dp),
            site_opposition_cosine=cos(sitep,-sitem),site_relative_even_component=float(np.linalg.norm((sitep+sitem)/2)/np.linalg.norm((sitep-sitem)/2)),
            downstream_response_cosine=cos(dp,dm),downstream_opposition_cosine=cos(dp,-dm),
            mean_token_response_cosine=float(np.mean([cos(a,b) for a,b in zip(dp,dm)])),
            sign_even_energy_fraction=ee/(ee+oe),even_to_odd_norm_ratio=float(np.sqrt(ee/oe)),
            exact_dot_identity_relative_error=abs(measured_dot-identity)/max(1e-20,abs(measured_dot),ee+oe),
            native_coefficient_ranges=coeffs))
integrity={}
for folder in [ORIGINAL,OLD]:
    manifest=json.loads((folder/'EXPORT_HASHES.json').read_text())
    for member,digest in manifest.items():assert hashlib.sha256((folder/member).read_bytes()).hexdigest()==digest
    integrity[str(folder)]=len(manifest)
result={'status':'old findings reconciled, unchanged','records':rows,'native_manifest_files_verified':integrity,
    'important_scope':[
        'Same forced28-token histories, identical upstream and pre-injection output; intervention post block10, readout post block15.',
        'Persistent intervention at all nonpadding prefill positions and every cached decode step; downstream state therefore differs across signs even with identical token IDs.',
        'Minus arm used a native norm-matching coefficient grid rather than exact negation of plus; actual site changes are almost opposite, not mathematically identical.',
        'BF16 weights, additions, activations and cached/recurrent propagation; no FP32 causal-path control in the old panel.',
        'A fixed linear map sends opposite input deltas to opposite output deltas. A right-angle finite response cannot alone establish a shared rotation.',
        'Dot identity dplus dot dminus = ||even||^2 - ||odd||^2 is exact bookkeeping; even energy comparable to odd explains the angle descriptively, not its causal origin.',
        'Earlier four-context covariance/norm energy and14-question family-withheld detector failures remain failures. Stopped consistency monitor is not resumed.'
    ]}
(ROOT/'OLD_RECONCILIATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'decode':[r for r in rows if r['phase']=='decode'],'verified_files':sum(integrity.values())},indent=2))
