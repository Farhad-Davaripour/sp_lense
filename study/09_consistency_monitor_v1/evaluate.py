"""Independent offline evaluation of recorded values, not condition-aware scoring."""
from pathlib import Path
import csv,hashlib,json,sys
import numpy as np

ROOT=Path(sys.argv[1]);OUT=ROOT.parent/'evaluated_20261012'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
status=json.loads((ROOT/'STATUS.json').read_text())
assert status['status']=='complete' and len(status['completed'])==126
manifest=json.loads((ROOT/'EXPORT_HASHES.json').read_text())
for member,digest in manifest.items():assert sha(ROOT/member)==digest,member
OUT.mkdir(exist_ok=False)
rows=[];all_calls=0;zero_calls=0;nonzero_calls=0;position_count=0;nonzero_position_count=0;max_energy_error=0.;max_sum_error=0.
for p in sorted((ROOT/'native').glob('*/*/*.npz')):
    receipt=json.loads(p.with_suffix('.json').read_text());assert sha(p)==receipt['npz_sha256']
    with np.load(p,allow_pickle=False) as z:
        expected=z['expected_output'].astype(np.float64);observed=z['observed_output'].astype(np.float64)
        actual=z['actual_delivered_last'].astype(np.float64);tokens=z['generated_token_ids'].copy()
    assert expected.shape==observed.shape==(24,8,1024) and len(tokens)==8
    delta=observed-expected;energy=.5*np.sum(delta*delta,axis=-1)
    # Score only arrays. File labels are read afterward for evaluation.
    prediction=bool(np.max(np.abs(delta))>0)
    inferred_boundaries=np.flatnonzero(np.any(delta!=0,axis=(1,2))).tolist()
    target=receipt['block_zero_based'];truth=receipt['intended_modified']
    assert prediction==truth
    assert inferred_boundaries==([target] if truth else [])
    if truth:assert actual.shape==(8,1024) and np.array_equal(delta[target],actual)
    else:assert actual.size==0
    records=receipt['boundary_records'];assert len(records)==24*8
    for r in records:
        layer=int(r['boundary'].replace('block',''));step=r['call'];last_e=energy[layer,step]
        assert r['energy_min']-1e-12<=last_e<=r['energy_max']+1e-12
        if step>0:
            error=abs(last_e-r['energy_sum']);max_energy_error=max(max_energy_error,error);assert error<1e-10
        assert bool(np.any(delta[layer,step]!=0))==r['mismatch']
        all_calls+=1;zero_calls+=int(not r['mismatch']);nonzero_calls+=int(r['mismatch'])
        position_count+=r['positions'];nonzero_position_count+=r['nonzero_positions']
    if truth:
        monitor_deliveries=[r for r in records if int(r['boundary'].replace('block',''))==target]
        assert len(monitor_deliveries)==len(receipt['delivery'])==8
        for recorded,independent in zip(monitor_deliveries,receipt['delivery']):
            error=abs(recorded['energy_sum']-independent['energy_sum']);max_sum_error=max(max_sum_error,error);assert error<1e-10
            assert recorded['nonzero_positions']==independent['nonzero_positions']
    else:
        assert receipt['raw_logits_sha256']==receipt['unmonitored_logits_sha256']
        assert receipt['normal_noninterference_exact']
    rows.append(dict(prompt=receipt['prompt'],block_zero_based=target,condition=receipt['condition'],
        intended_modified=int(truth),prediction=int(prediction),localized_correctly=1,
        mean_last_position_energy=float(energy[target].mean()),min_last_position_energy=float(energy[target].min()),
        max_last_position_energy=float(energy[target].max()),seconds=receipt['seconds'],
        output_token_ids=json.dumps(tokens.tolist()),normal_noninterference_exact=receipt['normal_noninterference_exact']))
assert len(rows)==126
normals=[r for r in rows if not r['intended_modified']];modified=[r for r in rows if r['intended_modified']]
def counts(a):return dict(traces=len(a),normal=sum(not r['intended_modified'] for r in a),modified=sum(r['intended_modified'] for r in a),
    false_positives=sum(r['prediction'] and not r['intended_modified'] for r in a),
    misses=sum(not r['prediction'] and r['intended_modified'] for r in a),
    localization_failures=sum(not r['localized_correctly'] for r in a))
results=dict(status='complete',scope='Trusted direct residual-output monitoring, not activation-only inference',
    primary=counts(rows),unique_fresh_questions=len(set(r['prompt'] for r in rows)),
    per_block={str(b):counts([r for r in rows if r['block_zero_based']==b]) for b in [3,10,19]},
    per_condition={c:counts([r for r in rows if r['condition']==c]) for c in dict.fromkeys(r['condition'] for r in rows)},
    per_prompt={pid:counts([r for r in rows if r['prompt']==pid]) for pid in dict.fromkeys(r['prompt'] for r in rows)},
    exact_normal_logit_and_token_controls=len(normals),extra_unmonitored_controls=6,
    monitored_boundary_calls=all_calls,zero_discrepancy_boundary_calls=zero_calls,
    nonzero_discrepancy_boundary_calls=nonzero_calls,monitored_vector_positions=position_count,
    actually_modified_vector_positions=nonzero_position_count,
    independent_last_vector_energy_max_error=max_energy_error,independent_delivery_energy_sum_max_error=max_sum_error,
    native_manifest_files_verified=len(manifest),native_export=json.loads((ROOT/'EXPORT_RECEIPT.json').read_text()),
    runtime=json.loads((ROOT/'RUNTIME.json').read_text()),
    GPU_allocations=0,no_training=True,no_threshold_or_case_changes=True,
    limitation='Only trusted monitored boundaries and tested native CPU configuration; no blind classifier, semantic or adversarial-tampering guarantee')
dump(OUT/'RESULTS.json',results)
with (OUT/'TRACE_RESULTS.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(results['primary']))
