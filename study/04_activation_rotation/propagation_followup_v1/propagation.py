"""Saved-data paired geometry, actual-injection accounting and depth controls."""
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np


def geometry(base, changed, floor=1e-12):
    b = np.linalg.norm(base, axis=-1)
    c = np.linalg.norm(changed, axis=-1)
    delta = changed-base
    absolute = np.linalg.norm(delta, axis=-1)
    valid = (b > floor) & (c > floor)
    q = np.full(b.shape, np.nan)
    q[valid] = c[valid]/b[valid]
    dot = np.einsum('...i,...i->...', base, changed)
    cosine = np.full(b.shape, np.nan)
    cosine[valid] = np.clip(dot[valid]/(b[valid]*c[valid]), -1, 1)
    radial = np.divide(np.einsum('...i,...i->...', delta, base), b,
                       out=np.full(b.shape, np.nan), where=b > floor)
    orthogonal = np.sqrt(np.maximum(absolute**2-radial**2, 0))
    angle = np.arctan2(orthogonal, np.divide(dot, b, out=np.zeros(b.shape), where=b > floor))
    relative2 = np.divide(absolute**2, b**2, out=np.full(b.shape, np.nan), where=b > floor)
    magnitude_term = (q-1)**2
    direction_term = 2*q*(1-cosine)
    identity_error = np.abs(relative2-magnitude_term-direction_term)
    return dict(valid=valid, baseline_norm=b, changed_norm=c, q=q, angle=angle,
                absolute_displacement=absolute, relative_displacement=np.sqrt(relative2),
                magnitude_term=magnitude_term, direction_term=direction_term,
                radial_component=radial, orthogonal_component=orthogonal,
                identity_absolute_error=identity_error)


def cayley(a):
    eye = np.eye(len(a))
    return np.linalg.solve((eye-a/2).T, (eye+a/2).T).T


def run(root, output):
    sys.dont_write_bytecode = True
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((root/'source/pilot.json').read_text())
    spec = importlib.util.spec_from_file_location('archived_analysis', root/'source/analyze.py')
    archived = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archived)
    manifest = json.loads((root/'EXPORT_HASHES.json').read_text())
    blocks = json.loads((root/'native/STATUS.json').read_text())['configured_block_types']
    rows, phase_rows, counts = [], [], {}
    max_identity = max_delivery_error = 0.
    for cohort in ['native', 'replication_seed_42', 'replication_seed_43', 'replication_seed_44']:
        status = json.loads((root/cohort/'STATUS.json').read_text())
        receipts = {(r['prompt'], r['condition']):r for r in status['completed']}
        count = 0
        for prompt in config['prompts']:
            folder = root/cohort/prompt['id']
            with np.load(folder/'base.npz') as z:
                base = z['residuals'].astype(np.float64)
                token_ids = z['generated_token_ids']
            for arm in config['conditions'][1:]:
                path = folder/(arm+'.npz')
                assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[str(path.relative_to(root)).replace('\\','/')]
                with np.load(path) as z:
                    assert np.array_equal(z['generated_token_ids'], token_ids)
                    changed = z['residuals'].astype(np.float64)
                values = geometry(base, changed)
                assert np.all(values['valid']), 'Zero/near-zero denominator; explicit handling needed'
                error = float(np.max(values['identity_absolute_error']))
                assert error < 1e-10
                max_identity = max(max_identity, error)
                injected = values['absolute_displacement'][10]
                delivered = np.array(receipts[(prompt['id'], arm)]['delivery_norm_and_changed_count'])[:,0]
                error = float(np.max(np.abs(injected-delivered)))
                assert error < 1e-6
                max_delivery_error = max(max_delivery_error, error)
                assert np.max(values['absolute_displacement'][:10]) == 0
                assert np.all(injected > 1e-12)
                values['gain_over_actual_injected_displacement'] = values['absolute_displacement']/injected[None,:]
                np.savez_compressed(output/(cohort+'_'+prompt['id']+'_'+arm+'.npz'), **values)
                count += int(np.prod(values['q'].shape))
                for block in range(len(base)):
                    relative_energy = np.square(values['relative_displacement'][block]).sum()
                    row = dict(cohort=cohort, prompt=prompt['id'], split=prompt['split'], condition=arm,
                               block_zero_based=block, block_type=blocks[block], positions=base.shape[1],
                               invalid_denominators=int((~values['valid'][block]).sum()))
                    for name in ['baseline_norm','changed_norm','q','angle','absolute_displacement',
                                 'relative_displacement','radial_component','orthogonal_component',
                                 'gain_over_actual_injected_displacement']:
                        row['mean_'+name] = float(np.mean(values[name][block]))
                    row['magnitude_fraction_of_relative_displacement_energy'] = float(values['magnitude_term'][block].sum()/relative_energy) if relative_energy else None
                    row['direction_fraction_of_relative_displacement_energy'] = float(values['direction_term'][block].sum()/relative_energy) if relative_energy else None
                    row['prefill_angle'] = float(values['angle'][block,0])
                    row['decode_only_mean_angle'] = float(values['angle'][block,1:].mean())
                    rows.append(row)
                gains = values['gain_over_actual_injected_displacement'].mean(axis=1)
                for phase in ['linear_attention','full_attention']:
                    selected = [i for i in range(11,24) if blocks[i]==phase]
                    phase_rows.append(dict(cohort=cohort,prompt=prompt['id'],split=prompt['split'],condition=arm,
                        target_block_type=phase, transitions=len(selected),
                        mean_adjacent_change_in_injection_normalized_displacement=float(np.mean([gains[i]-gains[i-1] for i in selected]))))
        counts[cohort] = count
    depth = {}
    for arm in config['conditions']:
        trajectories = []
        for prompt in config['prompts']:
            with np.load(root/'native'/prompt['id']/(arm+'.npz')) as z:
                trajectories.append(z['residuals'][:,0,:].astype(np.float64))
        train, test = trajectories[:4], trajectories[4:]
        model = archived.fit(train, config['relative_ridge'])
        implicit = archived.evaluate(model, test)
        profile = np.mean([np.diff(h,axis=0) for h in train],axis=0)
        per_prompt=[]
        for prompt, h in zip(config['prompts'][4:], test):
            x,y = h[:-1],h[1:]
            target=y-x
            denominator=np.square(target).sum()
            row={'prompt':prompt['id'],'mean_transition_profile_score_vs_persistence':float(1-np.square(target-profile).sum()/denominator)}
            for kind in ['skew','free']:
                operator=cayley(model[kind])
                z=(x-model['center'])@model['basis']
                predicted=x+(z@(operator-np.eye(len(operator))))@model['basis'].T
                row[kind+'_causal_Cayley_score_vs_persistence']=float(1-np.square(y-predicted).sum()/denominator)
            per_prompt.append(row)
        depth[arm]={'per_prompt':per_prompt,'mean_transition_profile_score_vs_persistence':float(np.mean([r['mean_transition_profile_score_vs_persistence'] for r in per_prompt])),
                    'historical_implicit_midpoint_skew_score':implicit['skew_r2_vs_zero_velocity'],
                    'historical_implicit_midpoint_free_score':implicit['free_r2_vs_zero_velocity'],
                    'causal_skew_Cayley_score_vs_persistence':float(np.mean([r['skew_causal_Cayley_score_vs_persistence'] for r in per_prompt])),
                    'causal_free_Cayley_score_vs_persistence':float(np.mean([r['free_causal_Cayley_score_vs_persistence'] for r in per_prompt]))}
    for name,data in [('PROPAGATION_BY_PROMPT_BLOCK.csv', rows),('ARCHITECTURE_PHASES.csv',phase_rows)]:
        with (output/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    result={'paired_vectors_checked':counts,'identity_max_absolute_error':max_identity,
            'actual_injection_norm_vs_native_receipt_max_absolute_error':max_delivery_error,
            'invalid_denominators':0,'upstream_exactly_unchanged':True,
            'depth_position0_mean_transition_control':depth,'rows':rows,'architecture_phase_rows':phase_rows}
    (output/'PROPAGATION.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','architecture_phase_rows']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();run(args.root,args.output)
