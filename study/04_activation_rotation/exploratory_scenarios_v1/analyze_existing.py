"""Exploratory scenario comparisons using preserved traces; CPU only."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

TYPES = ['ordinary_explanation', 'arithmetic_planning',
         'fictional_completed_work', 'fictional_pending_deadline']
COHORTS = ['native', 'replication_seed_42', 'replication_seed_43', 'replication_seed_44']


def describe(h):
    norms = np.linalg.norm(h, axis=1)
    differences = np.diff(h, axis=0)
    speeds = np.linalg.norm(differences, axis=1)
    products = norms[:-1]*norms[1:]
    valid = products > 1e-24
    angles = np.arccos(np.clip(np.einsum('ij,ij->i', h[:-1], h[1:])[valid]/products[valid], -1, 1))
    products = speeds[:-1]*speeds[1:]
    valid = products > 1e-24
    bends = np.arccos(np.clip(np.einsum('ij,ij->i', differences[:-1], differences[1:])[valid]/products[valid], -1, 1))
    return {'mean_norm': float(norms.mean()), 'norm_std': float(norms.std()),
            'mean_adjacent_angle_radians': float(angles.mean()) if len(angles) else None,
            'mean_velocity_turn_radians': float(bends.mean()) if len(bends) else None,
            'velocity_rms': float(np.sqrt(np.square(speeds).mean())),
            'net_displacement_over_path_length': float(np.linalg.norm(h[-1]-h[0])/speeds.sum()) if speeds.sum() else None}


def main(root, output):
    output.mkdir(parents=True, exist_ok=False)
    # Use the exact diagnostic source archived with the completed experiment.
    spec = importlib.util.spec_from_file_location('frozen_rotation_analysis', root/'source/analyze.py')
    diagnostic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    config = json.loads((root/'source/pilot.json').read_text())
    labels = {p['id']: TYPES[i % 4] for i, p in enumerate(config['prompts'])}
    rows, delivery, missing, verification = [], [], [], []
    original_hashes = json.loads((root/'EXPORT_HASHES.json').read_text())
    for cohort in COHORTS:
        original = json.loads((root/cohort/'ANALYSIS.json').read_text())
        arrays = {}
        for arm in config['conditions']:
            arrays[arm] = {}
            for prompt in config['prompts']:
                path = root/cohort/prompt['id']/(arm+'.npz')
                if not path.exists():
                    missing.append({'cohort': cohort, 'prompt': prompt['id'], 'condition': arm})
                    continue
                assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hashes[str(path.relative_to(root)).replace('\\', '/')]
                with np.load(path, allow_pickle=False) as archive:
                    arrays[arm][prompt['id']] = archive['residuals'].astype(np.float64)
            for axis, position in [('token_block16', 15), ('depth_position_0', 0),
                                   ('depth_position_15', 15), ('depth_position_31', 31), ('depth_position_47', 47)]:
                available = [p for p in config['prompts'] if p['id'] in arrays[arm]
                             and (axis.startswith('token') or arrays[arm][p['id']].shape[1] > position)]
                trajectories = {p['id']: (arrays[arm][p['id']][position] if axis.startswith('token')
                                        else arrays[arm][p['id']][:, position]) for p in available}
                train = [trajectories[p['id']] for p in available if p['split'] == 'train']
                model = diagnostic.fit(train, config['relative_ridge'])
                evaluated = diagnostic.evaluate(model, [trajectories[p['id']] for p in available])
                old = (original['conditions'][arm]['token_axis_block16'] if axis.startswith('token')
                       else original['conditions'][arm]['depth_axis_fixed_prediction_positions'][str(position)])
                test_index = 0
                for prompt, fitted in zip(available, evaluated['per_trajectory']):
                    if prompt['split'] == 'test':
                        for key in ['skew_r2_vs_zero_velocity', 'free_r2_vs_zero_velocity', 'constant_r2_vs_zero_velocity']:
                            assert abs(fitted[key]-old['per_trajectory'][test_index][key]) < 1e-12
                        test_index += 1
                    features = {**fitted, **describe(trajectories[prompt['id']])}
                    rows.append(dict(cohort=cohort, prompt=prompt['id'], scenario_type=labels[prompt['id']],
                        split=prompt['split'], condition=arm, axis=axis,
                        score_role='held_out' if prompt['split']=='test' else 'in_sample_descriptive_only',
                        **features))
                verification.append({'cohort':cohort,'condition':arm,'axis':axis,
                                     'original_held_out_scores_reproduced':test_index})
        for prompt in config['prompts']:
            base = arrays['base'].get(prompt['id'])
            if base is None:
                continue
            for arm in config['conditions'][1:]:
                changed = arrays[arm].get(prompt['id'])
                if changed is None:
                    continue
                assert changed.shape == base.shape
                difference = changed-base
                for block in range(len(base)):
                    delivery.append({'cohort':cohort,'prompt':prompt['id'], 'scenario_type':labels[prompt['id']],
                        'split':prompt['split'],'condition':arm,'block_zero_based':block,
                        'mean_L2_difference':float(np.linalg.norm(difference[block],axis=1).mean()),
                        'mean_relative_L2_difference':float((np.linalg.norm(difference[block],axis=1)/np.linalg.norm(base[block],axis=1)).mean()),
                        'max_absolute_coordinate_difference':float(np.abs(difference[block]).max())})
    for name, data in [('PER_PROMPT_AXES.csv', rows), ('STEERING_BY_BLOCK.csv', delivery)]:
        with (output/name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    result = {'exploratory_after_results':True, 'source_cohorts_unchanged':True,
              'group_fit_refitted':False, 'original_global_train_split_preserved':True,
              'unique_held_out_prompts_per_type':1, 'additional_prompts_executed':0,
              'missing_attempted_original_traces':missing,'score_reproduction_checks':verification,
              'per_prompt_axes':rows,'steering_by_block':delivery,
              'groups':{}}
    for cohort in COHORTS:
        result['groups'][cohort] = {}
        for group, types in [('ordinary', TYPES[:2]), ('fictional', TYPES[2:])]:
            selected = [r for r in rows if r['cohort']==cohort and r['scenario_type'] in types
                        and r['split']=='test' and r['axis']=='token_block16']
            result['groups'][cohort][group] = {arm:{key:float(np.mean([r[key] for r in selected if r['condition']==arm]))
                for key in ['skew_r2_vs_zero_velocity','mean_norm','mean_adjacent_angle_radians',
                            'mean_velocity_turn_radians','net_displacement_over_path_length']}
                for arm in config['conditions']}
    (output/'PATTERNS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'rows':len(rows),'steering_rows':len(delivery),'missing':missing,'groups':result['groups']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    main(args.root, args.output)
