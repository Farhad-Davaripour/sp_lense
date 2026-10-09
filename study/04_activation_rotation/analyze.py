"""Full-width exploratory rotation diagnostic; never loads a language model."""
import argparse
import json
from pathlib import Path
import numpy as np


def pairs(trajectories, center):
    # Midpoint finite differences avoid confusing discrete circular chords with
    # a radial contraction. Never connect the end of one prompt to the next.
    sources, velocities = [], []
    for trajectory in trajectories:
        h = np.asarray(trajectory, dtype=np.float64) - center
        sources.append((h[:-1] + h[1:]) / 2)
        velocities.append(h[1:] - h[:-1])
    return np.concatenate(sources), np.concatenate(velocities)


def training_span(states):
    # Complete numerically supported row span; no top-k PCA truncation.
    gram = states @ states.T
    values, axes = np.linalg.eigh(gram)
    tolerance = max(gram.shape) * np.finfo(np.float64).eps * max(values[-1], 1e-30)
    keep = values > tolerance
    if not keep.any():
        raise ValueError('Training states have zero numerical rank')
    return states.T @ (axes[:, keep] / np.sqrt(values[keep]))


def fit(train, relative_ridge=0.01):
    center = np.concatenate(train).astype(np.float64).mean(axis=0)
    states = np.concatenate(train).astype(np.float64) - center
    basis = training_span(states)
    x, v = pairs(train, center)
    z, velocity = x @ basis, v @ basis
    covariance = z.T @ z
    c, u = np.linalg.eigh(covariance)
    c = np.maximum(c, 0)
    ridge = relative_ridge * max(float(c.sum()) / len(c), 1e-30)
    cross = z.T @ velocity
    # Exact skew-constrained ridge solution in the complete training span:
    # C A + A C + 2 lambda A = X'V - V'X, A' = -A.
    rhs = u.T @ (cross - cross.T) @ u
    skew = u @ (rhs / (c[:, None] + c[None, :] + 2 * ridge)) @ u.T
    free = u @ ((u.T @ cross) / (c[:, None] + ridge))
    return {'center': center, 'basis': basis, 'skew': skew, 'free': free,
            'constant': v.mean(axis=0), 'ridge': ridge}


def evaluate(model, test):
    rows = []
    for trajectory in test:
        x, v = pairs([trajectory], model['center'])
        denominator = float(np.square(v).sum())
        if denominator <= 1e-24:
            rows.append({'valid': False, 'reason': 'zero_velocity'})
            continue
        z = x @ model['basis']
        row = {'valid': True, 'steps': len(v), 'velocity_energy': denominator}
        for kind in ['skew', 'free']:
            # Error includes ALL original coordinates, including held-out
            # directions outside the training span. No test-space refitting.
            predicted = (z @ model[kind]) @ model['basis'].T
            row[kind + '_r2_vs_zero_velocity'] = 1 - float(np.square(v-predicted).sum()) / denominator
        row['constant_r2_vs_zero_velocity'] = 1 - float(np.square(v-model['constant']).sum()) / denominator
        norms = np.linalg.norm(trajectory, axis=1)
        dot = np.einsum('ij,ij->i', trajectory[:-1], trajectory[1:])
        valid = norms[:-1] * norms[1:] > 1e-24
        angles = np.arccos(np.clip(dot[valid] / (norms[:-1]*norms[1:])[valid], -1, 1))
        row['mean_norm'] = float(norms.mean())
        row['mean_adjacent_angle_radians'] = float(angles.mean()) if len(angles) else None
        rows.append(row)
    valid = [r for r in rows if r['valid']]
    result = {'per_trajectory': rows, 'valid_trajectories': len(valid),
              'training_span_rank': model['basis'].shape[1],
              'full_width': model['basis'].shape[0], 'ridge': model['ridge']}
    for key in ['skew_r2_vs_zero_velocity', 'free_r2_vs_zero_velocity', 'constant_r2_vs_zero_velocity']:
        result[key] = float(np.mean([r[key] for r in valid])) if valid else None
    return result


def diagnostic(train, test, ridge, repetitions, seed):
    if len(train) < 3 or len(test) < 3:
        return {'status': 'insufficient_data', 'train': len(train), 'test': len(test)}
    if sum(float(np.square(np.diff(h, axis=0)).sum()) for h in train) <= 1e-24:
        return {'status': 'insufficient_velocity'}
    observed = evaluate(fit(train, ridge), test)
    if observed['skew_r2_vs_zero_velocity'] is None:
        return {'status': 'insufficient_velocity', **observed}
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(repetitions):
        shuffled_train = [h[rng.permutation(len(h))] for h in train]
        shuffled_test = [h[rng.permutation(len(h))] for h in test]
        score = evaluate(fit(shuffled_train, ridge), shuffled_test)['skew_r2_vs_zero_velocity']
        if score is not None:
            nulls.append(score)
    observed['order_permutation_scores'] = nulls
    observed['order_permutation_upper_tail_fraction'] = (1 + sum(s >= observed['skew_r2_vs_zero_velocity'] for s in nulls)) / (1 + len(nulls))
    observed['status'] = 'exploratory_only'
    return observed


def main(directory, config):
    output = {'study': config['study'], 'axes_separate': True, 'conditions': {}}
    for arm in config['conditions']:
        train, test, ids, temporal_ids = [], [], [], {'train': [], 'test': []}
        depth = {str(a): {'train': [], 'test': []} for a in config['depth_prediction_positions']}
        for prompt in config['prompts']:
            path = directory / prompt['id'] / (arm + '.npz')
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as archive:
                h = archive['residuals'].astype(np.float64)  # layer, prediction step, width
                token_ids = archive['generated_token_ids']
                if arm != 'base':
                    with np.load(directory / prompt['id'] / 'base.npz', allow_pickle=False) as baseline:
                        if not np.array_equal(token_ids, baseline['generated_token_ids']):
                            raise ValueError('Matched-token replay mismatch')
            if h.ndim != 3 or not np.isfinite(h).all():
                raise ValueError('Invalid activation trace')
            temporal = h[config['primary_temporal_layer_zero_based']]
            destination = train if prompt['split'] == 'train' else test
            if len(temporal) >= 8:
                destination.append(temporal)
                temporal_ids[prompt['split']].append(prompt['id'])
            anchors = [a for a in config['depth_prediction_positions'] if a < h.shape[1]]
            for anchor in anchors:
                depth[str(anchor)][prompt['split']].append(h[:, anchor, :])
            ids.append({'id': prompt['id'], 'split': prompt['split'], 'tokens': h.shape[1], 'depth_anchors': anchors})
        args = (config['relative_ridge'], config['null_repetitions'], config['random_seed'])
        output['conditions'][arm] = {'traces': ids,
            'temporal_prompt_ids': temporal_ids,
            'token_axis_block32': diagnostic(train, test, *args),
            'depth_axis_fixed_prediction_positions': {a: diagnostic(rows['train'], rows['test'], *args)
                                                     for a, rows in depth.items()}}
    output['paired_token_axis_differences'] = {}
    base = output['conditions']['base']
    for arm in config['conditions'][1:]:
        treatment = output['conditions'][arm]
        if base['temporal_prompt_ids'] != treatment['temporal_prompt_ids']:
            output['paired_token_axis_differences'][arm] = {'status': 'unmatched_missing_data'}
            continue
        b = base['token_axis_block32'].get('per_trajectory', [])
        t = treatment['token_axis_block32'].get('per_trajectory', [])
        output['paired_token_axis_differences'][arm] = [
            {'prompt': prompt, 'skew_score_delta': tr['skew_r2_vs_zero_velocity']-br['skew_r2_vs_zero_velocity']}
            for prompt, br, tr in zip(base['temporal_prompt_ids']['test'], b, t)
            if br['valid'] and tr['valid']]
    output['interpretation'] = 'Pilot geometry only; order shuffles do not preserve temporal covariance. No brain equivalence or behavioral claim.'
    (directory / 'ANALYSIS.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({'saved': str(directory / 'ANALYSIS.json'), 'conditions': list(output['conditions'])}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('pilot.json'))
    args = parser.parse_args()
    main(args.directory, json.loads(args.config.read_text(encoding='utf-8')))
