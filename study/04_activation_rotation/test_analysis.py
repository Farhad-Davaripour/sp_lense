"""Scientific numerical checks; no model, GPU or service requests."""
import json
from pathlib import Path
import tempfile
import numpy as np
from analyze import fit, evaluate, diagnostic, main


def run():
    rng = np.random.default_rng(42)
    basis, _ = np.linalg.qr(rng.normal(size=(128, 2)))
    time = np.arange(64) * 0.12
    rotations = [np.column_stack([np.cos(time+p), np.sin(time+p)]) @ basis.T for p in np.linspace(0, 5, 8)]
    model = fit(rotations[:4])
    rotation = evaluate(model, rotations[4:])
    assert rotation['skew_r2_vs_zero_velocity'] > 0.95, rotation
    assert np.linalg.norm(model['skew'] + model['skew'].T) < 1e-8
    # A straight line can be smooth and change angle around the origin without
    # being a rotating trajectory. A constant velocity should win this case.
    drift = [np.arange(64)[:, None] * basis[:, 0][None, :] + p*basis[:, 1] for p in range(8)]
    straight = evaluate(fit(drift[:4]), drift[4:])
    assert straight['constant_r2_vs_zero_velocity'] > 0.999
    assert straight['skew_r2_vs_zero_velocity'] < 0.5, straight
    # New held-out motion outside the training span must count as unexplained.
    outside = rng.normal(size=128)
    outside -= basis @ (basis.T @ outside)
    outside /= np.linalg.norm(outside)
    changed = [h + 3*np.sin(time)[:, None]*outside for h in rotations[4:]]
    unobserved = evaluate(model, changed)
    assert unobserved['skew_r2_vs_zero_velocity'] < rotation['skew_r2_vs_zero_velocity'] - 0.4
    flat = [np.ones((16, 8)) for _ in range(4)]
    assert diagnostic(flat, flat, 0.01, 1, 1)['status'] == 'insufficient_velocity'
    # Exercise array loading, all three matched arms and both distinct axes.
    config = json.loads(Path(__file__).with_name('pilot.json').read_text())
    config['null_repetitions'] = 1  # Software check, not the frozen real analysis.
    with tempfile.TemporaryDirectory(prefix='research4_synthetic_') as temporary:
        directory = Path(temporary)
        for i, prompt in enumerate(config['prompts']):
            folder = directory/prompt['id']; folder.mkdir()
            phase = time[None, :] + np.arange(48)[:, None]*0.1 + i*0.7
            h = np.zeros((48, 64, 8))
            h[:,:,0], h[:,:,1] = np.cos(phase), np.sin(phase)
            for arm in config['conditions']:
                np.savez(folder/(arm+'.npz'), residuals=h, generated_token_ids=np.arange(64))
        main(directory, config)
        report = json.loads((directory/'ANALYSIS.json').read_text())
        assert len(report['conditions']) == 3
        assert len(report['conditions']['base']['depth_axis_fixed_prediction_positions']) == 4
        assert len(report['paired_token_axis_differences']['instruction_vector']) == 4
        # A fault can leave a counterbalanced treatment saved before its base.
        # Preserve that fact without crashing or claiming a matched comparison.
        (directory/config['prompts'][0]['id']/'base.npz').unlink()
        main(directory, config)
        partial = json.loads((directory/'ANALYSIS.json').read_text())
        assert len(partial['missing_matched_baselines']) == 2
        assert len(partial['conditions']['instruction_vector']['traces']) == 7
    print(json.dumps({'synthetic_checks': 'passed',
        'known_rotation_score': rotation['skew_r2_vs_zero_velocity'],
        'straight_drift_score': straight['skew_r2_vs_zero_velocity'],
        'unseen_motion_score': unobserved['skew_r2_vs_zero_velocity']}))


if __name__ == '__main__':
    run()
