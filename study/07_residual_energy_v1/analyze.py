"""Exploratory, CPU-only baseline-referenced scalar energies; see frozen PLAN.json."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

SOURCE = Path(__file__).parent
OUT = Path('C:/Users/farha/Research4/artifacts/residual_energy_v1/analysis_20261012')
ORIGINAL = Path('C:/Users/farha/Research4/artifacts/rotation_pilot_v1/executed_20261009/verified_native')
MATCHED = Path('C:/Users/farha/Research4/artifacts/propagation_followup_v1/recovery_20261010/verified_export/verified_native')
KINDS = ['norm', 'diagonal', 'quadratic']
D = 1024

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def dump(p, obj):
    p.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')

def csvwrite(p, rows):
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def fit(x):
    mu = x.mean(axis=0)
    xc = x-mu
    _, s, vt = np.linalg.svd(xc/np.sqrt(len(x)), full_matrices=False)
    rank = int(np.sum(s > max(xc.shape)*np.finfo(float).eps*s[0]))
    v = vt[:rank].T
    eigen = s[:rank]**2
    var = np.mean(xc**2, axis=0)
    tau = float(var.mean())
    m = dict(mu=mu, v=v, eigen=eigen, var=var, tau=tau, rank=rank)
    m['calibration'] = raw_energy(x, m)[0].mean(axis=1)
    return m

def raw_energy(x, m):
    xc = x-m['mu']
    projection = xc@m['v']
    outside = np.maximum(np.sum(xc**2, axis=1)-np.sum(projection**2, axis=1), 0)
    within_term = np.sum(projection**2/(.9*m['eigen']+.1*m['tau']), axis=1)/D
    outside_term = outside/(.1*m['tau']*D)
    energies = np.stack([np.mean(x*x, axis=1),
        np.mean(xc*xc/(.9*m['var']+.1*m['tau']), axis=1),
        within_term+outside_term])
    return energies, outside_term

def check_implementation():
    rng = np.random.default_rng(20261012)
    # Exercise the same full-dimensional inverse formula with a small rank.
    train = rng.standard_normal((7, D))
    m = fit(train)
    x = rng.standard_normal((3, D)); delta = rng.standard_normal((3, D))*.1
    e = raw_energy(x, m)[0][2]
    ep = raw_energy(x+delta, m)[0][2]
    em = raw_energy(x-delta, m)[0][2]
    precision_delta = raw_energy(delta+m['mu'], m)[0][2]
    parity_error = float(np.max(np.abs(ep+em-2*e-2*precision_delta)))
    # Independently check a small direct covariance inverse.
    small = rng.standard_normal((7, 11)); center = small-small.mean(axis=0)
    cov = center.T@center/len(small); tau = np.trace(cov)/11
    _, s, vt = np.linalg.svd(center/np.sqrt(len(small)), full_matrices=False)
    v = vt[s>1e-12].T; eig = s[s>1e-12]**2
    z = rng.standard_normal((3, 11)); proj = z@v
    spectral = np.sum(proj*proj/(.9*eig+.1*tau), axis=1)+(np.sum(z*z, axis=1)-np.sum(proj*proj, axis=1))/(.1*tau)
    direct = np.sum(z*np.linalg.solve(.9*cov+.1*tau*np.eye(11), z.T).T, axis=1)
    inverse_error = float(np.max(np.abs(spectral-direct)))
    rotation_error = abs(np.dot([1., 0.], [1., 0.])-np.dot([0., 1.], [0., 1.]))
    assert parity_error<1e-10 and inverse_error<1e-10 and rotation_error==0
    return dict(quadratic_symmetric_identity_max_error=parity_error,
        direct_inverse_max_error=inverse_error, norm_rotation_blindness_error=rotation_error)

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    for name in ['PLAN.json', 'analyze.py']:
        shutil.copyfile(SOURCE/name, OUT/name)
    manifests = {str(r):json.loads((r/'EXPORT_HASHES.json').read_text()) for r in [ORIGINAL, MATCHED]}
    used = {}; cache = {}
    def read(root, cohort, pid, arm, full=False):
        key = (str(root), cohort, pid, arm, full)
        if key in cache: return cache[key]
        p = root/cohort/pid/(arm+'.npz')
        digest = sha(p)
        assert digest==manifests[str(root)][p.relative_to(root).as_posix()]
        used[str(p)] = digest
        with np.load(p, allow_pickle=False) as f:
            h = f['residuals'].astype(np.float64)
            tokens = f['generated_token_ids' if 'generated_token_ids' in f else 'forced_token_ids'].copy()
        assert h.shape[0]==24 and h.shape[2]==D and np.isfinite(h).all()
        result = (h if full else h[:, :8].copy(), tokens if full else tokens[:8].copy())
        cache[key] = result
        return result
    train = [read(ORIGINAL, 'native', 'r4_0'+str(i), 'base', True)[0] for i in range(1, 5)]
    assert all(x.shape==(24, 64, D) for x in train)
    models = {}; references = []
    for phase, sl in [('prefill', slice(0, 1)), ('decode', slice(1, None))]:
        for layer in range(24):
            x = np.concatenate([h[layer, sl] for h in train])
            m = fit(x); models[phase, layer] = m
            references.append(dict(phase=phase, layer_zero_based=layer, samples=len(x), rank=m['rank'],
                unobserved_dimensions=D-m['rank'], tau=m['tau'], calibration=dict(zip(KINDS, m['calibration'].tolist()))))
        print('FIT', phase, flush=True)
    def score(h):
        e = np.empty((3, 24, 8)); outside = np.empty((24, 8))
        for layer in range(24):
            for phase, sl in [('prefill', slice(0, 1)), ('decode', slice(1, 8))]:
                m = models[phase, layer]
                a, o = raw_energy(h[layer, sl], m)
                e[:, layer, sl] = a/m['calibration'][:, None]
                outside[layer, sl] = o/m['calibration'][2]
        assert np.isfinite(e).all() and np.min(e)>=0
        return e, outside
    settings = []
    for cohort in ['native', 'replication_seed_42', 'replication_seed_43', 'replication_seed_44']:
        for arm in ['existing_vector', 'random_vector']:
            settings.append(('original_TEST', ORIGINAL, cohort, arm, [('r4_0'+str(i), 'r4_0'+str(i)) for i in range(5, 9)]))
    panel = json.loads((MATCHED/'native/PANEL.json').read_text())
    matched_prompts = [(p['id'], p['pair']) for p in panel['prompts']]
    for arm in ['plus_v', 'minus_v', 'random1', 'random2', 'norm_only']:
        settings.append(('older_matched_OOD', MATCHED, 'native', arm, matched_prompts))
    rows = []; layer_rows = []; tensors = {}; upstream_max = 0.
    for dataset, root, cohort, arm, prompts in settings:
        for pid, group in prompts:
            b, bt = read(root, cohort, pid, 'base'); s, st = read(root, cohort, pid, arm)
            assert np.array_equal(bt, st) and np.array_equal(b[:10], s[:10])
            be, bo = score(b); se, so = score(s)
            upstream_max = max(upstream_max, float(np.max(np.abs(se[:, :10]-be[:, :10]))))
            key = '__'.join([dataset, cohort, pid, arm])
            tensors[key+'__base'] = be; tensors[key+'__steered'] = se
            for k, kind in enumerate(KINDS):
                for region, layers in [('global', slice(None)), ('downstream', slice(10, None)), ('block16', slice(15, 16))]:
                    for phase, positions in [('all8', slice(None)), ('prefill', slice(0, 1)), ('decode7', slice(1, 8))]:
                        base = float(be[k, layers, positions].mean()); steered = float(se[k, layers, positions].mean())
                        rows.append(dict(dataset=dataset, cohort=cohort, prompt=pid, group=group, condition=arm,
                            energy=kind, region=region, phase=phase, base=base, steered=steered,
                            delta=steered-base, relative_change_percent=100*(steered-base)/base,
                            quadratic_base_outside_fraction=float(bo[layers, positions].sum()/be[2, layers, positions].sum()),
                            quadratic_steered_outside_fraction=float(so[layers, positions].sum()/se[2, layers, positions].sum())))
                for layer in range(24):
                    layer_rows.append(dict(dataset=dataset, cohort=cohort, prompt=pid, condition=arm, energy=kind,
                        layer_zero_based=layer, base=float(be[k, layer].mean()), steered=float(se[k, layer].mean()),
                        delta=float((se[k, layer]-be[k, layer]).mean())))
        print('SCORE', dataset, cohort, arm, flush=True)
    assert upstream_max==0
    summary = []
    for dataset, root, cohort, arm, prompts in settings:
        for kind in KINDS:
            a = [r for r in rows if r['dataset']==dataset and r['cohort']==cohort and r['condition']==arm and
                r['energy']==kind and r['region']=='global' and r['phase']=='all8']
            base = np.array([r['base'] for r in a]); steer = np.array([r['steered'] for r in a])
            comparison = steer[:, None]-base[None, :]
            summary.append(dict(dataset=dataset, cohort=cohort, condition=arm, energy=kind, prompts=len(a),
                independent_context_groups=len(set(r['group'] for r in a)), paired_energy_increases=int(np.sum(steer>base)),
                paired_energy_decreases=int(np.sum(steer<base)), base_range=[float(base.min()), float(base.max())],
                steered_range=[float(steer.min()), float(steer.max())],
                mean_relative_change_percent=float(np.mean([r['relative_change_percent'] for r in a])),
                min_relative_change_percent=float(min(r['relative_change_percent'] for r in a)),
                max_relative_change_percent=float(max(r['relative_change_percent'] for r in a)),
                AUROC_high_energy_positive=float(np.mean((comparison>0)+.5*(comparison==0))),
                quadratic_mean_base_outside_fraction=float(np.mean([r['quadratic_base_outside_fraction'] for r in a]))))
    integrity = {}
    for root in [ORIGINAL, MATCHED]:
        checked = 0
        for member, digest in manifests[str(root)].items():
            assert sha(root/member)==digest, member
            checked += 1
        integrity[str(root)] = checked
    dump(OUT/'REFERENCE_SUMMARY.json', references)
    dump(OUT/'USED_SOURCE_HASHES.json', used)
    csvwrite(OUT/'PROMPT_SCORES.csv', rows); csvwrite(OUT/'LAYER_SCORES.csv', layer_rows)
    np.savez_compressed(OUT/'ENERGY_TENSORS.npz', **tensors)
    dump(OUT/'RESULTS.json', dict(plan_sha256=sha(SOURCE/'PLAN.json'), implementation_sha256=sha(SOURCE/'analyze.py'),
        scope='Exploratory statistical energy audit; no physical/Lyapunov or general detection guarantee',
        checks=check_implementation(), exact_upstream_energy_difference=upstream_max,
        complete_native_manifest_files_verified=integrity, fresh_panel_accessed=False, GPU_or_model_calls=False,
        primary_global_all8=summary))
    print('DONE', OUT, flush=True)

if __name__=='__main__':
    main()
