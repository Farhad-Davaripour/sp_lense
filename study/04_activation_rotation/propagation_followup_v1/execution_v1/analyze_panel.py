"""Frozen panel primary contrast, evaluated on CPU after owned-GPU release."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def run(root, output):
    output.mkdir(exist_ok=False)
    specification=json.loads((root/'native/PANEL.json').read_text())
    status=json.loads((root/'native/STATUS.json').read_text())
    mechanics=json.loads((root/'mechanics_probes/STATUS.json').read_text())
    assert mechanics['status']=='passed' and len(mechanics['completed'])==6
    assert status['null_hook_logits_exact'] and status['validation_probes_completed']==2
    manifest=json.loads((root/'EXPORT_HASHES.json').read_text())
    rows=[];missing=[];max_upstream=0.;max_match=0.;max_norm_error=0.;max_norm_site_angle=0.
    for prompt in specification['prompts']:
        folder=root/'native'/prompt['id']
        if not (folder/'base.npz').exists():
            missing.append({'prompt':prompt['id'],'condition':'base'})
            for condition in specification['conditions'][1:]:
                missing.append({'prompt':prompt['id'],'condition':condition,
                                'reason':'unmatched_missing_baseline' if (folder/(condition+'.npz')).exists() else 'trace_missing'})
            continue
        with np.load(folder/'base.npz') as z:
            base=z['residuals'].astype(np.float64);pre=z['pre_injection']
        for condition in specification['conditions']:
            path=folder/(condition+'.npz')
            if not path.exists():missing.append({'prompt':prompt['id'],'condition':condition});continue
            assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest[str(path.relative_to(root)).replace('\\','/')]
            with np.load(path) as z:
                assert np.array_equal(z['forced_token_ids'],specification['forced_continuation_ids'])
                assert np.array_equal(z['pre_injection'],pre)
                changed=z['residuals'].astype(np.float64)
            assert base.shape==changed.shape==(24,28,1024)
            assert np.isfinite(changed).all()
            upstream=float(np.max(np.abs(changed[:10]-base[:10])));max_upstream=max(max_upstream,upstream)
            assert upstream==0
            b=base[15];c=changed[15];bn=np.linalg.norm(b,axis=1);cn=np.linalg.norm(c,axis=1)
            assert np.all(bn>1e-12) and np.all(cn>1e-12)
            cosine=np.clip(np.einsum('ij,ij->i',b,c)/(bn*cn),-1,1)
            # Identical baseline defines exact zero, avoiding acos roundoff.
            angles=np.zeros(len(b)) if condition=='base' else np.arccos(cosine)
            absolute=np.linalg.norm(c-b,axis=1)
            row={'prompt':prompt['id'],'pair':prompt['pair'],'status':prompt['status'],'condition':condition,
                 'primary_mean_angle_radians':float(angles.mean()),'decode_only_mean_angle_radians':float(angles[1:].mean()),
                 'prefill_angle_radians':float(angles[0]),'mean_norm_ratio':float((cn/bn).mean()),
                 'mean_absolute_displacement':float(absolute.mean()),'mean_relative_displacement':float((absolute/bn).mean())}
            rows.append(row)
            record=next(r for r in status['completed'] if r['prompt']==prompt['id'] and r['condition']==condition)
            if record['delivery']:
                audit=np.asarray(record['delivery'])
                if condition=='norm_only':
                    max_norm_error=max(max_norm_error,float(audit[:,3].max()));max_norm_site_angle=max(max_norm_site_angle,float(audit[:,4].max()))
                else:max_match=max(max_match,float(audit[:,2].max()))
    pairs=[]
    for pair in dict.fromkeys(p['pair'] for p in specification['prompts']):
        subset=[r for r in rows if r['pair']==pair]
        differences={}
        for condition in specification['conditions']:
            selected={r['status']:r for r in subset if r['condition']==condition}
            differences[condition]=(selected['pending']['primary_mean_angle_radians']-selected['completed']['primary_mean_angle_radians']) if set(selected)=={'pending','completed'} else None
        controls=[differences[c] for c in ['random1','random2','norm_only']]
        support=(differences['plus_v']>0 and all(differences['plus_v']>v for v in controls)) if None not in controls+[differences['plus_v']] else None
        pairs.append({'pair':pair,'pending_minus_completed_angle_radians':differences,'predeclared_pair_support':support})
    hypothesis=all(p['predeclared_pair_support'] is True for p in pairs) if not missing else None
    result={'study':specification['study'],'completed_study_traces':len(status['completed']),
            'paired_traces_scored':len(rows),'scheduled_study_traces':36,
            'mechanics_probes_passed':6,'null_untreated_probe_calls_passed':2,'missing':missing,
            'primary_metric':'mean angle at zero-based block15, all 28 positions, prompt level',
            'pairs':pairs,'predeclared_three_pair_hypothesis_survives':hypothesis,
            'max_upstream_coordinate_difference':max_upstream,'max_additive_relative_native_matching_error':max_match,
            'max_norm_only_relative_native_norm_error':max_norm_error,'max_norm_only_site_angle_radians':max_norm_site_angle,
            'rows':rows,'interpretation':'Exploratory fixed-token angular response; no behavioral self-preservation inference.'}
    (output/'PANEL_ANALYSIS.json').write_text(json.dumps(result,indent=2))
    with (output/'PROMPT_CONDITION_METRICS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();run(args.root,args.output)
