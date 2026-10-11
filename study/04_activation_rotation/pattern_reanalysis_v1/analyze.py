"""Exploratory frozen-data response fingerprints; no model imports/inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ANCHOR=15
START=10

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def unit(x):
    n=np.linalg.norm(x,axis=-1,keepdims=True)
    assert np.all(n>1e-12)
    return x/n

def score(y,p):
    denominator=np.square(y).sum()
    return float(1-np.square(y-p).sum()/denominator) if denominator>1e-20 else None

def load(root,cohort,prompt,condition,manifest,used):
    p=root/cohort/prompt/(condition+'.npz')
    rel=p.relative_to(root).as_posix()
    digest=sha(p)
    assert manifest[rel]==digest, rel
    used[str(p)]=digest
    with np.load(p) as z:
        h=z['residuals'].astype(np.float64)
        tokens=z['forced_token_ids' if 'forced_token_ids' in z else 'generated_token_ids'].copy()
    assert h.shape[0]==24 and h.shape[2]==1024 and np.isfinite(h).all()
    return h,tokens

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def run(original,matched,output):
    output.mkdir(parents=True,exist_ok=False)
    plan=Path(__file__).with_name('ANALYSIS_PLAN.json')
    (output/'ANALYSIS_PLAN.json').write_bytes(plan.read_bytes())
    (output/'analyze.py').write_bytes(Path(__file__).read_bytes())
    old=json.loads((original/'source/pilot.json').read_text())
    panel=json.loads((matched/'native/PANEL.json').read_text())
    manifests={str(r):json.loads((r/'EXPORT_HASHES.json').read_text()) for r in [original,matched]}
    used={};training=[];bases=[];train_ids=[]
    for prompt in old['prompts']:
        if prompt['split']!='train':continue
        pid=prompt['id'];train_ids.append(pid)
        b,t=load(original,'native',pid,'base',manifests[str(original)],used)
        bases.append(b[ANCHOR,1:])
        profiles=[]
        for condition in ['existing_vector','random_vector']:
            c,tc=load(original,'native',pid,condition,manifests[str(original)],used)
            assert np.array_equal(t,tc) and np.array_equal(c[:START],b[:START])
            d=c-b;a=np.linalg.norm(d[START],axis=-1)
            assert np.all(a>1e-12)
            profiles.append((d[:,1:]/a[None,1:,None]).mean(axis=1))
        training.append(profiles)
    templates=np.asarray(training).mean(axis=0)
    # Fixed k=4 baseline TRAIN basis, no steered or evaluation state used.
    btrain=np.concatenate(bases,axis=0);center=btrain.mean(axis=0)
    _,singular,vt=np.linalg.svd(btrain-center,full_matrices=False)
    pcs=vt[:4].T
    coverage=float(np.square(singular[:4]).sum()/np.square(singular).sum())
    np.savez_compressed(output/'TRAIN_TEMPLATES.npz',templates=templates,baseline_center=center,baseline_pcs=pcs)
    rows=[];depth_rows=[];symmetry=[];pair_displacements={};baseline_rows=[]
    cohorts=[(original,c,'original',old['prompts']) for c in ['native','replication_seed_42','replication_seed_43','replication_seed_44']]
    cohorts.append((matched,'native','matched',panel['prompts']))
    for root,cohort,dataset,prompts in cohorts:
        for prompt in prompts:
            if dataset=='original' and prompt['split']!='test':continue
            pid=prompt['id'];pair=prompt.get('pair',pid)
            b,t=load(root,cohort,pid,'base',manifests[str(root)],used)
            bc=b[ANCHOR,1:]-center
            baseline_rows.append(dict(dataset=dataset,cohort=cohort,prompt=pid,pair=pair,
                baseline_energy_in_fixed_four_PC_basis=float(np.square(bc@pcs).sum()/np.square(bc).sum())))
            conditions=['existing_vector','random_vector'] if dataset=='original' else panel['conditions'][1:]
            deltas={}
            for condition in conditions:
                c,tc=load(root,cohort,pid,condition,manifests[str(root)],used)
                assert np.array_equal(t,tc) and np.array_equal(c[:START],b[:START])
                d=c-b;deltas[condition]=d
                if dataset=='matched':pair_displacements[(pid,condition)]=d
                a=np.linalg.norm(d[START],axis=-1)
                assert np.all(a>1e-12)
                x=d/a[None,:,None]
                is_template=condition in ['existing_vector','random_vector','plus_v','minus_v','random1']
                ti=0 if condition in ['existing_vector','plus_v','minus_v'] else 1
                sign=-1 if condition=='minus_v' else 1
                template=sign*templates[ti]
                wrong=sign*templates[1-ti]
                common=sign*templates.mean(axis=0)
                for phase,sl in [('decode',slice(1,None)),('prefill',slice(0,1))]:
                    for layer in range(START,24):
                        bb=b[layer,sl];cc=c[layer,sl];dd=d[layer,sl];xx=x[layer,sl]
                        bn=np.linalg.norm(bb,axis=-1);cn=np.linalg.norm(cc,axis=-1)
                        dot=np.einsum('ij,ij->i',bb,cc)
                        cos=np.clip(dot/(bn*cn),-1,1)
                        radial=np.einsum('ij,ij->i',dd,bb)/bn
                        orth=np.sqrt(np.maximum(np.square(dd).sum(axis=-1)-radial**2,0))
                        angles=np.arctan2(orth,dot/bn)
                        q=cn/bn;relative2=np.square(dd/bn[:,None]).sum(axis=-1)
                        direction=2*q*(1-cos);magnitude=(q-1)**2
                        assert np.max(np.abs(relative2-direction-magnitude))<1e-10
                        propagated=xx-x[START,sl]
                        cosine=float(np.mean(unit(xx)@unit(template[layer]))) if is_template else None
                        entry=dict(dataset=dataset,cohort=cohort,prompt=pid,pair=pair,condition=condition,phase=phase,
                            block_zero_based=layer,positions=len(bb),mean_angle_radians=float(angles.mean()),
                            mean_absolute_displacement=float(np.linalg.norm(dd,axis=-1).mean()),
                            mean_gain_over_actual_injection=float(np.linalg.norm(xx,axis=-1).mean()),
                            direction_fraction_of_relative_energy=float(direction.sum()/relative2.sum()),
                            mean_norm_ratio=float(q.mean()),
                            template_score_vs_zero=score(xx,template[layer]) if is_template else None,
                            unchanged_injection_score_vs_zero=score(xx,x[START,sl]),
                            wrong_template_score_vs_zero=score(xx,wrong[layer]) if is_template else None,
                            common_template_score_vs_zero=score(xx,common[layer]) if is_template else None,
                            mean_cosine_to_train_template=cosine,
                            additional_propagation_template_score=score(propagated,template[layer]-template[START]) if is_template else None,
                            propagation_energy_fraction=float(np.square(propagated).sum()/np.square(xx).sum()))
                        depth_rows.append(entry)
                        if layer==ANCHOR:
                            entry=entry.copy()
                            entry['displacement_energy_in_fixed_four_PC_basis']=float(np.square(dd@pcs).sum()/np.square(dd).sum())
                            if phase=='decode':
                                tangent=b[layer,1:]-b[layer,:-1]
                                tangent=unit(tangent)
                                entry['displacement_energy_along_baseline_token_tangent']=float(np.square(np.einsum('ij,ij->i',dd,tangent)).sum()/np.square(dd).sum())
                            else:entry['displacement_energy_along_baseline_token_tangent']=None
                            rows.append(entry)
            if dataset=='matched':
                dp=deltas['plus_v'];dm=deltas['minus_v']
                odd=(dp-dm)/2;even=(dp+dm)/2
                for phase,sl in [('decode',slice(1,None)),('prefill',slice(0,1))]:
                    for layer in range(START,24):
                        op=odd[layer,sl];ep=even[layer,sl];pp=dp[layer,sl];mm=dm[layer,sl]
                        energy=np.square(op).sum()+np.square(ep).sum()
                        assert abs(energy-(np.square(pp).sum()+np.square(mm).sum())/2)<1e-8
                        symmetry.append(dict(prompt=pid,pair=pair,phase=phase,block_zero_based=layer,
                            even_energy_fraction=float(np.square(ep).sum()/energy),
                            plus_minus_mirror_cosine=float(np.mean(np.einsum('ij,ij->i',unit(pp),unit(-mm)))),
                            minus_from_negative_plus_score_vs_zero=score(mm,-pp),
                            plus_to_minus_displacement_norm_ratio=float(np.linalg.norm(pp,axis=-1).mean()/np.linalg.norm(mm,axis=-1).mean())))
    # Same frozen neutral tokens permit a paired check of response directions.
    pair_rows=[]
    for pair in dict.fromkeys(p['pair'] for p in panel['prompts']):
        members={p['status']:p['id'] for p in panel['prompts'] if p['pair']==pair}
        for condition in panel['conditions'][1:]:
            dc=pair_displacements[(members['completed'],condition)][ANCHOR,1:]
            dp=pair_displacements[(members['pending'],condition)][ANCHOR,1:]
            pair_rows.append(dict(pair=pair,condition=condition,
                mean_response_cosine_across_status=float(np.mean(np.einsum('ij,ij->i',unit(dc),unit(dp)))),
                pending_from_completed_response_score_vs_zero=score(dp,dc),
                pending_over_completed_response_norm=float(np.linalg.norm(dp,axis=-1).mean()/np.linalg.norm(dc,axis=-1).mean())))
    # Architecture check uses prespecified targets, with only prompt summaries counted.
    phase_rows=[]
    groups={}
    for r in depth_rows:
        if r['phase']!='decode':continue
        key=(r['dataset'],r['cohort'],r['prompt'],r['condition'])
        groups.setdefault(key,{})[r['block_zero_based']]=r
    for (dataset,cohort,pid,condition),group in groups.items():
        for kind in ['full_attention','linear_attention']:
            layers=[i for i in range(11,24) if (i%4==3)==(kind=='full_attention')]
            phase_rows.append(dict(dataset=dataset,cohort=cohort,prompt=pid,condition=condition,
                target_block_type=kind,transitions=len(layers),
                mean_gain_increment=float(np.mean([group[i]['mean_gain_over_actual_injection']-group[i-1]['mean_gain_over_actual_injection'] for i in layers])),
                mean_angle_increment=float(np.mean([group[i]['mean_angle_radians']-group[i-1]['mean_angle_radians'] for i in layers]))))
    for name,data in [('ANCHOR_METRICS.csv',rows),('ALL_DOWNSTREAM_BLOCKS.csv',depth_rows),
                      ('SIGN_SYMMETRY.csv',symmetry),('MATCHED_STATUS_RESPONSE.csv',pair_rows),
                      ('BASELINE_PC_COVERAGE.csv',baseline_rows),('ARCHITECTURE_PHASES.csv',phase_rows)]:
        write_csv(output/name,data)
    result=dict(date='2026-10-11',analysis_plan_sha256=sha(plan),analysis_source_sha256=sha(Path(__file__)),
                template_train_prompts=train_ids,train_four_PC_baseline_variance_coverage=coverage,
                original_unique_test_prompts=4,original_continuation_cohorts=4,matched_unique_prompts=6,matched_independent_wording_pairs=3,
                consumed_native_files=len(used),all_consumed_native_hashes_verified=True,
                upstream_exactly_unchanged=True,no_new_model_inference=True,
                anchor_rows=rows,baseline_rows=baseline_rows,sign_symmetry=symmetry,matched_status_response=pair_rows,architecture=phase_rows,
                interpretation='Exploratory geometric replication on reused saved data; no confirmatory p-value or semantic/behavioral claim.')
    (output/'NATIVE_INPUT_HASHES.json').write_text(json.dumps(used,indent=2))
    (output/'RESULTS.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('original',type=Path);p.add_argument('matched',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();run(args.original,args.matched,args.output)
