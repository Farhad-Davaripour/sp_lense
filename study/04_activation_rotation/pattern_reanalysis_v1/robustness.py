"""Sensitivity checks and matched-pair crossvalidation on already saved traces."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze import load,score,unit,write_csv,ANCHOR,START

def run(original,matched,analysis):
    output=analysis/'robustness';output.mkdir(exist_ok=False)
    for name in ['ROBUSTNESS_PLAN.json','robustness.py']:
        (output/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    old=json.loads((original/'source/pilot.json').read_text())
    panel=json.loads((matched/'native/PANEL.json').read_text())
    manifests={str(r):json.loads((r/'EXPORT_HASHES.json').read_text()) for r in [original,matched]}
    used={};train=[];cache={}
    for prompt in old['prompts']:
        if prompt['split']!='train':continue
        b,t=load(original,'native',prompt['id'],'base',manifests[str(original)],used)
        profiles=[]
        for arm in ['existing_vector','random_vector']:
            c,tc=load(original,'native',prompt['id'],arm,manifests[str(original)],used)
            assert np.array_equal(t,tc)
            d=c-b;a=np.linalg.norm(d[START],axis=-1)
            profiles.append((d[ANCHOR,1:]/a[1:,None]).mean(axis=0))
        train.append(profiles)
    train=np.asarray(train)
    with np.load(analysis/'TRAIN_TEMPLATES.npz') as z:template=z['templates'][:,ANCHOR].copy()
    jackknife=[];windows=[]
    cohorts=[(original,c,'original',old['prompts']) for c in ['native','replication_seed_42','replication_seed_43','replication_seed_44']]
    cohorts.append((matched,'native','matched',panel['prompts']))
    for root,cohort,dataset,prompts in cohorts:
        for prompt in prompts:
            if dataset=='original' and prompt['split']!='test':continue
            pid=prompt['id'];b,t=load(root,cohort,pid,'base',manifests[str(root)],used)
            arms=['existing_vector','random_vector'] if dataset=='original' else panel['conditions'][1:]
            for arm in arms:
                c,tc=load(root,cohort,pid,arm,manifests[str(root)],used)
                assert np.array_equal(t,tc)
                d=c-b;x=d[ANCHOR]/np.linalg.norm(d[START],axis=-1)[:,None]
                if dataset=='matched':cache[(pid,arm)]=dict(x=x,d=d[ANCHOR])
                if arm not in ['existing_vector','random_vector','plus_v','minus_v','random1']:continue
                ti=0 if arm in ['existing_vector','plus_v','minus_v'] else 1;sign=-1 if arm=='minus_v' else 1
                y=x[1:]
                for omitted in range(4):
                    p=sign*np.delete(train,omitted,axis=0).mean(axis=0)[ti]
                    jackknife.append(dict(dataset=dataset,cohort=cohort,prompt=pid,condition=arm,
                        omitted_train_prompt=old['prompts'][omitted]['id'],score_vs_zero=score(y,p)))
                cut=(len(y)+1)//2
                for name,z in [('first_decode_half',y[:cut]),('second_decode_half',y[cut:])]:
                    p=sign*template[ti];token_score=1-np.square(z-p).sum(axis=1)/np.square(z).sum(axis=1)
                    windows.append(dict(dataset=dataset,cohort=cohort,prompt=pid,condition=arm,window=name,
                        score_vs_zero=score(z,p),fraction_token_scores_positive=float(np.mean(token_score>0)),
                        minimum_token_score=float(token_score.min()),maximum_token_score=float(token_score.max())))
    pairs=list(dict.fromkeys(p['pair'] for p in panel['prompts']));arms=panel['conditions'][1:]
    crossval=[];cosine_rows=[];symmetry=[]
    for pair in pairs:
        trainp=[p['id'] for p in panel['prompts'] if p['pair']!=pair]
        testp=[p['id'] for p in panel['prompts'] if p['pair']==pair]
        learned={a:np.mean([cache[(pid,a)]['x'][1:] for pid in trainp],axis=0) for a in arms}
        for pid in testp:
            for arm in arms:
                y=cache[(pid,arm)]['x'][1:];p=learned[arm]
                common=np.mean([learned[a] for a in arms if a!=arm],axis=0)
                crossval.append(dict(pair=pair,prompt=pid,condition=arm,
                    position_matched_same_condition_score=score(y,p),
                    position_matched_other_conditions_mean_score=score(y,common),
                    position_averaged_same_condition_score=score(y,p.mean(axis=0)),
                    position_averaged_other_conditions_mean_score=score(y,common.mean(axis=0))))
            for a in arms:
                for b in arms:
                    xa=cache[(pid,a)]['d'][1:];xb=cache[(pid,b)]['d'][1:]
                    cosine_rows.append(dict(pair=pair,prompt=pid,condition_a=a,condition_b=b,
                        mean_cosine=float(np.mean(np.einsum('ij,ij->i',unit(xa),unit(xb))))))
            dp=cache[(pid,'plus_v')]['d'];dm=cache[(pid,'minus_v')]['d']
            cut=1+(len(dp)-1+1)//2
            for name,sl in [('prefill',slice(0,1)),('first_decode_half',slice(1,cut)),('second_decode_half',slice(cut,None))]:
                pp=dp[sl];mm=dm[sl];odd=(pp-mm)/2;even=(pp+mm)/2
                symmetry.append(dict(pair=pair,prompt=pid,window=name,
                    even_energy_fraction=float(np.square(even).sum()/(np.square(odd).sum()+np.square(even).sum())),
                    mirror_cosine=float(np.mean(np.einsum('ij,ij->i',unit(pp),unit(-mm))))))
    # Independent explicit coordinate sums for all six primary symmetry values.
    independent=[]
    for prompt in panel['prompts']:
        pid=prompt['id'];pp=cache[(pid,'plus_v')]['d'][1:];mm=cache[(pid,'minus_v')]['d'][1:]
        even_sum=0.;both_sum=0.;mirror=0.
        for p,m in zip(pp,mm):
            pp2=sum(float(v)*float(v) for v in p);mm2=sum(float(v)*float(v) for v in m)
            cross=sum(float(v)*float(w) for v,w in zip(p,m))
            even_sum+=(pp2+mm2+2*cross)/4;both_sum+=(pp2+mm2)/2
            mirror+=-cross/(pp2*mm2)**0.5
        independent.append(dict(prompt=pid,even_energy_fraction=even_sum/both_sum,mirror_cosine=mirror/len(pp)))
    first=old['prompts'][4]['id'];b,_=load(original,'native',first,'base',manifests[str(original)],used)
    c,_=load(original,'native',first,'existing_vector',manifests[str(original)],used)
    d=c-b;y=d[ANCHOR,1:]/np.linalg.norm(d[START,1:],axis=-1)[:,None]
    independent_template=1-sum(float(v-p)**2 for row in y for v,p in zip(row,template[0]))/sum(float(v)**2 for row in y for v in row)
    primary=json.loads((analysis/'RESULTS.json').read_text())
    max_error=max(abs(v[k]-next(x[k if k=='even_energy_fraction' else 'plus_minus_mirror_cosine'] for x in primary['sign_symmetry'] if x['prompt']==v['prompt'] and x['phase']=='decode' and x['block_zero_based']==ANCHOR)) for v in independent for k in ['even_energy_fraction','mirror_cosine'])
    reference=next(x['template_score_vs_zero'] for x in primary['anchor_rows'] if x['dataset']=='original' and x['cohort']=='native' and x['prompt']==first and x['condition']=='existing_vector' and x['phase']=='decode')
    assert max_error<1e-12 and abs(independent_template-reference)<1e-12
    result=dict(jackknife=jackknife,windows=windows,matched_leave_pair_out=crossval,
        full_response_cosines=cosine_rows,window_sign_symmetry=symmetry,
        independent_sign_symmetry=independent,independent_symmetry_max_error=max_error,
        independent_template_score_error=abs(independent_template-reference),input_hashes_verified=len(used))
    for name,data in [('TRAIN_JACKKNIFE.csv',jackknife),('DECODE_WINDOWS.csv',windows),
        ('MATCHED_LEAVE_PAIR_OUT.csv',crossval),('FULL_RESPONSE_COSINES.csv',cosine_rows),('WINDOW_SIGN_SYMMETRY.csv',symmetry)]:write_csv(output/name,data)
    (output/'ROBUSTNESS.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('original',type=Path);p.add_argument('matched',type=Path);p.add_argument('analysis',type=Path)
    a=p.parse_args();run(a.original,a.matched,a.analysis)
