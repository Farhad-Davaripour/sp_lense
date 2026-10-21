"""Descriptive analysis of audited saved scores. No fitting or model execution."""
import numpy as np


def summarize(rows,scored):
    if [r['state_id'] for r in rows]!=[s['state_id'] for s in scored]:raise ValueError('Score membership/order differs')
    p=np.array([s['probability'] for s in scored],dtype=float)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid saved score')
    result={}
    for family in sorted(set(r['family'] for r in rows)):
        members=[i for i,r in enumerate(rows) if r['family']==family];positive=[i for i in members if rows[i]['label']==1];negative=[i for i in members if rows[i]['label']==0]
        a=p[positive];b=p[negative]
        auc=float(((a[:,None]>b[None,:]).astype(float)+.5*(a[:,None]==b[None,:])).mean())
        lookup={};pairs=[]
        for i in members:
            row=rows[i]
            if row['condition'] not in ['own','peer']:continue
            context=row['tick'] if family=='same_format' else row['inactive_state']
            key=(row['seed'],context);lookup.setdefault(key,{})[row['condition']]=i
        for (seed,context),pair in sorted(lookup.items()):
            if set(pair)!= {'own','peer'}:raise ValueError('Incomplete role pair')
            own,peer=pair['own'],pair['peer']
            if rows[own]['label']!=1 or rows[peer]['label']!=0:raise ValueError('Pair labels differ')
            pairs.append({'seed':seed,'context':context,'own_state_id':rows[own]['state_id'],'peer_state_id':rows[peer]['state_id'],
                          'own_probability':float(p[own]),'peer_probability':float(p[peer]),'own_minus_peer':float(p[own]-p[peer])})
        deltas=np.array([r['own_minus_peer'] for r in pairs])
        result[family]={'rows':len(members),'ROC_AUC_descriptive':auc,'matched_pairs':len(pairs),'pairs_correct_order':int((deltas>0).sum()),'pairs_reversed':int((deltas<0).sum()),'pairs_tied':int((deltas==0).sum()),
                        'positive_score_min':float(a.min()),'positive_score_median':float(np.median(a)),'positive_score_max':float(a.max()),
                        'negative_score_min':float(b.min()),'negative_score_median':float(np.median(b)),'negative_score_max':float(b.max()),'pairs':pairs}
    return result
