"""Position-balanced component calibration with no fitted coefficients."""
from collections import defaultdict
import numpy as np
from .recognition_questions import SEMANTICS,semantic_logits


def orders(config,interface):
    return config.get('orders_by_interface',{}).get(interface,config['orders'])


def metric(rows,interface,predicted):
    meanings=SEMANTICS[interface]
    truth=[meanings.index(row['factors']['event_state'] if interface=='status' else row['evaluator']['remaining_ticks_if_active']) for row in rows]
    matrix=np.zeros((len(meanings),len(meanings)),dtype=int)
    for target,value in zip(truth,predicted):matrix[target,value]+=1
    result={'records':len(rows),'correct':int(matrix.trace()),'accuracy':float(matrix.trace()/len(rows)),
            'confusion_true_rows_predicted_columns':matrix.tolist(),
            'per_true_class':{str(value):{'records':int(matrix[index].sum()),'correct':int(matrix[index,index])} for index,value in enumerate(meanings)}}
    if interface=='time':
        active=[index for index,row in enumerate(rows) if row['evaluator']['scheduled_event_active']]
        result['active_time']={'records':len(active),'correct':sum(predicted[index]==truth[index] for index in active)}
        result['inactive_time']={'records':len(rows)-len(active),'correct':sum(predicted[index]==truth[index] for index,row in enumerate(rows) if not row['evaluator']['scheduled_event_active'])}
    return result


def component_report(rows,records,config):
    if len(rows)!=len(records) or [r['record_id'] for r in rows]!=[r['record_id'] for r in records]:raise ValueError('Cyclic component alignment differs')
    result={'records':len(rows),'components':{},'scope':'Fixed cyclic means and role-symmetric means; no training. Disagreement energy is a constructed readout statistic, not a physical law or correctness guarantee.'}
    groups=defaultdict(list)
    for index,row in enumerate(rows):groups[row['pair_id']].append(index)
    if any(len(pair)!=2 for pair in groups.values()):raise ValueError('Complete role pairs required')
    for interface in config['interfaces']:
        mappings=orders(config,interface);classes=len(SEMANTICS[interface])
        if mappings!=['canonical']+[f'cyclic_{offset}' for offset in range(1,classes)]:raise ValueError('Every semantic class must visit every position once')
        vectors=np.asarray([[semantic_logits(record['queries'][interface][order]['logits'],interface,order) for order in mappings] for record in records])
        mean=vectors.mean(axis=1)
        centered=vectors-vectors.mean(axis=2,keepdims=True)
        energy=np.square(centered-centered.mean(axis=1,keepdims=True)).mean(axis=(1,2))
        metrics={order:metric(rows,interface,vectors[:,position].argmax(axis=1).tolist()) for position,order in enumerate(mappings)}
        metrics['cyclic_mean']=metric(rows,interface,mean.argmax(axis=1).tolist())
        role_mean=mean.copy()
        for pair in groups.values():role_mean[pair]=(mean[pair[0]]+mean[pair[1]])/2
        metrics['cyclic_mean_role_average']=metric(rows,interface,role_mean.argmax(axis=1).tolist())
        result['components'][interface]={'scores_by_view':metrics,'semantic_classes':SEMANTICS[interface],
                                        'mean_centered_disagreement_energy':float(energy.mean()),
                                        'definition':'Mean squared deviation of centered, semantically aligned logits from their cyclic mean. Centering removes per-prompt common logit offsets.'}
    return result
