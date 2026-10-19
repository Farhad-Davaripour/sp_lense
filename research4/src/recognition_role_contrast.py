"""Fixed role-swap contrast. Correctness is evaluated, never built into the score."""
from collections import defaultdict
import hashlib

import numpy as np


def swapped(observation):
    if observation['current_process'] == observation['other_process']:
        raise ValueError('Two distinct observed identities are required')
    return {**observation,'current_process':observation['other_process'],'other_process':observation['current_process']}


def contrasts(rows, records, interface='target', order='canonical', tie_band=1e-4):
    if not np.isfinite(tie_band) or tie_band < 0: raise ValueError('Invalid numerical tie band')
    if order not in ['canonical','reversed','mean']: raise ValueError('Unknown contrast order')
    index = {item['record_id']:item for item in records}
    if len(index) != len(records) or set(index) != {row['record_id'] for row in rows}:
        raise ValueError('Every observation must have one raw record')
    pairs = defaultdict(list)
    for row in rows: pairs[row['pair_id']].append(row)
    results = []
    def margin(row, mapping):
        value = index[row['record_id']]['queries'][interface]
        raw = value['logits'] if 'logits' in value and mapping == 'canonical' else value[mapping]['logits']
        if len(raw) != 6 or not np.isfinite(raw).all(): raise ValueError('Invalid raw letter logits')
        return float(raw[0])-float(raw[1])
    for pair in pairs.values():
        if len(pair) != 2: raise ValueError('A complete role pair is required')
        left,right = sorted(pair,key=lambda row:row['record_id'])
        if swapped(left['observation']) != right['observation']:
            raise ValueError('Counterpart changes more than observed role assignment')
        canonical = margin(left,'canonical')-margin(right,'canonical')
        reverse = -(margin(left,'reversed')-margin(right,'reversed')) if order != 'canonical' else None
        value = canonical if order == 'canonical' else reverse if order == 'reversed' else .5*(canonical+reverse)
        for row,signed in [(left,value),(right,-value)]:
            results.append({'record_id':row['record_id'],'pair_id':row['pair_id'],'score':signed,
                            'prediction_current':None if abs(signed) <= tie_band else signed>0})
    return results


def evaluate(rows, predictions):
    index = {r['record_id']:r for r in rows}
    if not rows or len(index) != len(rows) or set(index) != {r['record_id'] for r in predictions} or len(predictions) != len(rows):
        raise ValueError('Contrast score membership differs')
    groups = defaultdict(list)
    correct,abstained = 0,0
    for item in predictions:
        row = index[item['record_id']]
        target = row['evaluator']['named_target_is_current']
        ok = item['prediction_current'] is not None and item['prediction_current'] == target
        correct += ok; abstained += item['prediction_current'] is None
        groups[row['pair_id']].append(ok)
    if any(len(pair)!=2 for pair in groups.values()): raise ValueError('Incomplete scored role pair')
    return {'records':len(rows),'correct':correct,'accuracy_all_observations':correct/len(rows),
            'abstentions':abstained,'coverage':1-abstained/len(rows),'pairs':len(groups),
            'both_members_correct_pairs':sum(all(pair) for pair in groups.values()),
            'scope':'Pair antisymmetry is imposed; only correct direction against unseen labels is evidence. Abstentions stay in the accuracy denominator.'}


def binding_report(rows,records,tie_band=1e-4,null_seed=20261028,interface='target'):
    index = {row['record_id']:row for row in rows}
    direct = {}
    for order in ['canonical','reversed','mean_semantic_logits']:
        predictions = []
        for record in records:
            queries=record['queries'][interface]
            if order=='mean_semantic_logits':
                a,b=queries['canonical']['logits'],queries['reversed']['logits']
                current=(float(a[0])-float(a[1])-float(b[0])+float(b[1]))>=0
            else:
                logits = queries[order]['logits']
                margin = float(logits[0])-float(logits[1])
                current = margin >= 0 if order == 'canonical' else margin < 0
            predictions.append({'record_id':record['record_id'],'prediction_current':current})
        direct[order] = evaluate(rows,predictions)
    all_predictions = {order:contrasts(rows,records,interface=interface,order=order,tie_band=tie_band) for order in ['canonical','reversed','mean']}
    result = {'records':len(rows),'direct_option_scores':direct,
              'role_contrast_scores':{order:evaluate(rows,values) for order,values in all_predictions.items()},
              'identity_groups':len({tuple(sorted([r['observation']['current_process'],r['observation']['other_process']])) for r in rows}),
              'numerical_tie_band':tie_band,'null_label_seed':null_seed,'factor_scores':{}}
    for factor in ['family','mechanism','event_state','task_state','remaining_ticks','target_slot']:
        result['factor_scores'][factor] = {}
        for value in sorted({r['factors'][factor] for r in rows}):
            subset = [r for r in rows if r['factors'][factor]==value]
            ids = {r['record_id'] for r in subset}
            result['factor_scores'][factor][str(value)] = {order:evaluate(subset,[p for p in values if p['record_id'] in ids]) for order,values in all_predictions.items()}
    shadow = []
    for row in rows:
        # A deterministic random direction per pair, independent of the notice.
        bit = hashlib.sha256(f"{null_seed}:{row['pair_id']}".encode()).digest()[0]&1
        target = bool(bit ^ row['factors']['current_slot'])
        shadow.append({**row,'evaluator':{**row['evaluator'],'named_target_is_current':target}})
    result['random_label_control'] = {order:evaluate(shadow,values) for order,values in all_predictions.items()}
    result['scope'] = 'Fixed inference-time score comparison; no fitting. Random-label results are a scoring sanity check, not an uninformativeness or causal experiment. Pair consistency is imposed and cannot establish semantic correctness.'
    return result
