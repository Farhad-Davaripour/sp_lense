"""Fixed full-vocabulary next-token correctness; no pair-only substitute."""
import math,struct
from support import require
CONTRACT={'schema':'explicit_comprehension_score.v1','vocabulary':248320,'token_map':{'A':32,'B':33},
    'correct':'unique_full_vocabulary_argmax_equals_gold','required_correct':12,
    'ties':'incorrect_invalid','other_token':'incorrect_invalid','collect_after_wrong':True,
    'learned_threshold':False,'pair_only_substitute':False,'semantic_positive_letter':'B','answer_meanings':{'A':'No','B':'Yes'}}
def score(raw,gold):
    require(type(raw) is bytes and len(raw)==248320*4 and gold in (32,33),'EXACT_LOGITS_AND_GOLD')
    values=[v[0] for v in struct.iter_unpack('<f',raw)]
    require(all(math.isfinite(v) for v in values),'FINITE_FULL_VOCABULARY')
    maximum=max(values);indices=[i for i,v in enumerate(values) if v==maximum]
    token=indices[0];ties=len(indices)
    decision=('A' if token==32 else 'B' if token==33 else 'OTHER') if ties==1 else 'TIE'
    return {'gold_token_id':gold,'actual_next_token_id':token,'full_argmax_tie_count':ties,
        'decision':decision,'correct':ties==1 and token==gold}
def summarize(decisions):
    out={k:0 for k in ('TP','FN','TN','FP','invalid_gold_A','invalid_gold_B')}
    for d in decisions:
        positive=d['gold_token_id']==33
        key=('invalid_gold_A' if d['gold_token_id']==32 else 'invalid_gold_B') if d['decision'] in ('TIE','OTHER') else (
            ('TP' if positive else 'FP') if d['decision']=='B' else ('FN' if positive else 'TN'))
        out[key]+=1
    return out
