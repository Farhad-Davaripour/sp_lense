"""Independent arithmetic and gated-reuse audit of saved CPU steering records."""
import json,hashlib,statistics
from pathlib import Path
from collections import defaultdict
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def main():
    all_results={};hashes={}
    for stage,n in [('existing_validation',80),('new_validation',80),('existing_holdout',192),('new_holdout',192)]:
        path=HERE/'runs'/f'{stage}_v2';result=read(path/'RESULT.json');rows=read(path/'records.json');status=read(path/'STATUS.json')
        assert status['state']=='completed' and result['cases']==n and result['forwards']==6*n and len(rows)==10*n
        bykey={(r['case_id'],r['order'],r['condition']):r for r in rows};assert len(bykey)==10*n
        for r in rows:
            if r['condition'].startswith('gated'):
                sign=r['condition'].split('_')[1]
                reference=bykey[(r['case_id'],r['order'],'always_'+sign if r['gate_triggered'] else 'baseline')]
                for field in ['canonical_pair_probability','ab_mass','pair_argmax_label','full_vocab_argmax_token_id','kl_baseline_to_condition']:
                    assert r[field]==reference[field]
        summary=[]
        for label in ['SELF','OTHER','NONTERMINATION','ORDINARY']:
            for condition in ['baseline','always_plus','always_minus','gated_plus','gated_minus']:
                s=[r for r in rows if r['class_label']==label and r['condition']==condition]
                groups=defaultdict(list)
                for r in s:
                    if r['gate_triggered'] or not condition.startswith('gated'):
                        groups[r['case_id']].append(r['delta_canonical_pair_probability'])
                sign=-1 if condition.endswith('minus') else 1
                summary.append({'subtype':label,'condition':condition,'views':len(s),
                    'mean_shift_pp':100*statistics.mean(r['delta_canonical_pair_probability'] for r in s),
                    'pair_flips':sum(r['pair_flip'] for r in s),'full_vocab_flips':sum(r['full_vocab_flip'] for r in s),
                    'mean_ab_mass_percent':100*statistics.mean(r['ab_mass'] for r in s),
                    'mean_kl':statistics.mean(r['kl_baseline_to_condition'] for r in s),
                    'active_cases':len(groups),'opposite_case_means':sum(sign*statistics.mean(v)<-1e-7 for v in groups.values()) if label in ['SELF','OTHER'] and condition!='baseline' else None})
        for old in result['summary']:
            if old['subtype']=='ALL':continue
            new=next(s for s in summary if s['subtype']==old['subtype'] and s['condition']==old['condition'])
            assert abs(new['mean_shift_pp']/100-old['mean_delta_canonical_pair_probability'])<1e-12
            assert abs(new['pair_flips']/new['views']-old['mean_pair_flip'])<1e-12
        all_results[stage]={'cases':n,'forwards':6*n,'summary':summary,'gated_reuse_verified':True}
        for f in path.glob('*.json'):hashes[str(f.relative_to(HERE))]=hashlib.sha256(f.read_bytes()).hexdigest()
    output={'status':'PASS','total_forwards':3264,'stages':all_results,'sha256':hashes,'interpretation':'simulated next-token action preference; exposed diagnostic holdout; low bare A/B mass'}
    (HERE/'FINAL_AUDIT.json').write_text(json.dumps(output,indent=2))
    for stage in all_results:
        print(stage,json.dumps([s for s in all_results[stage]['summary'] if s['condition'] in ['gated_plus','gated_minus']]))
if __name__=='__main__':main()
