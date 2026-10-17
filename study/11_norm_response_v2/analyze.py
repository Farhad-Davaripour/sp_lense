"""Independent prospective support rules and saved-array endpoint checks."""
from pathlib import Path
import csv,hashlib,json,math
import numpy as np

BASE=Path('C:/Users/farha/Research4/artifacts/norm_response_v2')
ROOT=BASE/'prospective_20261012';OUT=BASE/'evaluated_20261012'
OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def js(name):return json.loads((ROOT/name).read_text())
def span(values):
    x=[float(v) for v in values if v is not None]
    return {'n':len(x),'min':min(x),'median':float(np.median(x)),'max':max(x)} if x else {'n':0,'min':None,'median':None,'max':None}
def cosine(a,b):
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip(np.sum(a*b)/den,-1,1)) if den>1e-20 else None
def angle(a,b):
    c=cosine(a,b);return math.degrees(math.acos(c)) if c is not None else None
def csvwrite(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def KL(base,changed):
    b=np.asarray(base,dtype=float);c=np.asarray(changed,dtype=float)
    b=b-b.max();c=c-c.max();lb=b-np.log(np.exp(b).sum());lc=c-np.log(np.exp(c).sum())
    return float(np.sum(np.exp(lb)*(lb-lc)))
status=js('STATUS.json');assert status['status']=='complete'
manifest=js('EXPORT_HASHES.json')
for member,digest in manifest.items():assert sha(ROOT/member)==digest,member
for name,digest in js('PREDICTION_FREEZE.json').items():assert sha(ROOT/'predictions'/name)==digest
selection=js('CALIBRATION_SELECTION_LOCK.json');selected=selection['selected']
metrics=js('INTERNAL_METRICS.json')
held=[r for r in metrics if r['split']=='held_out' and r['direction'] in [2,3] and r['precision']=='FP32']
names=['linear','quadratic','rational','exact_norm_gate2'];strong=[.5,1.25,2.];small=[.0078125,.03125]
question_rules=[];ranges=[];flat=[]
for question in sorted(set(r['question'] for r in held)):
    for kind in ['branch','complete']:
        a=[r for r in held if r['question']==question and r['map']==kind]
        high=[r for r in a if r['strength'] in strong];low=[r for r in a if r['strength'] in small];two=[r for r in a if r['strength']==2.]
        med=lambda rows,name:float(np.median([r['errors'][name]['relative'] for r in rows if r['resolved']]))
        high_old=med(high,'quadratic');high_new=med(high,selected);small_old=med(low,'quadratic');small_new=med(low,selected);two_new=med(two,selected)
        passes={'all_resolved':all(r['resolved'] for r in high+low),'strong_median_improvement':high_new<=.75*high_old,
            'strength2_median_limit':two_new<=.3,'strength2_every_case_limit':all(r['errors'][selected]['relative']<=.5 for r in two if r['resolved']),
            'small_every_case_limit':all(r['errors'][selected]['relative']<=.05 for r in low if r['resolved']),
            'small_median_retention':small_new<=small_old+.005}
        question_rules.append({'question':question,'map':kind,'strong_quadratic_median':high_old,'strong_selected_median':high_new,
            'strong_median_improvement_fraction':1-high_new/high_old,'strength2_selected_median':two_new,
            'small_quadratic_median':small_old,'small_selected_median':small_new,'checks':passes,'pass':all(passes.values())})
for precision in ['FP32','BF16']:
    for kind in ['branch','complete']:
        for eps in [.0078125,.03125,.125,.5,1.25,2.]:
            a=[r for r in metrics if r['split']=='held_out' and r['direction']>=2 and r['precision']==precision and r['map']==kind and r['strength']==eps]
            for name in names:
                ranges.append({'precision':precision,'map':kind,'strength':eps,'equation':name,'cases':len(a),'independent_questions':len(set(r['question'] for r in a)),
                    'resolved':sum(r['resolved'] for r in a),'relative_vector_error':span(r['errors'][name]['relative'] for r in a if r['resolved']),
                    'absolute_vector_error':span(r['errors'][name]['absolute'] for r in a),
                    'magnitude_relative_error':span(r['errors'][name]['magnitude_relative_error'] for r in a if r['resolved']),
                    'direction_angle_error_degrees':span(math.degrees(math.acos(r['errors'][name]['direction_cosine'])) if r['errors'][name]['direction_cosine'] is not None else None for r in a if r['resolved']),
                    'actual_input_relative_error':span(r['actual_input_errors'][name]['relative'] for r in a if r['resolved']),
                    'zero_delivered':sum(r['zero_input'] for r in a),'zero_output':sum(r['zero_output'] for r in a)})
for r in metrics:
    for name in names:
        e=r['errors'][name]
        flat.append({k:r[k] for k in ['question','split','family','position','direction','direction_split','precision','map','strength','sign','actual_input_norm','input_projection','off_direction_norm','zero_input','zero_output','observed_magnitude','observed_gain','resolved']}|
            {'equation':name,**e,'actual_input_relative_error':r['actual_input_errors'][name]['relative']})
csvwrite(OUT/'ALL_VECTOR_ERRORS.csv',flat);dump(OUT/'VECTOR_ERROR_RANGES.json',ranges);dump(OUT/'QUESTION_SUPPORT.json',question_rules)

angle_rows=[];verification_error=0.
for pid in sorted(set(r['question'] for r in held)):
    for position in ['prefill','decode1']:
        for di in [2,3]:
            p=ROOT/'predictions'/f'{pid}_{position}_d{di}_FP32.npz'
            with np.load(p,allow_pickle=False) as z:pred={k:z[k].copy() for k in ['nominal_branch','nominal_complete']}
            with np.load(ROOT/'responses'/p.name,allow_pickle=False) as z:actual={'branch':z['branch_delta'].copy(),'complete':z['complete_delta'].copy()}
            for kind in ['branch','complete']:
                for i,eps in enumerate([.0078125,.03125,.125,.5,1.25,2.]):
                    dp=actual[kind][2*i].astype(float);dm=actual[kind][2*i+1].astype(float);measured=angle(dp,dm)
                    for ci,name in enumerate(names):
                        pp=pred['nominal_'+kind][2*i,ci].astype(float);pm=pred['nominal_'+kind][2*i+1,ci].astype(float)
                        pa=angle(pp,pm)
                        angle_rows.append({'question':pid,'position':position,'direction':di,'map':kind,'strength':eps,'equation':name,
                            'measured_response_angle':measured,'predicted_response_angle':pa,'absolute_angle_error':abs(pa-measured) if pa is not None and measured is not None else None})
                        for sign,dy,py in [(1,dp,pp),(-1,dm,pm)]:
                            expected=next(r for r in held if r['question']==pid and r['position']==position and r['direction']==di and r['map']==kind and r['strength']==eps and r['sign']==sign)['errors'][name]['relative']
                            observed=float(np.linalg.norm(dy-py)/np.linalg.norm(dy));verification_error=max(verification_error,abs(observed-expected))
assert verification_error<1e-10
csvwrite(OUT/'SIGNED_RESPONSE_ANGLES.csv',angle_rows)

output=js('OUTPUT_METRICS.json') if (ROOT/'OUTPUT_METRICS.json').is_file() else []
output_ranges=[];output_rules=[];output_verification=0.
for position in ['prefill','decode1']:
    for eps in [.125,.5,2.]:
        a=[r for r in output if r['split']=='held_out' and r['position']==position and r['strength']==eps]
        if not a:continue
        for name in ['Fisher','linear_endpoint','quadratic_endpoint','rational_endpoint','selected_endpoint']:
            output_ranges.append({'position':position,'strength':eps,'predictor':name,'independent_questions':len(set(r['question'] for r in a)),'cases':len(a),
                'resolved':sum(r['resolved'] for r in a),'relative_error':span(r['relative_errors'][name] for r in a if r['resolved']),
                'absolute_error_nats':span(r['absolute_errors'][name] for r in a),'actual_KL_nats':span(r['measured_endpoint_KL'] for r in a),
                'KL_actual_to_predicted':span(r['KL_actual_to_predicted'].get(name) for r in a)})
for pid in sorted(set(r['question'] for r in output if r['split']=='held_out')):
    all_cases=[r for r in output if r['question']==pid and r['strength']==.5];a=[r for r in all_cases if r['resolved']]
    med=lambda name:float(np.median([r['relative_errors'][name] for r in a]))
    predicted=med('selected_endpoint');old_fisher=med('Fisher');old_quadratic=med('quadratic_endpoint')
    checks={'all_cases_resolved':len(a)==len(all_cases) and len(a)>0,'improve_over_Fisher':predicted<=.75*old_fisher,'improve_over_quadratic_endpoint':predicted<=.75*old_quadratic,
        'every_case_below25pct':all(r['relative_errors']['selected_endpoint']<=.25 for r in a)}
    output_rules.append({'question':pid,'cases':len(a),'selected_median':predicted,'Fisher_median':old_fisher,'quadratic_endpoint_median':old_quadratic,'checks':checks,'pass':all(checks.values())})
# Recompute endpoint costs independently from full saved logits, one case at a time.
case_keys=sorted(set((r['question'],r['position'],r['direction']) for r in output))
for pid,position,di in case_keys:
    path=ROOT/'output'/f'{pid}_{position}_d{di}_predictions.npz';meta=js('output/'+path.with_suffix('.json').name)
    assert sha(path)==meta['sha256']
    with np.load(path,allow_pickle=False) as z:pred={k:z[k].copy() for k in z.files}
    with np.load(ROOT/'output'/f'{pid}_{position}_d{di}_actual.npz',allow_pickle=False) as z:actual=z['logits'].copy()
    for i,alpha in enumerate(pred['alphas']):
        row=next(r for r in output if r['question']==pid and r['position']==position and r['direction']==di and r['strength']==abs(alpha) and r['sign']==(1 if alpha>0 else -1))
        measured=KL(pred['base_logits'],actual[i]);output_verification=max(output_verification,abs(measured-row['measured_endpoint_KL']))
        for name in ['quadratic','rational',selected]:
            logits=pred['base_logits']+pred['q_'+name][i]
            value=KL(pred['base_logits'],logits);output_verification=max(output_verification,abs(value-row['predictions'][name+'_endpoint']))
assert output_verification<1e-10
dump(OUT/'OUTPUT_ERROR_RANGES.json',output_ranges);dump(OUT/'OUTPUT_QUESTION_SUPPORT.json',output_rules)
results={'date':'2026-10-12','status':'prospective equation-improvement study complete','selected_equation':selected,
    'independent_questions':8,'calibration_questions':4,'held_out_questions':4,'fresh_directions':4,'held_out_directions':2,
    'calibration_lock':selection,'internal_primary_support':all(r['pass'] for r in question_rules),'question_rules':question_rules,
    'vector_error_ranges':ranges,'output_primary_support':bool(output_rules) and all(r['pass'] for r in output_rules),'output_question_rules':output_rules,
    'output_error_ranges':output_ranges,'internal_numerical_gate_pass':js('NUMERICAL_GATE.json')['pass'],
    'output_derivative_gates':js('OUTPUT_DERIVATIVE_GATE.json') if (ROOT/'OUTPUT_DERIVATIVE_GATE.json').is_file() else None,
    'checks':{'native_manifest_files_verified':len(manifest),'prediction_files_verified':len(js('PREDICTION_FREEZE.json')),
        'independent_vector_relative_error_max_difference':verification_error,'independent_endpoint_KL_max_difference':output_verification},
    'whole_backbone_calls':status['whole_backbone_calls'],'signed_MLP_points':status['signed_MLP_points'],'output_direction_cases':status['output_cases'],
    'GPU_allocations':0,'native_export':js('EXPORT_RECEIPT.json'),'old_monitor_remains_stopped':True,
    'scope':'Finite local vector-response approximation on pinned block11 and conditional endpoint-cost experiment; not a classifier, full-branch exact identity, per-token bound or global physical energy'}
dump(OUT/'RESULTS.json',results)
print(json.dumps({'selected':selected,'internal_primary_support':results['internal_primary_support'],'output_primary_support':results['output_primary_support'],
    'output_rules':output_rules,'strength2_branch_ranges':[r for r in ranges if r['precision']=='FP32' and r['map']=='branch' and r['strength']==2.],
    'output_strength0_5_selected':[r for r in output_ranges if r['strength']==.5 and r['predictor']=='selected_endpoint']},indent=2))
