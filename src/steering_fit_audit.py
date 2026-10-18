"""Recompute saved correction errors on CPU; never fit or load a language model."""
from pathlib import Path
import argparse
import collections
import json
import math
from src.utils import sha256,write_json


def audit(root, output):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    root=Path(root);output=Path(output)
    if output.exists():raise FileExistsError('Use a new audit output')
    torch.set_num_threads(4)
    packet=root/'packet';read=lambda p:json.loads(p.read_bytes())
    contexts=read(packet/'TRAIN_CONTEXTS_PRIVATE.json');panel=read(packet/'FOCUSED_CONTEXTS_PRIVATE.json')
    index=read(packet/'POSITION_INDEX_PRIVATE.json');weights=np.load(packet/'CASE_NORMALIZED_DECISION_WEIGHTS.npy')
    if len(index)!=4539 or len(contexts)!=136 or len(weights)!=len(index):raise ValueError('Capture membership differs')
    anchors=[(c,a) for c in panel for a in c['anchors']]
    alignment=[]
    for c,a in anchors:
        source=contexts[c['context_index']];j=a['prediction_index'];r=index[a['row']]
        aligned=(c['case_id']==source['case_id']==r['case_id'] and r['context_index']==c['context_index']
                 and r['prediction_index']==j and c['prefix_ids']==source['prefix_ids']
                 and c['source_emitted_ids'][:j]==source['source_emitted_ids'][:j])
        if not aligned:raise ValueError('Fixed anchor inputs are misaligned')
        alignment.append({'row':a['row'],'case_id':c['case_id'],'context_list_index':c['context_index'],
            'parent_context_metadata_index':source['context_index'],'actual_causal_tokens_exact':True})
    original=root/'native_recovery3_20261015T155553Z/native/run'
    p48=root/'native_final48_shared6/run';p96=root/'native_final96_cold_post49_20261016T034017Z/postrefresh'
    states={'initializer':load_file(str(original/'RIDGE_INITIALIZER.safetensors'))}
    checkpoint_pins={}
    for label,folder,update in [('update48',p48,48),('update96',p96,96)]:
        path=folder/f'UPDATE_{update:03}.pt';state=torch.load(path,map_location='cpu',weights_only=True)
        if state['completed_updates']!=update or state['rank_each']!=32 or tuple(state['sites'])!=(31,47,62):
            raise ValueError('Wrong checkpoint identity')
        if any(float(v['step'])!=update for v in state['optimizer']['state'].values()):
            raise ValueError('Adam step counters differ from checkpoint dose')
        if any(not torch.isfinite(v).all() for v in state['maps'].values()):raise ValueError('Nonfinite correction weights')
        states[label]=state['maps'];checkpoint_pins[label]={'sha256':sha256(path),'completed_updates':update,
            'Adam_parameter_states':len(state['optimizer']['state']),'next_schedule_index':state.get('next_schedule_index')}
    active=np.array([r['gate_on'] for r in index]);w=weights.astype(np.float64);w/=w.mean()
    scales_recorded=read(p48/'REUSED_CAPTURE_AND_RESUME.json')['scales']
    scales_after=read(p96/'REUSED_CAPTURE_AND_RESUME.json')['scales']
    if scales_recorded!=scales_after:raise ValueError('The supposedly fixed normalizers changed')
    keys=[f'{site}/{kind}' for site in (31,47,62) for kind in ('attention','output')]
    details={};fixed={label:np.zeros(len(anchors)) for label in states}
    overall={label:np.zeros(len(index)) for label in [*states,'zero']}
    for key in keys:
        x_path=original/'bootstrap'/key/'x.npy';y_path=original/'bootstrap'/key/'target.npy'
        x=np.load(x_path,mmap_mode='r');y=np.load(y_path,mmap_mode='r')
        if x.shape!=y.shape or x.shape!=(4539,5120):raise ValueError('Cached feature/target shape differs')
        energy=0.;zero_off=True
        for i in range(0,len(index),128):
            z=np.asarray(y[i:i+128],dtype=np.float64);energy+=float((z*z*w[i:i+128,None]).sum())
            zero_off=zero_off and not np.any(z[~active[i:i+128]])
        scale=max(energy/(w[active].sum()*5120),1e-6)
        if not math.isclose(scale,scales_recorded[key],rel_tol=1e-8,abs_tol=1e-12):
            raise ValueError('Cached target-energy normalizer cannot be reproduced')
        errors={label:[] for label in [*states,'zero']}
        for i in range(0,len(index),128):
            inputs=torch.from_numpy(np.array(x[i:i+128],copy=True));targets=torch.from_numpy(np.array(y[i:i+128],copy=True))
            for label,maps in states.items():
                prediction=(inputs@maps[key+'.down'])@maps[key+'.up'].T
                error=((prediction-targets).square().mean(-1)/scale).numpy().astype(np.float64)
                errors[label].extend(error.tolist());overall[label][i:i+len(error)]+=error/len(keys)
            zero=(targets.square().mean(-1)/scale).numpy().astype(np.float64)
            errors['zero'].extend(zero.tolist());overall['zero'][i:i+len(zero)]+=zero/len(keys)
        for label in states:
            # Match the original one-row float32 evaluation, not a batch matmul.
            for k,(c,a) in enumerate(anchors):
                inp=torch.from_numpy(np.array(x[a['row']],copy=True));target=torch.from_numpy(np.array(y[a['row']],copy=True))
                pred=(inp@states[label][key+'.down'])@states[label][key+'.up'].T
                fixed[label][k]+=float((pred-target).square().mean()/scale)/len(keys)
        details[key]={'scale':scale,'scale_matches_saved':True,'gate_OFF_targets_zero':bool(zero_off),
            'feature_sha256':sha256(x_path),'target_sha256':sha256(y_path),
            'weighted_active_normalized_MSE':{label:float(np.average(v,weights=w*active)) for label,v in errors.items()},
            'weighted_control_normalized_MSE':{label:float(np.average(v,weights=w*~active)) for label,v in errors.items()}}
    own={c['case_id'] for c in panel if c['gate_on']};fixed_summary={};per_case={}
    for label,errors in fixed.items():
        groups=collections.defaultdict(list)
        for (c,a),e in zip(anchors,errors):groups[c['case_id']].append(float(e))
        per_case[label]={cid:float(np.mean(v)) for cid,v in groups.items()}
        value=float(np.mean([per_case[label][cid] for cid in own]));fixed_summary[label]=value
        if label in ['update48','update96']:
            recorded=read((p48 if label=='update48' else p96)/('UPDATE_048_FIXED_TRAIN_DIAGNOSTIC.json' if label=='update48' else 'UPDATE_096_FIXED_TRAIN_DIAGNOSTIC.json'))
            expected=recorded['own_case_macro_fixed_bootstrap_normalized_MSE']
            if not math.isclose(value,expected,rel_tol=1e-4,abs_tol=1e-6):raise ValueError('Saved fixed-anchor MSE cannot be reproduced')
    corpus={}
    for label,errors in overall.items():
        groups=collections.defaultdict(list)
        for row,e in zip(index,errors):groups[row['case_id']].append(float(e))
        corpus[label]={'weighted_active_MSE':float(np.average(errors,weights=w*active)),
            'weighted_control_MSE':float(np.average(errors,weights=w*~active)),
            'case_macro_MSE':{cid:float(np.mean(v)) for cid,v in groups.items()}}
    result={'format':'research3-steering-fit-cpu-audit-v1','status':'SAVED_ERROR_ARITHMETIC_REPRODUCED',
        'language_model_loads':0,'optimizer_updates':0,'new_model_forwards':0,'training_rows':len(index),
        'fixed_anchors':len(anchors),'fixed_cases':len({c['case_id'] for c in panel}),
        'fixed_own_MSE':fixed_summary,'fixed_case_MSE':per_case,'all_bootstrap_MSE':corpus,
        'normalizers_and_sources':details,'anchor_alignment':alignment,'checkpoints':checkpoint_pins,
        'fixed_prefix_alignment_passed':True,'fixed_normalizers_identical':True,
        'CPU_recompute_tolerance':{'relative':1e-4,'absolute':1e-6},
        'scope':'CPU arithmetic audit only; not a CPU/GPU training-equivalence qualification or fresh model evaluation.',
        'objective_limit':'The training targets are current local teacher references; fixed-bootstrap error and full-model output KL are separate diagnostics. Post-update48 training also includes refreshed TRAIN trajectories.'}
    write_json(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=audit(a.root,a.output);print(json.dumps({k:r[k] for k in ['status','fixed_own_MSE','fixed_case_MSE','fixed_prefix_alignment_passed']}))
