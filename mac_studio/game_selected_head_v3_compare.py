"""Bounded CPU comparison of saved F32 gradients, independent of MLX."""
import hashlib
import json
import math
from pathlib import Path

CHUNK=1024**2
TOLERANCES={'loss_atol':0.0001,'loss_rtol':0.0001,'gradient_atol':0.000001,'gradient_rtol':0.001,
    'tensor_relative_l2':0.001,'tiny_reference_norm':0.00000001,'tiny_tensor_max_abs':0.00000001,'global_relative_l2':0.001}
LOGIT_TOLERANCES={'gradient_atol':0.001,'gradient_rtol':0.0001,'tensor_relative_l2':0.0001,
    'tiny_reference_norm':0.00000001,'tiny_tensor_max_abs':0.00000001,'global_relative_l2':0.0001}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(CHUNK),b''):digest.update(block)
    return digest.hexdigest()


def loss_equivalence(reference,selected,count_reference,count_selected,tolerances):
    if type(count_reference)is not int or count_reference<1 or count_selected!=count_reference:
        raise ValueError('Supervised target counts differ')
    if any(isinstance(v,bool)or not isinstance(v,(int,float))or not math.isfinite(v)for v in(reference,selected)):
        raise ValueError('Nonfinite target likelihood')
    difference=abs(reference-selected);limit=tolerances['loss_atol']+tolerances['loss_rtol']*abs(reference)
    return {'reference_mean_nll':reference,'selected_mean_nll':selected,'absolute_difference':difference,
        'allowed_difference':limit,'target_tokens':count_reference,'passed':difference<=limit}


def _validated_header(path,expected):
    from game_all_layers_layout import read_tensor_header
    header,offset,_=read_tensor_header(path)
    tensors={key:value for key,value in header.items()if key!='__metadata__'}
    if set(tensors)!=set(expected):raise ValueError('Gradient tensor inventory differs')
    intervals=[]
    for name,spec in tensors.items():
        if spec.get('dtype')!='F32'or tuple(spec.get('shape',[]))!=tuple(expected[name]):raise ValueError('Gradient shape/dtype differs')
        start,end=spec.get('data_offsets',[None,None]);size=math.prod(expected[name])*4
        if type(start)is not int or type(end)is not int or start<0 or end-start!=size:raise ValueError('Gradient payload interval differs')
        intervals.append((start,end))
    intervals.sort();cursor=0
    for start,end in intervals:
        if start!=cursor:raise ValueError('Gradient payload has overlap/gap')
        cursor=end
    if offset+cursor!=Path(path).stat().st_size:raise ValueError('Gradient payload file size differs')
    return tensors,offset


def parameter_state_sha256(path,expected):
    """Reproduce the live name/shape/value digest without loading a model."""
    tensors,offset=_validated_header(path,expected);result=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for name in sorted(expected):
            result.update(name.encode());result.update(str(tuple(expected[name])).encode())
            start,end=tensors[name]['data_offsets'];stream.seek(offset+start);remaining=end-start
            while remaining:
                block=stream.read(min(CHUNK,remaining))
                if not block:raise ValueError('Truncated parameter-state payload')
                result.update(block);remaining-=len(block)
    return result.hexdigest()


def compare_gradients(reference_path,selected_path,expected,tolerances,require_zero_a=False):
    import numpy as np
    reference,reference_offset=_validated_header(reference_path,expected)
    selected,selected_offset=_validated_header(selected_path,expected)
    leaves=[];reference_global=0.0;difference_global=0.0;all_pass=True
    with Path(reference_path).open('rb')as a,Path(selected_path).open('rb')as b:
        for name in sorted(expected):
            a.seek(reference_offset+reference[name]['data_offsets'][0]);b.seek(selected_offset+selected[name]['data_offsets'][0])
            remaining=math.prod(expected[name])*4;reference_norm=0.0;selected_norm=0.0;difference_norm=0.0
            maximum=0.0;reference_maximum=0.0;violations=0;nonzero_a=nonzero_b=0
            while remaining:
                size=min(CHUNK,remaining);raw_a=a.read(size);raw_b=b.read(size)
                if len(raw_a)!=size or len(raw_b)!=size:raise ValueError('Truncated gradient payload')
                x=np.frombuffer(raw_a,dtype='<f4').astype(np.float64);y=np.frombuffer(raw_b,dtype='<f4').astype(np.float64)
                if not np.isfinite(x).all()or not np.isfinite(y).all():raise ValueError('Nonfinite saved gradient')
                difference=y-x;absolute=np.abs(difference)
                reference_norm+=float(np.dot(x,x));selected_norm+=float(np.dot(y,y));difference_norm+=float(np.dot(difference,difference))
                maximum=max(maximum,float(absolute.max(initial=0)));reference_maximum=max(reference_maximum,float(np.abs(x).max(initial=0)))
                violations+=int(np.count_nonzero(absolute>tolerances['gradient_atol']+tolerances['gradient_rtol']*np.abs(x)))
                nonzero_a+=int(np.count_nonzero(x));nonzero_b+=int(np.count_nonzero(y));remaining-=size
            norm=math.sqrt(reference_norm);relative=math.sqrt(difference_norm/reference_norm)if reference_norm else None
            tiny=norm<tolerances['tiny_reference_norm']
            norm_pass=maximum<=tolerances['tiny_tensor_max_abs']if tiny else relative<=tolerances['tensor_relative_l2']
            zero_a_pass=not(require_zero_a and name.endswith('.lora_a'))or(nonzero_a==0 and nonzero_b==0)
            passed=violations==0 and norm_pass and zero_a_pass;all_pass=all_pass and passed
            leaves.append({'name':name,'shape':list(expected[name]),'reference_l2':norm,'selected_l2':math.sqrt(selected_norm),
                'difference_l2':math.sqrt(difference_norm),'relative_l2':relative,'maximum_absolute_difference':maximum,
                'reference_maximum_absolute':reference_maximum,'allclose_element_violations':violations,
                'tiny_reference':tiny,'exact_zero_a_required':require_zero_a and name.endswith('.lora_a'),
                'exact_zero_a_passed':zero_a_pass,'passed':passed})
            reference_global+=reference_norm;difference_global+=difference_norm
    if reference_global<=0:raise ValueError('All reference gradients are zero; cannot establish update equivalence')
    global_relative=math.sqrt(difference_global/reference_global)
    return {'schema':'research4.selected-head-gradient-equivalence.v3','passed':all_pass and global_relative<=tolerances['global_relative_l2'],
        'reference_sha256':sha(reference_path),'selected_sha256':sha(selected_path),'tensor_count':len(expected),'all_finite':True,
        'global_relative_l2':global_relative,'reference_global_l2':math.sqrt(reference_global),
        'failed_tensor_count':sum(not row['passed']for row in leaves),'tolerances':tolerances,'tensors':leaves,
        'scope':'CPU replay of saved gradients; does not run or certify a model by itself.'}


def compare_logits(reference_path,selected_path,shape):
    result=compare_gradients(reference_path,selected_path,{'target_logits':tuple(shape)},LOGIT_TOLERANCES)
    return {'schema':'research4.selected-head-target-logit-equivalence.v3','passed':result['passed'],
        'reference_sha256':result['reference_sha256'],'selected_sha256':result['selected_sha256'],
        'shape':list(shape),'valid_target_rows_only':True,'all_finite':True,'relative_l2':result['global_relative_l2'],
        'maximum_absolute_difference':result['tensors'][0]['maximum_absolute_difference'],
        'allclose_element_violations':result['tensors'][0]['allclose_element_violations'],'tolerances':LOGIT_TOLERANCES}
