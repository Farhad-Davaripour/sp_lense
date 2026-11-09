"""Bounded CPU checks of native hidden/cotangent evidence; no MLX import."""
import math
from pathlib import Path

from game_all_layers_layout import read_tensor_header
from game_selected_head_v3_compare import _validated_header,compare_gradients,sha


def native_descriptor(path,key,shape):
    header,offset,_=read_tensor_header(path);tensors={name:entry for name,entry in header.items()if name!='__metadata__'}
    if set(tensors)!={key}:raise ValueError('Expected exactly one native hidden tensor')
    spec=tensors[key];dtype=spec.get('dtype');size={'BF16':2,'F16':2,'F32':4}.get(dtype)
    if size is None or tuple(spec.get('shape',[]))!=tuple(shape):raise ValueError('Native hidden shape/dtype differs')
    if spec.get('data_offsets')!=[0,math.prod(shape)*size]or Path(path).stat().st_size!=offset+math.prod(shape)*size:
        raise ValueError('Native hidden payload interval differs')
    return {'dtype':dtype,'shape':list(shape),'offset':offset,'element_bytes':size,'sha256':sha(path)}


def verify_native_audit_copy(native_path,native_key,audit_path,audit_key,shape):
    import numpy as np
    native=native_descriptor(native_path,native_key,shape);_,offset=_validated_header(audit_path,{audit_key:tuple(shape)})
    remaining=math.prod(shape)
    with Path(native_path).open('rb')as a,Path(audit_path).open('rb')as b:
        a.seek(native['offset']);b.seek(offset)
        while remaining:
            count=min(262144,remaining);raw=a.read(count*native['element_bytes']);audit=b.read(count*4)
            if len(raw)!=count*native['element_bytes']or len(audit)!=count*4:raise ValueError('Truncated hidden evidence')
            if native['dtype']=='BF16':values=(np.frombuffer(raw,dtype='<u2').astype('<u4')<<16).view('<f4')
            elif native['dtype']=='F16':values=np.frombuffer(raw,dtype='<f2').astype('<f4')
            else:values=np.frombuffer(raw,dtype='<f4')
            expected=np.frombuffer(audit,dtype='<f4')
            if not np.isfinite(values).all()or not np.array_equal(values,expected):raise ValueError('Audit copy changed native hidden/cotangent values')
            remaining-=count
    return {'passed':True,'native_dtype':native['dtype'],'shape':list(shape),
        'native_sha256':native['sha256'],'audit_sha256':sha(audit_path),'native_shadow_values_equal':True}


def covector_support(path,shape,prefix,length):
    import numpy as np
    if len(shape)!=3 or shape[0]!=1:raise ValueError('Full hidden covector must have shape [1,T,D]')
    _,offset=_validated_header(path,{'hidden_gradient':tuple(shape)})
    allowed=set(range(prefix-1,length-1));nonzero_rows=[]
    with Path(path).open('rb')as stream:
        stream.seek(offset)
        for row in range(shape[1]):
            raw=stream.read(shape[2]*4)
            if len(raw)!=shape[2]*4:raise ValueError('Truncated hidden covector row')
            values=np.frombuffer(raw,dtype='<f4')
            if not np.isfinite(values).all():raise ValueError('Nonfinite hidden covector')
            if np.count_nonzero(values):
                if row not in allowed:raise ValueError('Head covector has a nonzero masked-out hidden row')
                nonzero_rows.append(row)
    if not nonzero_rows:raise ValueError('The reference objective has a zero hidden covector everywhere')
    return {'outside_mask_exact_zero':True,'nonzero_valid_rows':nonzero_rows,
        'allowed_predictor_rows':[prefix-1,length-2],'all_hidden_positions_checked':shape[1]}


def compare_covectors(reference_path,selected_path,shape,prefix,length,tolerances):
    reference_support=covector_support(reference_path,shape,prefix,length)
    selected_support=covector_support(selected_path,shape,prefix,length)
    result=compare_gradients(reference_path,selected_path,{'hidden_gradient':tuple(shape)},tolerances)
    return {'schema':'research4.full-hidden-covector-equivalence.v4','passed':result['passed'],
        'reference_support':reference_support,'selected_support':selected_support,'numeric_comparison':result}


def exact_native_values(reference_path,recomputed_path,key,shape):
    reference=native_descriptor(reference_path,key,shape);recomputed=native_descriptor(recomputed_path,key,shape)
    if reference['dtype']!=recomputed['dtype']:raise ValueError('Recomputed native hidden dtype differs')
    remaining=math.prod(shape)*reference['element_bytes'];equal=True
    with Path(reference_path).open('rb')as a,Path(recomputed_path).open('rb')as b:
        a.seek(reference['offset']);b.seek(recomputed['offset'])
        while remaining:
            size=min(1024**2,remaining);left=a.read(size);right=b.read(size)
            if len(left)!=size or len(right)!=size:raise ValueError('Truncated native hidden comparison')
            equal=equal and left==right;remaining-=size
    return {'passed':equal,'exact_native_values_equal':equal,'native_dtype':reference['dtype'],'shape':list(shape),
        'reference_sha256':reference['sha256'],'recomputed_sha256':recomputed['sha256']}
