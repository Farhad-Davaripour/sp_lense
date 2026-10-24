"""Read-only tensor export hooks and source-bound disk/identity contracts.

No model loader, forward, optimizer step, provider submission or retry exists
here. Hooks read an already-owned stage and preserve complete raw state.
"""
import io,math,shutil
from pathlib import Path
from .io_utils import digest,sha

GiB=1024**3
PROFILE='v2-full-state-raw-evidence-v1'
RAW_BYTES=4*GiB
RESERVE_BYTES=9*GiB


def validate_export_plan(stage):
    expected={'raw_evidence_profile':PROFILE,'maximum_raw_evidence_bytes':RAW_BYTES,
              'physical_export_reserve_bytes':RESERVE_BYTES}
    if any(type(stage.get(k)) is not type(v) or stage[k]!=v for k,v in expected.items()):
        raise PermissionError('Reviewed full-state raw export envelope required')
    return expected


def disk_admission(cache,output,pin,stage,usage=None):
    validate_export_plan(stage);usage=shutil.disk_usage if usage is None else usage
    total=sum(v['size'] for v in pin['files'])
    if total!=55575985859:raise ValueError('Original complete model download budget changed')
    required=total+2*GiB+RESERVE_BYTES+RAW_BYTES
    if usage(cache).free<required or usage(output).free<RESERVE_BYTES+RAW_BYTES:
        raise OSError('Insufficient complete model/raw/archive/reservation disk space')
    return {'download_file_bytes':total,'required_cache_free_bytes':required,
            'raw_evidence_profile':PROFILE,'maximum_raw_evidence_bytes':RAW_BYTES,
            'physical_export_reserve_bytes':RESERVE_BYTES,'runtime_capacity_admitted':False}


def provider_binding(stage,workspace,value):
    """A controller publishes this one-use record after submission, before fit.

    This record is not allocation or release evidence. A later independent
    reader reconciles both provider APIs against the same ID and raw archive.
    """
    expected={'source_commit':stage['source_commit'],'phase':stage['phase'],
              'host':workspace['host'],'destination':stage['destination']}
    if any(value.get(k)!=v for k,v in expected.items()) or type(value.get('run_id')) is not int or value['run_id']<=0:
        raise PermissionError('Exact controller-owned provider binding required before model work')
    return value['run_id']


def optimizer_state(optimizer,adapters):
    """Serialize existing named AdamW state; never construct or step one."""
    import torch
    values={}
    for name,module in adapters.items():
        for part in ['A','B']:
            parameter=getattr(module,part);state=optimizer.state.get(parameter)
            if state is None:raise ValueError('Incomplete optimizer state')
            value={}
            for key in ['step','exp_avg','exp_avg_sq']:
                tensor=state[key]
                if not isinstance(tensor,torch.Tensor) or not torch.isfinite(tensor).all():raise ValueError('Finite raw AdamW tensors required')
                if key!='step' and (tensor.dtype!=torch.float32 or tensor.shape!=parameter.shape):
                    raise ValueError('Raw AdamW precision/geometry differs')
                value[key]=tensor.detach().cpu().clone()
            values[name+'.'+part]=value
    if len(optimizer.state)!=len(values):raise ValueError('Unknown original/missing parameters in optimizer state')
    return values


def write_optimizer_state(optimizer,adapters,output):
    import torch
    target=Path(output)/'OPTIMIZER_FINAL.pt'
    with target.open('xb') as stream:torch.save(optimizer_state(optimizer,adapters),stream)
    return {'path':'OPTIMIZER_FINAL.pt','bytes':target.stat().st_size,'sha256':sha(target)}


def runtime_evidence(actor):
    """Read actual current versions/devices/quantization, with no model call."""
    import torch
    from importlib.metadata import version
    from .composite_frozen_probe import text_projection,projection_quantization_proof
    quant={}
    for name,parameter in actor.base_references:
        if parameter.dtype==torch.uint8:
            module=actor.model.get_submodule(name.removesuffix('.weight'))
            module=getattr(module,'base',module)
            quant[name]=projection_quantization_proof(name,module,parameter)
    actual_map={}
    for key,value in actor.model.hf_device_map.items():
        if type(value) is int:index=value
        elif str(value).startswith('cuda:'):index=int(str(value).split(':')[1])
        else:raise ValueError('CPU/disk/automatic placement is outside the declared runtime')
        actual_map[key]=index
    return {'torch':str(torch.__version__),'packages':{name:version(name) for name in
        ['transformers','bitsandbytes','accelerate','safetensors','numpy','psutil']},
        'visible_GPUs':torch.cuda.device_count(),'BF16_supported':torch.cuda.is_bf16_supported(),
        'TF32_enabled':bool(torch.backends.cuda.matmul.allow_tf32),'device_map':actual_map,
        'kernel_profile':actor.kernel_profile,'full_attention':actor.kernel_proof['full_attention'],
        'quantized_original_projections':quant}


def detector_evidence(detector):
    from .composite_training import tensor_digest
    records=[]
    for name,value in detector.named_parameters():
        records.append({'name':name,'shape':list(value.shape),'dtype':str(value.dtype),'device':str(value.device),
            'requires_grad':value.requires_grad,'gradient_is_none':value.grad is None,'sha256':tensor_digest(value)})
    runtime={'config':detector.cfg,'head_mode':detector.head_mode,'projection_names':list(detector.adapters),
        'projection_scales':{name:module.scale for name,module in detector.adapters.items()},
        'all_modules_in_eval_mode':all(not module.training for module in detector.modules()),
        'full_attention':detector.encoder.config._attn_implementation}
    return records,runtime


class BypassWriter:
    def __init__(self,root):self.root=Path(root);self.records={};self.shape=None
    def __call__(self,index,kind,input_token_ids,logits):
        import numpy as np
        from .composite_runtime import contained
        if type(index) is not int or index not in [0,1] or kind not in ['wrapped_OFF','bare','wrapped_ON']:
            raise ValueError('Exact two-prompt raw bypass namespace required')
        array=logits.detach().float().cpu().numpy()
        if array.dtype!=np.float32 or array.ndim!=3 or array.shape[:2]!=(1,1) or not np.isfinite(array).all():
            raise ValueError('Full finite raw vocabulary logits required')
        if self.shape is None:self.shape=list(array.shape)
        if self.shape!=list(array.shape):raise ValueError('Raw bypass geometry changed')
        path='bypass/'+str(index).zfill(2)+'_'+kind+'.npy';target=contained(self.root,path)
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:np.save(stream,array,allow_pickle=False)
        row=self.records.setdefault(index,{'input_token_ids':list(input_token_ids)})
        if row['input_token_ids']!=input_token_ids:raise ValueError('Matched-prefix bypass prompt changed')
        row[kind]={'path':path,'sha256':sha(target)}
    def manifest(self):
        if set(self.records)!={0,1} or any(set(row)!={'input_token_ids','wrapped_OFF','bare','wrapped_ON'} for row in self.records.values()):
            raise ValueError('Complete two-prompt raw bypass export required')
        return {'logit_shape':self.shape,'records':[self.records[i] for i in range(2)]}
