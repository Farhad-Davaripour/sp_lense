"""Explicit full native Qwen loader; reachable only after owned root admission."""
import hashlib,importlib.metadata,json,os,sys,time
from pathlib import Path
from support import HERE,ROOT,require,sha,write_new,check_freeze
UNUSED_MTP=frozenset(('mtp.fc.weight','mtp.layers.0.input_layernorm.weight',
    'mtp.layers.0.mlp.down_proj.weight','mtp.layers.0.mlp.gate_proj.weight','mtp.layers.0.mlp.up_proj.weight',
    'mtp.layers.0.post_attention_layernorm.weight','mtp.layers.0.self_attn.k_norm.weight',
    'mtp.layers.0.self_attn.k_proj.weight','mtp.layers.0.self_attn.o_proj.weight','mtp.layers.0.self_attn.q_norm.weight',
    'mtp.layers.0.self_attn.q_proj.weight','mtp.layers.0.self_attn.v_proj.weight','mtp.norm.weight',
    'mtp.pre_fc_norm_embedding.weight','mtp.pre_fc_norm_hidden.weight'))
class DenyUnusedDependencies:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in ('transformer_lens','datasets','pyarrow'):raise ImportError('FORBIDDEN_NATIVE_DEPENDENCY')
def block_unused_dependencies():
    require(not any(n.split('.')[0] in ('transformer_lens','datasets','pyarrow') for n in sys.modules),'NO_ALREADY_IMPORTED_BLOCKED_DEPENDENCY')
    if not any(type(x) is DenyUnusedDependencies for x in sys.meta_path):sys.meta_path.insert(0,DenyUnusedDependencies())
def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while data:=f.read(8*1024*1024):h.update(data)
    return h.hexdigest()
def checkpoint():
    lock=json.loads((HERE/'CHECKPOINT.json').read_bytes());base=Path(lock['snapshot'])
    require(base.name=='2fc06364715b967f1860aea9cf38778875588b17','REVISION')
    for pin in lock['files']:
        path=base/pin['name'];require(path.stat().st_size==pin['bytes'] and file_sha(path)==pin['sha256'],'CHECKPOINT_BYTES')
    return lock
def coverage(named,state_shapes,checkpoint_keys,info):
    # Only this exact header-bound, source-declared unused MTP key set is excluded.
    excluded={k for k in checkpoint_keys if k.startswith('mtp.')}
    require(excluded==UNUSED_MTP,'EXACT_UNUSED_MTP_KEYS')
    required={k:tuple(v['shape']) for k,v in checkpoint_keys.items() if k not in excluded}
    tied='lm_head.weight';source='model.language_model.embed_tokens.weight'
    expected={**required,tied:required[source]}
    require(state_shapes==expected,'ALL_STATE_KEYS_SHAPES')
    pairs=dict(named);require(set(pairs)==set(expected),'ALL_PARAMETER_KEYS')
    require(pairs[tied] is pairs[source],'EXACT_TIED_ALIAS')
    require(len({id(p) for _,p in named})==len(required),'NO_EXTRA_PARAMETER_ALIASES')
    require(set(info)== {'missing_keys','unexpected_keys','mismatched_keys','error_msgs','conversion_errors'},'LOAD_INFO_SCHEMA')
    require(all(not value for value in info.values()),'LOAD_INFO_FAILURE')
    return {'checkpoint_key_count':len(checkpoint_keys),'native_unique_parameters':len(required),'named_occurrences':len(named),
        'excluded_exact_mtp_keys':sorted(excluded),'tied_alias':{tied:source},'complete_key_shape_coverage':True}

def load(counts,latch,execution,deadline):
    from authority import authenticate
    require(authenticate()['execution']==execution,'OWNED_AUTHORITY_JOIN')
    block_unused_dependencies()
    bootstrap=json.loads((HERE/'real_evidence'/'native_development_baseline_attempt_001'/'owned/production_worker/BOOTSTRAP.json').read_bytes())
    require(bootstrap['permission_received'] is True and bootstrap['actual_pid']==os.getpid() and bootstrap['execution']==execution,'OWNED_BEFORE_IMPORT')
    require(not any(n.split('.')[0] in ('transformer_lens','datasets','pyarrow') for n in sys.modules),'NO_BLOCKED_DEPENDENCY')
    check_freeze();locked=checkpoint();base=Path(locked['snapshot'])
    require(importlib.metadata.version('torch')=='2.13.0+cpu' and importlib.metadata.version('transformers')=='5.15.1','PINNED_PACKAGES')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    # No provider import occurs at module import time or before the checks above.
    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration,Qwen3_5Model,Qwen3_5TextModel,Qwen3_5VisionModel,Qwen3_5DecoderLayer
    from core import ForwardDerivativeGuard
    from receiver import NativeReceiver
    config=Qwen3_5Config.from_dict(locked['config'])
    require(config.architectures==['Qwen3_5ForConditionalGeneration'] and config.text_config.hidden_size==1024
        and config.text_config.num_hidden_layers==24 and config.text_config.vocab_size==248320 and config.tie_word_embeddings is True,'EXACT_FULL_CONFIG')
    guard=ForwardDerivativeGuard(Qwen3_5ForConditionalGeneration,counts,latch,deadline);guard.install()
    started=time.monotonic();model=None
    own_finalizer='_finalize_model_loading' in Qwen3_5ForConditionalGeneration.__dict__
    previous_descriptor=Qwen3_5ForConditionalGeneration.__dict__.get('_finalize_model_loading')
    original_finalizer=Qwen3_5ForConditionalGeneration._finalize_model_loading
    captured=[]
    def observed_finalizer(model,load_config,loading_info):
        result=original_finalizer(model,load_config,loading_info)
        require(not captured,'ONE_FINALIZER')
        captured.append({k:list(getattr(result,k)) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs','conversion_errors')})
        return result
    Qwen3_5ForConditionalGeneration._finalize_model_loading=staticmethod(observed_finalizer)
    try:
        counts.reserve('load')
        model,info=Qwen3_5ForConditionalGeneration.from_pretrained(str(base),config=config,dtype=torch.float32,
            attn_implementation='eager',local_files_only=True,trust_remote_code=False,output_loading_info=True,
            ignore_mismatched_sizes=False,weights_only=True,use_safetensors=True)
        model.eval()
        require(type(model) is Qwen3_5ForConditionalGeneration and type(model.model) is Qwen3_5Model
            and type(model.model.visual) is Qwen3_5VisionModel and type(model.model.language_model) is Qwen3_5TextModel,'FULL_NATIVE_CLASSES')
        require(len(model.model.language_model.layers)==24 and all(type(x) is Qwen3_5DecoderLayer for x in model.model.language_model.layers),'EXACT_DECODERS')
        require(model.config._attn_implementation=='eager' and model.config.text_config._attn_implementation=='eager'
            and model.config.vision_config._attn_implementation=='eager','ALL_EAGER')
        named=list(model.named_parameters(remove_duplicate=False));shapes={k:tuple(v.shape) for k,v in model.state_dict().items()}
        require(len(captured)==1 and all(not x for x in info.values()),'COMPLETE_LOAD_REPORT')
        proof=coverage(named,shapes,locked['checkpoint_keys'],captured[0]);info=captured[0]
        require(all(p.device.type=='cpu' and p.dtype==torch.float32 and p.grad is None for _,p in named),'CPU_FLOAT32_PARAMETERS')
        require(model.model.rope_deltas is None and model.lm_head.bias is None,'NO_ALIAS_NO_ROPE')
        require(not any(n.split('.')[0] in ('transformer_lens','datasets','pyarrow') for n in sys.modules),'NO_BLOCKED_DEPENDENCY_AFTER_LOAD')
        receiver=NativeReceiver(model,guard,logits_to_keep=1)
        metadata=[{'name':n,'identity':id(p),'shape':list(p.shape),'dtype':str(p.dtype),'device':str(p.device),
            'bytes':p.numel()*p.element_size(),'version':p._version,'requires_grad':p.requires_grad} for n,p in named]
        write_new('LOADER_READY.json',{'execution':execution,'declared_class':type(model).__name__,'load_seconds':time.monotonic()-started,
            'checkpoint_lock_sha256':sha((HERE/'CHECKPOINT.json').read_bytes()),'coverage':proof,'loading_info':info,
            'native_initial_sha256':receiver.initial_digest,'native_initial_buffer_sha256':receiver.initial_buffer_digest,
            'native_parameters':metadata,'old_digest_equivalence_claimed':False})
        return receiver,torch,guard
    except BaseException:
        latch.stop('LOADER_FAILURE')
        guard.restore();raise
    finally:
        if own_finalizer:Qwen3_5ForConditionalGeneration._finalize_model_loading=previous_descriptor
        else:delattr(Qwen3_5ForConditionalGeneration,'_finalize_model_loading')
