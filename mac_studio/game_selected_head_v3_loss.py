"""V3 fixed-window native-head projection; full causal body is unchanged.

No MLX import occurs until the explicitly owned model diagnostic calls this
module. Every rendered suffix must fit the fixed 32-position head window.
"""
HEAD_WINDOW=32
LOSS_VERSION='full_body_dynamic_gather32_native_head_P_le_target_lt_L.v3'


def gather_positions(prefix,length,width):
    """CPU geometry only; the differentiated route uses GPU array indices."""
    if any(type(value)is not int for value in(prefix,length,width))or not 0<prefix<length<=width<=1024:
        raise ValueError('Invalid full-context prompt/target geometry')
    if length-prefix>HEAD_WINDOW:raise ValueError('Suffix exceeds fixed head window; truncation is forbidden')
    target_positions=[prefix+offset for offset in range(HEAD_WINDOW)]
    return {'target_positions':target_positions,
        'predictor_indices':[min(max(index-1,0),width-2)for index in target_positions],
        'target_indices':[min(max(index,0),width-1)for index in target_positions],
        'valid':[index<length for index in target_positions]}


def assert_head_contract(model):
    language=model.language_model
    if language.args.tie_word_embeddings or language.sharding_group is not None:
        raise ValueError('Only the authenticated native untied, unsharded LM head is supported')


def components_with_runtime(runtime,neural,model,batch,lengths):
    inputs=batch[:,:-1]
    # Keep every original prefix/target/padding input and the native final norm.
    hidden=model.language_model.model(inputs,cache=None)
    positions=lengths[:,0:1]+runtime.arange(HEAD_WINDOW)[None,:]
    predictors=runtime.clip(positions-1,0,inputs.shape[1]-1)
    selected=runtime.take_along_axis(hidden,predictors[:,:,None],axis=1)
    logits=model.language_model.lm_head(selected)
    targets=runtime.take_along_axis(batch,runtime.clip(positions,0,batch.shape[1]-1),axis=1)
    valid=positions<lengths[:,1:]
    nll=neural.losses.cross_entropy(logits,targets)*valid
    return nll.astype(runtime.float32).sum(),valid.sum(),logits


def sum_with_runtime(runtime,neural,model,batch,lengths):
    total,count,_=components_with_runtime(runtime,neural,model,batch,lengths)
    return total,count


def full_components_with_runtime(runtime,neural,model,batch,lengths):
    """Independent full native head reference with the same corrected mask."""
    logits=model(batch[:,:-1]);targets=batch[:,1:]
    positions=runtime.arange(1,targets.shape[1]+1)
    valid=runtime.logical_and(positions>=lengths[:,0:1],positions<lengths[:,1:])
    nll=neural.losses.cross_entropy(logits,targets)*valid
    requested=lengths[:,0:1]+runtime.arange(HEAD_WINDOW)[None,:]-1
    gathered=runtime.take_along_axis(logits,runtime.clip(requested,0,logits.shape[1]-1)[:,:,None],axis=1)
    return nll.astype(runtime.float32).sum(),valid.sum(),gathered


def masked_target_sum(model,batch,lengths):
    import mlx.core as mx
    import mlx.nn as nn
    return sum_with_runtime(mx,nn,model,batch,lengths)


def masked_target_loss(model,batch,lengths):
    total,count=masked_target_sum(model,batch,lengths)
    return total/count,count
