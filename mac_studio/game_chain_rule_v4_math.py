"""V4 full-context head/body chain-rule primitives; no MLX import on import.

The controller must materialize/save/reload native-dtype hidden values before
head differentiation, release that graph, and supply the saved constant native
cotangent to the body VJP. Nothing here crops model context or changes GDN math.
"""
HEAD_WINDOW=32


def body_hidden(model,batch):
    return model.language_model.model(batch[:,:-1],cache=None)


def head_components(runtime,neural,language,hidden,batch,lengths,selected=False):
    if selected:
        positions=lengths[:,0:1]+runtime.arange(HEAD_WINDOW)[None,:]
        predictors=runtime.clip(positions-1,0,hidden.shape[1]-1)
        selected_hidden=runtime.take_along_axis(hidden,predictors[:,:,None],axis=1)
        logits=language.lm_head(selected_hidden)
        targets=runtime.take_along_axis(batch,runtime.clip(positions,0,batch.shape[1]-1),axis=1)
        valid=positions<lengths[:,1:]
        valid_logits=logits
    else:
        # The head sees exactly the body's native output dtype. Upcasting
        # before this quantized projection would change the reference method.
        logits=language.lm_head(hidden)
        targets=batch[:,1:];positions=runtime.arange(1,targets.shape[1]+1)
        valid=runtime.logical_and(positions>=lengths[:,0:1],positions<lengths[:,1:])
        requested=lengths[:,0:1]+runtime.arange(HEAD_WINDOW)[None,:]-1
        valid_logits=runtime.take_along_axis(logits,runtime.clip(requested,0,hidden.shape[1]-1)[:,:,None],axis=1)
    total=(neural.losses.cross_entropy(logits,targets)*valid).astype(runtime.float32).sum()
    count=valid.sum()
    return total/count,(count,valid_logits)


def direct_selected(runtime,neural,model,batch,lengths):
    hidden=body_hidden(model,batch)
    mean,(count,logits)=head_components(runtime,neural,model.language_model,hidden,batch,lengths,True)
    return mean,(count,logits,hidden)


def outer_full_components(runtime,neural,model,batch,lengths):
    """Actual outer-model forward reference for the final eval-mode gate."""
    logits=model(batch[:,:-1]);targets=batch[:,1:]
    positions=runtime.arange(1,targets.shape[1]+1)
    valid=runtime.logical_and(positions>=lengths[:,0:1],positions<lengths[:,1:])
    total=(neural.losses.cross_entropy(logits,targets)*valid).astype(runtime.float32).sum();count=valid.sum()
    requested=lengths[:,0:1]+runtime.arange(HEAD_WINDOW)[None,:]-1
    gathered=runtime.take_along_axis(logits,runtime.clip(requested,0,logits.shape[1]-1)[:,:,None],axis=1)
    return total/count,(count,gathered)


def body_contraction(runtime,model,batch,constant_cotangent):
    """Reference body VJP scalar Q; Q is NOT NLL and is not renormalized."""
    hidden=body_hidden(model,batch)
    if hidden.shape!=constant_cotangent.shape or hidden.dtype!=constant_cotangent.dtype:
        raise ValueError('The native hidden cotangent shape/dtype must be preserved')
    stopped=runtime.stop_gradient(constant_cotangent)
    # Multiplication stays in native dtype; accumulation is F32. The supplied
    # covector already comes from MEAN head loss, so there is no /token_count.
    contraction=(hidden*stopped).astype(runtime.float32).sum()
    return contraction,hidden


def predictor_rows(prefix,length,input_width):
    if any(type(value)is not int for value in(prefix,length,input_width))or not 0<prefix<length<=input_width+1:
        raise ValueError('Invalid original target geometry')
    if length-prefix>HEAD_WINDOW:raise ValueError('Every suffix must fit 32; truncation is forbidden')
    return list(range(prefix-1,length-1))
