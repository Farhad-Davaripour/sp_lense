"""CANDIDATE ONLY. No model imports/loading. Not a complete scientific launcher or proof of fit."""

def configure_candidate(model,*,explicit_execution_approval=False):
 if explicit_execution_approval is not True:raise PermissionError('Checkpointed Kaggle execution has not been approved')
 configs=[model.config,getattr(model.config,'text_config',None)]
 for c in configs:
  if c is not None and hasattr(c,'attention_dropout'):assert float(c.attention_dropout)==0.0
 for c in getattr(model,'peft_config',{}).values():assert float(c.lora_dropout)==0.0
 for name,module in model.named_modules():
  if 'Dropout' in type(module).__name__ and hasattr(module,'p'):assert float(module.p)==0.0
 params=list(model.named_parameters());trainable=[(n,p) for n,p in params if p.requires_grad]
 assert len(trainable)==372 and all('lora_' in n for n,p in trainable)
 assert all(str(p.dtype)=='torch.float32' for n,p in params if getattr(p,'is_floating_point',lambda:True)())
 assert callable(getattr(model,'gradient_checkpointing_enable',None))
 original_cache=getattr(model.config,'use_cache',False)
 model.config.use_cache=False
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False,'preserve_rng_state':True,'determinism_check':'default'})
 model.train()
 assert model.training and getattr(model,'is_gradient_checkpointing',False)
 return {'original_use_cache':original_cache,'fit_verified':False,'gradient_or_CPU_equivalence_verified':False,'only_builtin_checkpointing_requested':True}

def prepare_frozen_CPU_evaluation(model,configured):
 model.gradient_checkpointing_disable();model.eval();model.config.use_cache=configured['original_use_cache']
 # Platform owner must first remove the GPU-training forward bridge, verify actual270,
 # move the SAME model to CPU, and verify adapter/CPUAdam/scheduler56 and frozenbase.
 return model
