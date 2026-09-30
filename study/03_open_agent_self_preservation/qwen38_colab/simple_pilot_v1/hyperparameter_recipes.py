"""Small fixed hyperparameter comparison using the existing scientific runner."""
import random
import study_worker as study

RECIPES = [
    {'id':'H1_learning_rate_3e4','reference':'P2_completed_history',
     'change':'Learning rate only: 1e-4 to 3e-4.',
     'learning_rate':3e-4,'rank':8,'alpha':16},
    {'id':'H2_rank16','reference':'P2_completed_history',
     'change':'LoRA rank 8 to 16; keep alpha/rank equal to two.',
     'learning_rate':1e-4,'rank':16,'alpha':32},
]


def configure():
    config = study.read('EXECUTION_CONFIG.json')
    recipe = config['recipe']

    def adapter(base, seed=93):
        import torch
        from peft import LoraConfig, get_peft_model
        random.seed(seed)
        torch.manual_seed(seed)
        targets = [name for name,module in base.named_modules()
                   if '.language_model.layers.' in name and isinstance(module,torch.nn.Linear)]
        if not targets or any('visual' in name or 'lm_head' in name for name in targets):
            raise RuntimeError('Unexpected adapter targets')
        model = get_peft_model(base,LoraConfig(r=recipe['rank'],lora_alpha=recipe['alpha'],lora_dropout=0,
                                              target_modules=targets,task_type='CAUSAL_LM'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        if any('lora_' not in name for name,p in model.named_parameters() if p.requires_grad):
            raise RuntimeError('Non-adapter parameter trainable')
        study.emit({'stage':'hyperparameters_applied','candidate':recipe['id'],'rank':recipe['rank'],
                    'alpha':recipe['alpha'],'learning_rate':recipe['learning_rate'],'seed':seed,
                    'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)})
        return model,targets

    def optimizer_for(model, pilot=False):
        import torch
        from transformers import get_linear_schedule_with_warmup
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=recipe['learning_rate'],weight_decay=.01)
        scheduler = (torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.0) if pilot else
                     get_linear_schedule_with_warmup(optimizer,10,214))
        return optimizer,scheduler

    study.adapter = adapter
    study.optimizer_for = optimizer_for
