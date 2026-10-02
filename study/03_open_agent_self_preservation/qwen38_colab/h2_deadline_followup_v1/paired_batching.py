"""Existing equal-example loss with matched padded input lengths for both jobs."""
def paired_losses(model,prepared,pad_token_id):
    import torch
    import torch.nn.functional as F
    maximum=max(x.shape[1] for x,_ in prepared)
    x=torch.full((len(prepared),maximum),pad_token_id,dtype=torch.long,device='cuda')
    labels=torch.full_like(x,-100)
    attention=torch.zeros_like(x)
    for i,(tokens,target) in enumerate(prepared):
        x[i,:tokens.shape[1]]=tokens[0]
        labels[i,:target.shape[1]]=target[0]
        last=int(torch.where(target[0]!=-100)[0][-1])+1
        attention[i,:last]=1
    positions=torch.where((labels[:,1:]!=-100).any(dim=0))[0]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(input_ids=x,attention_mask=attention,use_cache=False,logits_to_keep=positions).logits
        losses=[]
        for i in range(len(prepared)):
            valid=labels[i,positions+1]!=-100
            losses.append(F.cross_entropy(logits[i,valid].float(),labels[i,positions[valid]+1]))
    return torch.stack(losses)
