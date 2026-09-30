"""True microbatches with the original equal-per-example training objective.

Candidate for a separate GPU timing run. Not installed into the active campaign.
"""


def batch_losses(model, prepared, pad_token_id):
    import torch
    import torch.nn.functional as F
    maximum = max(x.shape[1] for x, _ in prepared)
    batch = len(prepared)
    x = torch.full((batch, maximum), pad_token_id, dtype=torch.long, device='cuda')
    labels = torch.full_like(x, -100)
    attention = torch.zeros_like(x)
    for row, (tokens, target) in enumerate(prepared):
        length = tokens.shape[1]
        x[row, :length] = tokens[0]
        labels[row, :length] = target[0]
        attention[row, :length] = 1
    positions = torch.where((labels[:, 1:] != -100).any(dim=0))[0]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(input_ids=x, attention_mask=attention, use_cache=False,
                       logits_to_keep=positions).logits
        losses = []
        for row in range(batch):
            valid = labels[row, positions + 1] != -100
            losses.append(F.cross_entropy(logits[row, valid].float(), labels[row, positions[valid] + 1]))
    return torch.stack(losses)


def optimize_batched(model, optimizer, scheduler, prepared, microbatch, pad_token_id):
    import torch
    if len(prepared) != 4 or microbatch not in (1, 2, 4):
        raise ValueError('This candidate keeps the effective optimizer batch at four examples')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    # Sort only within this fixed four-example update, preserving its membership
    # and giving each example the same 1/4 contribution as the original loop.
    ordered = sorted(enumerate(prepared), key=lambda pair: pair[1][0].shape[1], reverse=True)
    output = [None] * 4
    for start in range(0, 4, microbatch):
        group = ordered[start:start + microbatch]
        losses = batch_losses(model, [pair[1] for pair in group], pad_token_id)
        if not torch.isfinite(losses).all():
            raise RuntimeError('Nonfinite batch loss')
        (losses.sum() / 4).backward()
        for (original_index, _), loss in zip(group, losses.detach().cpu().tolist()):
            output[original_index] = loss
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                  1.0, error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    torch.cuda.synchronize()
    return output
