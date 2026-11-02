"""Visible identity anchors; no status, event, revision or gold-label parsing."""
import re


def char_spans(text):
    match=re.match(r'Current worker=([^;\n]+); peer=([^;\n]+);',text)
    if not match or match[1]==match[2]:raise ValueError('Visible own/peer identity header required')
    return [list(match.span(1)),list(match.span(2))]


def token_masks(texts,offsets):
    import torch
    masks=[]
    for text,mapping in zip(texts,offsets.tolist()):
        spans=char_spans(text);pair=[[b>a and b>start and a<end for a,b in mapping] for start,end in spans]
        if any(not any(row) for row in pair):raise ValueError('Complete visible identity anchor was not tokenized')
        masks.append(pair)
    return torch.tensor(masks,dtype=torch.bool)
