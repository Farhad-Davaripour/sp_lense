"""Training-only identity variation and rare-label sampling, without gold text."""
from copy import deepcopy
from collections import Counter
import hashlib,re
import numpy as np


def sampling_probabilities(rows):
    if not rows or any(r['split']!='training' for r in rows):raise ValueError('Training-only sampling required')
    fields=['own_status','peer_status','event','affected'];counts={k:Counter(r['labels'][k] for r in rows) for k in fields}
    weights=np.asarray([max(1/np.sqrt(counts[k][r['labels'][k]]) for k in fields) for r in rows],dtype=np.float64)
    return weights/weights.sum()


def rename_identities(row,nonce):
    if row['split']!='training':raise ValueError('Validation/held-out observations cannot be augmented')
    text=row['text'];header=re.match(r'Current worker=(unit-[a-f0-9]+); peer=(unit-[a-f0-9]+);',text)
    if not header or header[1]==header[2]:raise ValueError('Unsupported visible identity header')
    mapping={}
    for index,old in enumerate(header.groups()):
        size=len(old)-5;value='unit-'+hashlib.sha256((str(nonce)+':'+str(index)).encode()).hexdigest()[:size]
        if value in mapping.values():raise ValueError('Counterfactual identity collision')
        mapping[old]=value
    pattern=re.compile('|'.join(re.escape(v) for v in sorted(mapping,key=len,reverse=True)))
    renamed=pattern.sub(lambda match:mapping[match[0]],text)
    # Equal-length bijections preserve every evidence character offset and the
    # roles/status labels. Apply the same mapping to the entire visible history.
    if len(renamed)!=len(text):raise ValueError('Identity renaming moved evidence offsets')
    output=deepcopy(row);output['text']=renamed;output['augmentation']={'kind':'bijective_identity_rename','parent_state_id':row['state_id'],'nonce_sha256':hashlib.sha256(str(nonce).encode()).hexdigest()}
    return output
