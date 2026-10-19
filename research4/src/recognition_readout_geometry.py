"""Readout-only logit lens on saved states, without loading or forwarding Qwen."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .io_utils import dump,read,sha
from .recognition_pilot import select_rows
from .recognition_role_contrast import contrasts,evaluate


def norm(x,weight,epsilon):
    return x * torch.rsqrt(x.square().mean(-1,keepdim=True)+epsilon) * (1+weight)


def audit(root,folder,corpus,model_path,output):
    from safetensors import safe_open
    from transformers import AutoTokenizer
    root,folder,corpus,model_path,output = map(Path,[root,folder,corpus,model_path,output])
    if output.exists(): raise FileExistsError('Preserve readout analyses')
    torch.set_num_threads(2)
    cfg = read(folder/'PRECALL_FREEZE.json')['config']
    study = read(model_path/'config.json')
    if not study['tie_word_embeddings'] or not study['text_config']['tie_word_embeddings']:
        raise ValueError('This release requires the pinned tied readout')
    rows = select_rows(read(corpus/'CALIBRATION.json'),cfg)
    records = read(folder/'RECORDS.json')
    weights = model_path/'model.safetensors-00001-of-00001.safetensors'
    if sha(weights) != '04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696':
        raise ValueError('Pinned checkpoint changed')
    tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    ids = [tokenizer.encode(letter,add_special_tokens=False) for letter in 'AB']
    if any(len(value)!=1 for value in ids): raise ValueError('Unexpected answer-token encoding')
    with safe_open(weights,framework='pt',device='cpu') as data:
        embeddings = data.get_slice('model.language_model.embed_tokens.weight')
        head = torch.cat([embeddings[value[0]:value[0]+1].float() for value in ids])
        scale = data.get_tensor('model.language_model.norm.weight').float()
    with np.load(folder/'FROZEN_FEATURES.npz',allow_pickle=False) as data:
        index = {key:i for i,key in enumerate(data['keys'].tolist())}
        features = torch.tensor(np.stack([data['X'][index[row['record_id']+':target']] for row in rows]))
    with torch.no_grad():
        projected = (norm(features,scale,study['text_config']['rms_norm_eps']) @ head.T).numpy()
    actual = np.asarray([record['queries']['target']['logits'][:2] for record in records])
    maximum_error = float(np.max(np.abs(projected[:,-1]-actual)))
    if not np.allclose(projected[:,-1],actual,rtol=1e-6,atol=1e-3):
        raise ValueError('Final layer projection does not reconstruct raw language-head scores')
    results = {}
    for layer_index,layer in enumerate(cfg['capture_layers_zero_based']):
        replacement = [{'record_id':row['record_id'],'queries':{'target':{'logits':projected[i,layer_index].tolist()+[0.]*4}}} for i,row in enumerate(rows)]
        results[str(layer)] = evaluate(rows,contrasts(rows,replacement))
    result = {'status':'posthoc_saved_state_analysis','observations':len(rows),'layer_results':results,
              'maximum_final_letter_logit_error':maximum_error,'source_weight_sha256':sha(weights),
              'letter_token_ids':[v[0] for v in ids],'norm_epsilon':study['text_config']['rms_norm_eps'],
              'formula':'D=(W_A-W_B)^T [RMSNorm(h(o))-RMSNorm(h(role_swap(o)))]',
              'normalization':'Qwen zero-centered RMS weight: x/sqrt(mean(x^2)+eps) * (1+weight)',
              'new_Qwen_backbone_forwards':0,'optimizer_updates':0,'GPU_allocations':0,
              'scope':'Final-layer score is an exact readout reconstruction within recorded FP32 tolerance. Earlier-layer frozen final-norm/head applications are an untuned logit-lens diagnostic, not a fitted probe or causal test. Do not select a layer on transfer outcomes.'}
    dump(output,result)
    np.savez_compressed(output.with_suffix('.npz'),projected_AB_logits=projected,record_ids=np.asarray([row['record_id'] for row in rows]),layers=np.asarray(cfg['capture_layers_zero_based']))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    for field in ['folder','corpus','model','output']: parser.add_argument(field)
    args=parser.parse_args()
    print(audit(Path(__file__).resolve().parents[1],args.folder,args.corpus,args.model,args.output))
