"""Authenticated offline tokenizer, unchanged renderer and native sampler AST.

No model runtime import. Measurement caps are explicitly separate from the
unchanged2048 configured cap and do not qualify any Mac/kernel width.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.update(USE_TORCH='0',USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
ROOT=Path(__file__).resolve().parents[3]
REFERENCE=ROOT/'.local/reference-tokenizer'
REFERENCE_SHA='de89053ae8ca8fe5c5175649b39b27b1c7929b463280d7f55be1a1cb765cade4'
RENDERER=ROOT/'mac_studio/general_preservation_render_v1.py'
RENDERER_SHA='994e3db99784ca119922c0fd70c506465374b72b04e224cde31c2363ea988d65'
SAMPLER=ROOT/'.local/experiments/agent_security500_r16_last8_strict512_20261115/proofs/installed_mlx_lm_0322/trainer.py'
SAMPLER_SHA='0b6027fdb11803e951ceee52c83a81f82daa922ea8c6b7ca6888953461064e8e'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tokenizer_and_renderer():
    assert sha(REFERENCE/'REFERENCE_MANIFEST.json')==REFERENCE_SHA
    manifest=json.loads((REFERENCE/'REFERENCE_MANIFEST.json').read_text())
    assert manifest['source_base_manifest_sha256']=='e126c678d420b65a2c5157c0bca0a3fb162cad6b96e4a067912418e0959d7461'
    for item in manifest['files']:
        path=REFERENCE/item['path'];assert path.stat().st_size==item['bytes']and sha(path)==item['sha256']
    assert len(manifest['files'])==5 and sha(RENDERER)==RENDERER_SHA
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(REFERENCE),local_files_only=True,trust_remote_code=False)
    assert tokenizer.eos_token_id==248046 and tokenizer.pad_token_id==248044
    assert sorted(json.loads((REFERENCE/'config.json').read_text())['eos_token_id'])==[248044,248046]
    spec=importlib.util.spec_from_file_location('unchanged_own_reason_action_renderer',RENDERER)
    renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
    return tokenizer,renderer


def sampler_mask_measurement(cache):
    """Execute exact installed sampler with NumPy stand-in, no semantic truncation."""
    import numpy as np
    assert sha(SAMPLER)==SAMPLER_SHA
    source=SAMPLER.read_text();tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='iterate_batches')
    env={'np':np,'mx':SimpleNamespace(array=np.array),'CacheDataset':type('UnusedCacheDataset',(),{}),
         'rprint':lambda *a:(_ for _ in ()).throw(AssertionError('No truncation/warning allowed'))}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SAMPLER)+'#CPU_ARRAY_STANDIN','exec'),env)
    maximum=max(len(row['token_ids'])for row in cache)
    measurement_cap=max(2048,1+32*((maximum+31)//32))
    lookup={(tuple(row['token_ids']),row['prefix_tokens']):row['id']for row in cache}
    observed={}
    dataset=[(row['token_ids'],row['prefix_tokens'])for row in cache]
    for batch,lengths in env['iterate_batches'](dataset,1,measurement_cap,loop=False,seed=0):
        p,L=(int(x)for x in lengths[0]);ids=tuple(int(x)for x in batch[0,:L]);ident=lookup[ids,p]
        positions=np.arange(1,batch.shape[1]);mask=(positions>=p)&(positions<L)
        target=batch[0,1:][mask]
        assert list(target)==list(ids[p:])and len(target)==L-p
        assert int(target[-1])==248046 and 248044 not in target
        assert (batch.shape[1]-1)==32*((L+31)//32)
        observed[ident]={'nativeX':batch.shape[1]-1,'raw_width':batch.shape[1],
                        'mask_supervised_tokens':int(mask.sum()),'EOS_supervised':True,'PAD_supervised':False}
    return {'actual_sampler_source_sha256':SAMPLER_SHA,'standin':'mx.array replaced by numpy.array; actual native Python sampler function AST',
            'measurement_only_no_truncation_cap':measurement_cap,'unchanged_configured_legacy_cap':2048,
            'measurement_cap_is_NOT_trainer_change_or_model_qualification':True,'numpy_version':np.__version__,'rows':observed}
