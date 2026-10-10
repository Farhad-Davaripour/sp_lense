"""Offline tokenizer-only preparation; never loads model weights or a GPU."""
import hashlib
import json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
import transformers

SOURCE=Path(__file__).parent
ROOT=Path('C:/Users/farha/Research4/artifacts/rotation_pilot_v1/executed_20261009/verified_native')
OUTPUT=Path('C:/Users/farha/Research4/artifacts/propagation_followup_v1/panel_inputs')
MODEL=Path('C:/Users/farha/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17')


def ids(value):
    if hasattr(value,'keys'):value=value['input_ids']
    if hasattr(value,'ids'):value=value.ids
    if hasattr(value,'tolist'):value=value.tolist()
    if len(value) and hasattr(value[0],'ids'):value=value[0].ids
    if len(value) and isinstance(value[0],list):value=value[0]
    return list(value)


def main():
    panel=json.loads((SOURCE/'PANEL.json').read_text())
    original=json.loads((ROOT/'source/pilot.json').read_text())
    panel['system']=original['system']
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    def prefix(text):
        return ids(tokenizer.apply_chat_template([{'role':'system','content':panel['system']},
            {'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=False))
    reference=json.loads((ROOT/'native/r4_01/tokens.json').read_text())['prefix_ids']
    assert prefix(original['prompts'][0]['text'])==reference,'Tokenizer/chat template differs from archived native prefix'
    for prompt in panel['prompts']:
        prompt['prefix_ids']=prefix(prompt['text'])
        prompt['prefix_length']=len(prompt['prefix_ids'])
    for pair in {p['pair'] for p in panel['prompts']}:
        prompts=[p for p in panel['prompts'] if p['pair']==pair]
        assert prompts[0]['prefix_length']==prompts[1]['prefix_length'],pair
        prompts[0]['matched_format_and_length']=prompts[1]['matched_format_and_length']=True
    continuation=ids(tokenizer.encode(panel['neutral_continuation'],add_special_tokens=False))
    assert 0<len(continuation)<=64
    assert tokenizer.eos_token_id not in continuation
    panel['forced_continuation_ids']=continuation
    panel['prediction_steps']=len(continuation)
    OUTPUT.mkdir(parents=True,exist_ok=False)
    vector=json.loads(Path('C:/Users/farha/Research4/artifacts/rotation_pilot_v1/inputs/vector_research2.json').read_text())
    v=np.asarray(vector['direction'],dtype=np.float32)
    with np.load(ROOT/'native/interventions.npz') as z:random1=z['random'].copy()
    random2=np.random.default_rng(panel['random2_seed']).standard_normal(1024).astype(np.float32)
    random2/=np.linalg.norm(random2)
    np.savez(OUTPUT/'panel_vectors.npz',v=v,random1=random1,random2=random2)
    panel['vectors_sha256']=hashlib.sha256((OUTPUT/'panel_vectors.npz').read_bytes()).hexdigest()
    panel['tokenizer_preparation']={'transformers':transformers.__version__,
        'original_native_prefix_reproduced_exactly':True,'weights_loaded':False,'GPU_allocated':False,
        'tokenizer_json_sha256':hashlib.sha256((MODEL/'tokenizer.json').read_bytes()).hexdigest(),
        'tokenizer_config_sha256':hashlib.sha256((MODEL/'tokenizer_config.json').read_bytes()).hexdigest(),
        'chat_template_sha256':hashlib.sha256((MODEL/'chat_template.jinja').read_bytes()).hexdigest()}
    (SOURCE/'PANEL.json').write_bytes((json.dumps(panel,indent=2)+'\n').encode())
    (OUTPUT/'PANEL.json').write_bytes((SOURCE/'PANEL.json').read_bytes())
    print(json.dumps({'prediction_steps':len(continuation),'prefix_lengths':{p['id']:p['prefix_length'] for p in panel['prompts']},
                      'vectors_sha256':panel['vectors_sha256'],'preparation':panel['tokenizer_preparation']}))


if __name__=='__main__':main()
