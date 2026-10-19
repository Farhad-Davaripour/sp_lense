"""Tokenizer-only positional controls; never loads Qwen parameters."""
import argparse
from pathlib import Path

from .io_utils import digest, dump, read
from .recognition_pilot import messages, select_rows


def span_tokens(offsets, start, length):
    return [index for index,(left,right) in enumerate(offsets) if right > start and left < start+length]


def audit(root, corpus, model_path, output):
    from transformers import AutoTokenizer
    root,corpus,model_path,output = map(Path,[root,corpus,model_path,output])
    if output.exists(): raise FileExistsError('Preserve tokenizer audit versions')
    cfg = read(root/'config/recognition_pilot.json')
    rows = select_rows(read(corpus/'CALIBRATION.json'),cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    if not tokenizer.is_fast: raise ValueError('Offsets require the pinned fast tokenizer')
    groups,spans = {},{}
    for row in rows:
        groups.setdefault(row['pair_id'],[]).append(row)
        for interface in cfg['interfaces']:
            chat = messages(row['observation'],interface)
            text = tokenizer.apply_chat_template(chat,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            tokenized = tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
            offsets = tokenized['offset_mapping']
            value = {'tokens':len(tokenized['input_ids']),'prompt_sha256':digest(chat)}
            for key in ['current_process','other_process']:
                start = text.index('"'+key+'":"')+len(key)+4
                value[key+'_token_count'] = len(span_tokens(offsets,start,len(row['observation'][key])))
            marker = '"notice":"'
            start = text.index(marker)+len(marker)
            value['notice_start_token'] = span_tokens(offsets,start,1)[0]
            spans[(row['record_id'],interface)] = value
    result = {'scope':'Tokenizer-only confound check. Matched total length does not imply aligned internal positions.',
              'records':len(rows),'pairs':len(groups),'interfaces':{},'Qwen_forwards':0,'optimizer_updates':0,'GPU_allocations':0}
    for interface in cfg['interfaces']:
        counts = {'pairs_with_equal_total_tokens':0,'pairs_with_shifted_notice_start':0,'pairs_with_unequal_current_ID_token_lengths':0}
        for left,right in groups.values():
            a,b = spans[(left['record_id'],interface)],spans[(right['record_id'],interface)]
            counts['pairs_with_equal_total_tokens'] += a['tokens'] == b['tokens']
            counts['pairs_with_shifted_notice_start'] += a['notice_start_token'] != b['notice_start_token']
            counts['pairs_with_unequal_current_ID_token_lengths'] += a['current_process_token_count'] != b['current_process_token_count']
        result['interfaces'][interface] = counts
    result['per_prompt'] = [{'record_id':key[0],'interface':key[1],**value} for key,value in spans.items()]
    dump(output,result)
    return {key:value for key,value in result.items() if key != 'per_prompt'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for field in ['corpus','model','output']: parser.add_argument(field)
    args = parser.parse_args()
    print(audit(Path(__file__).resolve().parents[1],args.corpus,args.model,args.output))
