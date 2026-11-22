"""Native final-assistant reasoning/action targets, ending at EOS; no MLX."""
import argparse
import hashlib
import json
from pathlib import Path
import time

PROJECT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(tokenizer,row):
    messages=row['messages'];last=messages[-1]
    if last.get('role')!='assistant'or not isinstance(last.get('reasoning_content'),str)or not last['reasoning_content'].strip()or not isinstance(last.get('content'),str):raise ValueError('Nonempty native reasoning and action target required')
    if any(tag in last['reasoning_content']+last['content']for tag in('<think>','</think>')):raise ValueError('Template owns reasoning markers; no manual/empty think blocks')
    json.loads(last['content'])
    full=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,enable_thinking=True,truncation=False,return_dict=False)
    prefix=tokenizer.apply_chat_template(messages[:-1],tokenize=True,add_generation_prompt=True,enable_thinking=True,truncation=False,return_dict=False)
    if not isinstance(full,list)or not isinstance(prefix,list)or not prefix or full[:len(prefix)]!=prefix:raise ValueError('Native prefix is not an exact token prefix of full assistant target')
    eos=tokenizer.eos_token_id
    if type(eos)is not int:raise ValueError('One authenticated canonical EOS is required')
    end=max((i+1 for i,t in enumerate(full)if t==eos),default=0)
    if not len(prefix)<end:raise ValueError('Final assistant EOS missing or no target')
    tail=tokenizer.decode(full[end:],skip_special_tokens=False)
    if tail.strip():raise ValueError('Only structural whitespace may follow final EOS')
    ids=full[:end];target=tokenizer.decode(ids[len(prefix):],skip_special_tokens=False)
    if last['reasoning_content'].strip()not in target or last['content'].strip()not in target:raise ValueError('Native template dropped reasoning or action from supervised target')
    if any(type(t)is not int or t<0 for t in ids):raise ValueError('Invalid token ID')
    return ids,len(prefix),{'raw_full_tokens':len(full),'tokens_through_EOS':end,'prefix_tokens':len(prefix),'target_tokens':end-len(prefix),'canonical_EOS':eos,'post_EOS_structural_tokens_excluded':len(full)-end,'nonempty_reasoning_rendered':True}


def audit(data,out):
    import qwen38_local as local
    local.prepare_local()
    from shared_validation_offline import load_verified_tokenizer
    tokenizer=load_verified_tokenizer();data=Path(data).resolve();out=Path(out).resolve()
    if 'onedrive'in str(data).lower()or not out.is_relative_to((PROJECT/'.local').resolve()):raise ValueError('Private non-OneDrive repo audit storage required')
    out.mkdir(parents=True,exist_ok=False);result={'schema':'research4.general-preservation-token-audit.v1','renderer_sha256':sha(__file__),'enable_thinking':True,'explicit_empty_think':False,'final_opened':False,'splits':{}}
    for split in('train','dev'):
        source=data/(split+'.jsonl');rows=[json.loads(line)for line in source.read_text().splitlines()];lengths=[];targets=[];excluded=0
        with(out/(split+'_tokens.jsonl')).open('x')as f:
            for index,row in enumerate(rows):
                ids,prefix,meta=render(tokenizer,row);lengths.append(len(ids));targets.append(len(ids)-prefix);excluded+=meta['post_EOS_structural_tokens_excluded']
                f.write(json.dumps({'row_index':index,'id':row.get('id',str(index)),'token_ids':ids,'prefix_tokens':prefix,'render':meta},separators=(',',':'))+'\n')
        result['splits'][split]={'rows':len(rows),'data_sha256':sha(source),'token_cache_sha256':sha(out/(split+'_tokens.jsonl')),'min_tokens':min(lengths),'max_tokens':max(lengths),'mean_tokens':sum(lengths)/len(lengths),'min_target_tokens':min(targets),'max_target_tokens':max(targets),'post_EOS_structural_tokens_excluded':excluded}
    maximum=max(s['max_tokens']for s in result['splits'].values());result['required_max_seq_length']=max(2048,32*((maximum+31)//32));result['all_semantic_tokens_preserved']=True
    (out/'AUDIT.json').open('x').write(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',required=True,type=Path);parser.add_argument('--out',required=True,type=Path);args=parser.parse_args();print(json.dumps(audit(args.data,args.out),indent=2))
