"""Provider-free finite retained-record structural/token swap witness; no decoding.

Uses only committed prepared text/IDs and their accepted release/input joins.
Label symbols identify record syntax, never their meanings, categories or gold.
This does not load features or import any model/tokenizer/provider module.
"""
from pathlib import Path
import hashlib,json,re,struct,subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCES=(
 ('native_supervised_gate_capture_v1','42d6a2cf1fc8d49222cb05a11a8b7442c5046cce39380a0aba97857c32317a3f'),
 ('native_supervised_gate_capture_coverage_v1','37fbb4bd116e119a59fbf2d88507cecae5d6804cf3a6dd8badd9a863ba3ac950'),
 ('native_supervised_gate_capture_order_v1','caa9247e5011424ae5a0eaf9e130afa485fd4db413f4d8984a9ffadc26213a1d'),
 ('native_gate_coverage_increment_train_capture_v1','2e0a0b19ead9f01033c430851a57eb9ffc5d4e249b670c4b1fd36c12dff7a2a3'),
 ('native_gate_frozen_transfer_v1/capture','c707f335c4b1025e00896c3569c59cbad42cfe38dc0c2db2131ebaa749d72c19'))

def sha(raw):return hashlib.sha256(raw).hexdigest()
def canon(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def require(test,name):
    if not test:raise ValueError(name)
def read(path,expected=None):
    raw=path.read_bytes()
    require(len(raw)<=5*1024**2 and not path.is_symlink(),'FINITE_REGULAR_FILE')
    require(expected is None or sha(raw)==expected,'EXPECTED_RAW_HASH')
    rel=path.relative_to(ROOT).as_posix()
    require(subprocess.check_output(['git','cat-file','blob','HEAD:'+rel],cwd=ROOT)==raw,'RAW_GIT_EQUALITY')
    return json.loads(raw),{'path':rel,'sha256':sha(raw),'bytes':len(raw)}

def retained():
    records={};pins=[];roles={}
    for index,(name,release_hash) in enumerate(SOURCES):
        base=ROOT/'development'/name/'root_release'
        release,pin=read(base/'RELEASE.json',release_hash);pins.append(pin)
        require(release['approved'] is True,'ACCEPTED_RELEASE')
        text,pin=read(base/'TEXT_LOCK.json',release['text_lock_sha256']);pins.append(pin)
        inp,pin=read(base/'preparation/inputs.json',release['preparation_files']['inputs.json']);pins.append(pin)
        prompts={r['id']:r for r in text['rendered_prompts']}
        for case in inp['cases']:
            key=case['case_key'];p=base/'preparation'/(key+'.json')
            rec,pin=read(p,release['preparation_files'][key+'.json']);pins.append(pin)
            require(key not in records and rec['case_key']==key,'UNIQUE_RECORD_KEY')
            prompt=prompts[key]['prompt'];ids=rec['full_token_ids'];mask=rec['attention_mask']
            require(sha(prompt.encode())==rec['prompt_sha256']==prompts[key]['prompt_sha256'],'EXACT_PROMPT_JOIN')
            require(rec['rendered_chat']=='<|im_start|>user\n'+prompt+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n','EXACT_SAVED_CHAT_WRAPPER')
            require(sha(rec['rendered_chat'].encode())==rec['rendered_chat_utf8_sha256'],'EXACT_CHAT_HASH')
            require(rec['exact_generation_prefix'] and rec['exactly_one_content_token'] and not rec['truncation'],'RETAINED_BOUNDARY_PROOFS')
            require(ids==case['input']['input_ids'] and mask==case['input']['attention_mask']==[1]*len(ids),'EXACT_IDS_MASK_JOIN')
            require(rec['content_token_ids']==case['input']['token_map'],'EXACT_LABEL_ID_JOIN')
            require(rec['content_token_ids'] in ({'KEEP':50057,'STOP':48964},{'A':32,'B':33})
                and all(rec['full_suffix_token_ids'][label]==[token,248046,198] for label,token in rec['content_token_ids'].items()),'ADMITTED_SINGLETON_LABEL_PROOFS')
            require(rec['chat_template_sha256']=='273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80','UNCHANGED_CHAT_CONTRACT')
            require(sha(struct.pack('<'+'q'*len(ids),*ids))==case['input_binding']['derived_input_int64_le_sha256'],'INT64_ID_HASH')
            require(rec['final_input_index']==len(ids)-1==case['input']['final_input_index'],'TRUE_FINAL_INDEX_UNCHANGED')
            require(rec['generation_header_suffix_ids']==[248045,74455,198,248068,271,248069,271] and ids[-7:]==rec['generation_header_suffix_ids'],'EXACT_HEADER')
            records[key]=rec;roles[key]='EXPOSED_DIAGNOSTIC' if index==4 else 'TRAIN'
    return records,roles,pins

def pair_witness(left,right):
    texts=[left['rendered_chat'],right['rendered_chat']]
    pieces=[];orders=[]
    for text,rec in zip(texts,(left,right)):
        labels=rec['content_token_ids']
        matches=list(re.finditer(r'(?m)^(KEEP|STOP|A|B)\) [^\n]*\n',text))
        require(len(matches)==2 and {m.group(1) for m in matches}==set(labels),'EXACT_TWO_LABELED_RECORDS')
        require(all(text.count(label+')')==1 for label in labels),'UNIQUE_TEXT_LABEL_CLOSEPAREN_SENTINEL')
        require(matches[0].end()==matches[1].start(),'ADJACENT_OPTION_RECORDS')
        pieces.append((text[:matches[0].start()],matches[0].group(),matches[1].group(),text[matches[1].end():]))
        orders.append(tuple(m.group(1) for m in matches))
    a,b=pieces
    require(a[0]==b[0] and a[3]==b[3] and a[1]==b[2] and a[2]==b[1],'EXACT_TEXT_RECORD_SWAP')
    require(orders[0]==tuple(reversed(orders[1])) and left['content_token_ids']==right['content_token_ids'],'LABEL_ATTACHED_INVOLUTION')
    text_lcp=next(i for i,(x,y) in enumerate(zip(*texts)) if x!=y)
    require(text_lcp==len(a[0]) and a[0].endswith('\n'),'TEXT_DIVERGENCE_AT_FIRST_LABEL')
    ids=[left['full_token_ids'],right['full_token_ids']]
    lcp=next(i for i,(x,y) in enumerate(zip(*ids)) if x!=y)
    require(0<lcp<min(map(len,ids))-7 and ids[0][lcp-1]==198,'NONEMPTY_SHARED_PREFIX_BEFORE_LABEL')
    positions=[]
    for row,order,rec in zip(ids,orders,(left,right)):
        found=[]
        for label in order:
            token=rec['content_token_ids'][label]
            hits=[i for i in range(len(row)-1) if row[i:i+2]==[token,8]]
            require(len(hits)==1,'UNIQUE_BOUND_LABEL_CLOSEPAREN_SENTINEL')
            found.append(hits[0])
        require(found[0]==lcp and found[1]>lcp,'FIRST_DIVERGENCE_IS_FIRST_BOUND_LABEL')
        positions.append(found)
    lengths=[p[1]-lcp for p in positions];end=lcp+sum(lengths)
    require(ids[0][lcp:positions[0][1]]==ids[1][positions[1][1]:end]
        and ids[1][lcp:positions[1][1]]==ids[0][positions[0][1]:end]
        and ids[0][end:]==ids[1][end:],'EXACT_ID_RECORD_SWAP_AND_COMMON_SUFFIX')
    return {'view_keys':[left['case_key'],right['case_key']],
        'readout_index':lcp-1,'shared_prefix_length':lcp,'shared_prefix_ids_sha256':sha(canon(ids[0][:lcp])),
        'structural_prefix_utf8_sha256':sha(a[0].encode()),'structural_prefix_characters':len(a[0]),
        'label_symbols':orders,'first_divergent_label_ids':[row[lcp] for row in ids],
        'label_closeparen_positions':positions,'option_token_block_lengths':lengths,
        'last_shared_id':198,'final_input_indices':[len(row)-1 for row in ids],
        'text_and_id_swap_witness':True}

def main():
    records,roles,pins=retained();groups={}
    for key in records:
        stem=key.split('__',1)[0]
        groups.setdefault(stem,[]).append(key)
    require(len(records)==74 and len(groups)==37 and all(len(v)==2 for v in groups.values()),'EXACT74_VIEWS37_PAIRS')
    witnesses=[]
    for stem,keys in groups.items():
        keys.sort(key=lambda k:tuple(records[k]['content_token_ids']))
        w=pair_witness(*(records[k] for k in keys));w['scenario']=stem;w['role']=roles[keys[0]]
        require(all(roles[k]==w['role'] for k in keys),'NO_CROSS_ROLE_PAIR')
        witnesses.append(w)
    require(sum(w['role']=='TRAIN' for w in witnesses)==29 and sum(w['role']=='EXPOSED_DIAGNOSTIC' for w in witnesses)==8,'EXACT_ROLE_COUNTS')
    return {'schema':'prechoice_boundary_retained_witness.v1','status':'FINITE_WITNESS_PASS_REQUIRES_INDEPENDENT_ADMISSION',
        'script_sha256':sha(Path(__file__).read_bytes()),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'training_pairs':29,'diagnostic_pairs':8,'views':74,'pins':pins,'pairs':witnesses,
        'tokenizer_or_decoder_calls':0,'provider_imports':0,'feature_reads':0,'category_gold_outcome_used':False,
        'limitation':'Alignment uses previously accepted full-input roundtrip and singleton-label proofs plus unique labeled-token sentinels and exact whole-record token/text swaps; no new prefix decoding was performed.',
        'actual_work_authorized':False}

if __name__=='__main__':print(json.dumps(main(),sort_keys=True,separators=(',',':')))
