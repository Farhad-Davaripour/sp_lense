"""Input-only pre-option selector and saved-record contract, no provider imports."""
import hashlib,json,struct
CONTRACT={'checkpoint':'Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17',
    'native_target':'model.language_model.layers.10','position':'last_shared_preoption_input',
    'selector':'retained_exact_option_record_swap_lcp_v1','residual_dtype':'float32','width':1024,
    'final_logit_position':'final_input','pair_average':'0.5*float64(hcanonical0)+0.5*float64(hcanonical1)'}
def require(ok,code):
    if not ok:raise ValueError(code)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def canon(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def int_hash(ids):return sha(struct.pack('<'+'q'*len(ids),*ids))
def bind(ids,witness,case_key,certificate_sha256,*,expected_input_ids_sha256):
    require(type(ids) is list and 1<=len(ids)<=320 and all(type(v) is int and 0<=v<248320 for v in ids),'EXACT_RETAINED_INPUT_IDS')
    require(type(certificate_sha256) is str and len(certificate_sha256)==64 and all(c in '0123456789abcdef' for c in certificate_sha256),'CERTIFICATE_SHA256')
    require(int_hash(ids)==expected_input_ids_sha256,'RETAINED_FULL_INPUT_ID_PIN')
    require(witness['text_and_id_swap_witness'] is True and case_key in witness['view_keys'],'CERTIFIED_EXACT_VIEW')
    view=witness['view_keys'].index(case_key);index=witness['readout_index'];length=witness['shared_prefix_length']
    require(type(index) is int and index==length-1 and 0<=index<len(ids)-7,'PREOPTION_INDEX_BOUND')
    require(witness['final_input_indices'][view]==len(ids)-1,'TRUE_FINAL_BOUNDARY')
    require(ids[index]==witness['last_shared_id']==198 and ids[index+1]==witness['first_divergent_label_ids'][view],'EXACT_PREFIX_LABEL_BOUNDARY')
    require(sha(canon(ids[:length]))==witness['shared_prefix_ids_sha256'],'EXACT_SHARED_PREFIX_IDS')
    return {'schema':'prechoice_selector.v1','case_key':case_key,'certificate_sha256':certificate_sha256,
        'witness_sha256':sha(canon(witness)),'input_ids_sha256':int_hash(ids),'shared_prefix_ids_sha256':witness['shared_prefix_ids_sha256'],
        'readout_index':index,'first_option_label_id':ids[index+1],'final_input_index':len(ids)-1,'feature_contract':CONTRACT}
def validate(ids,selector):
    require(selector['schema']=='prechoice_selector.v1' and selector['feature_contract']==CONTRACT,'EXACT_PRECHOICE_CONTRACT')
    require(type(ids) is list and all(type(v) is int and 0<=v<248320 for v in ids) and 1<=len(ids)<=320,'EXACT_RETAINED_INPUT_IDS')
    i=selector['readout_index']
    require(type(i) is int and 0<=i<len(ids)-7 and selector['final_input_index']==len(ids)-1,'PREOPTION_INDEX_BOUND')
    require(int_hash(ids)==selector['input_ids_sha256'] and sha(canon(ids[:i+1]))==selector['shared_prefix_ids_sha256'],'EXACT_SELECTOR_INPUT_JOIN')
    require(ids[i]==198 and ids[i+1]==selector['first_option_label_id'] and ids[i+1] in (32,33,50057,48964),'EXACT_PREFIX_LABEL_BOUNDARY')
    for name in ('certificate_sha256','witness_sha256'):
        require(type(selector[name]) is str and len(selector[name])==64 and all(c in '0123456789abcdef' for c in selector[name]),'SELECTOR_PROVENANCE_HASH')
    return i
def validate_capture(ids,selector,capture):
    i=validate(ids,selector)
    require(capture['hook']=='blocks.10.hook_out' and capture['native_target']==CONTRACT['native_target'] and capture['hook_calls']==1,'EXACT_NATIVE_SITE')
    require(capture['final_input_index']==len(ids)-1 and capture['readout_index']==i
        and capture['readout_selector_sha256']==sha(canon(selector)) and capture['readout_input_ids_sha256']==int_hash(ids),'EXACT_RECORDED_SELECTOR')
    require(capture['readout_nonselected_positions']==len(ids)-1 and capture['all_positions_unchanged'] is True,'READONLY_ALL_POSITIONS')
    value=capture['readout_nonselected_sha256']
    require(type(value) is str and len(value)==64 and all(ch in '0123456789abcdef' for ch in value),'READOUT_NONSELECTED_HASH')
    require(capture['feature_contract']==CONTRACT and capture['logit_count']==248320 and capture['parameter_versions_unchanged'] is True,'EXACT_RECORDED_CONTRACT')
    return True
