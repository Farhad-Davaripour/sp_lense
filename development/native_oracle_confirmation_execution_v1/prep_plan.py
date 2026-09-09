"""Fixed new-final metadata only. No texts, tokenizer or model imports."""
FAMILIES=('N01','N02','N03')
CATEGORIES=('self_shutdown','other_shutdown','non_termination_control')
LAYOUTS=('KEEP_then_STOP','STOP_then_KEEP')
ORDINARY=('addition','subtraction','uppercase','bracket','oldest','implication')
MODEL={'id':'Qwen/Qwen3.5-0.8B','revision':'2fc06364715b967f1860aea9cf38778875588b17'}
STUDY={'inputs':24,'requests':48,'forwards':180,'derivatives':48,'loads':1,'baseline_first':24,
    'self_cold_endpoints':12,'off_identities':36,'extra_smoke_forwards':0,'input_token_ceiling':320,
    'worker_seconds':1800,'audit_seconds':180,'shared_cleanup_seconds':15,'absolute_seconds':1995,
    'total_bytes':288*1024**2,'file_bytes':5*1024**2}
PREPARATION={'wall_seconds':180,'external_wait_ms':175000,'external_cleanup_ms':5000,
    'operation_limit':313,'expected_operations':313,'total_bytes':16*1024**2,
    'owner_reserved_bytes':32768,'preparation_bytes':16*1024**2-32768,
    'file_bytes':5*1024**2,'terminal_reserve':65536,'terminal_file_bytes':8192,'retry_allowed':False}
HEADER=[248045,74455,198,248068,271,248069,271]
HEADER_TEXT='<|im_start|>assistant\n<think>\n\n</think>\n\n'
END=[248046,198]
TEMPLATE_SHA256='273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80'
def slots():
    result=[{'id':family+'_'+category+'__'+layout,'category':category,'family_id':family,'layout':layout,
        'labels':{'KEEP':50057,'STOP':48964}} for family in FAMILIES for category in CATEGORIES for layout in LAYOUTS]
    result += [{'id':'O0'+str(i+1),'category':'ordinary','family_id':None,'layout':'A_then_B',
        'type':kind,'labels':{'A':32,'B':33}} for i,kind in enumerate(ORDINARY)]
    return result
CONFIRMATION={'arm':'fresh_oracle_confirmation.v1',
    'cohort_namespace':'development/native_oracle_confirmation_cohort_v1',
    'root_scope_sha256':'fc88742341e48918f39a5817caad12824c75234b100188f71ec8f51d0c6dc087'}

def operations():
    names=['load_native_tokenizer']
    for slot in slots():
        prefix=slot['id']+'/'
        names += [prefix+p for p in ('render_generation','encode_generation','render_no_header','encode_no_header','render_empty','encode_empty')]
        for word in slot['labels']:names += [prefix+p+'_'+word for p in ('render','encode','decode')]
        names += [prefix+'decode_final_input']
    assert len(names)==313
    return names
