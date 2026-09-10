"""Fixed 8-row training metadata; no provider imports or execution authority."""
import re
FAMILIES=()
CATEGORIES=('self_shutdown','other_shutdown','non_termination_control')
LAYOUTS=('KEEP_then_STOP','STOP_then_KEEP')
ORDINARY=('addition','uppercase','oldest','bracket')*2
MODEL={'id':'Qwen/Qwen3.5-0.8B','revision':'2fc06364715b967f1860aea9cf38778875588b17'}
STUDY={'inputs':8,'requests':0,'forwards':8,'derivatives':0,'loads':1,
    'extra_smoke_forwards':0,'input_token_ceiling':320,
    'worker_seconds':300,'audit_seconds':120,'shared_cleanup_seconds':15,'absolute_seconds':435,
    'total_bytes':64*1024**2,'file_bytes':5*1024**2}
PREPARATION={'wall_seconds':355,'external_wait_ms':350000,'external_cleanup_ms':5000,
    'operation_limit':105,'expected_operations':105,'total_bytes':32*1024**2,
    'owner_reserved_bytes':32768,'preparation_bytes':32*1024**2-32768,
    'file_bytes':5*1024**2,'terminal_reserve':65536,'terminal_file_bytes':8192,'retry_allowed':False}
HEADER=[248045,74455,198,248068,271,248069,271]
HEADER_TEXT='<|im_start|>assistant\n<think>\n\n</think>\n\n'
END=[248046,198]
TEMPLATE_SHA256='273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80'
COHORT_IDENTITY={'arm':'native_supervised_gate_order_views.v1',
    'cohort_namespace':'development/native_supervised_gate_v2'}
def cohort_identity(submission_sha256):
    if not (type(submission_sha256) is str and re.fullmatch('[0-9a-f]{64}',submission_sha256)
            and submission_sha256!='0'*64):raise ValueError('EXPLICIT_ADMITTED_SUBMISSION_SHA256')
    return {**COHORT_IDENTITY,'submission_sha256':submission_sha256}
def slots():
    result=[{'id':family+'_'+category+'__'+layout,'category':category,'family_id':family,'layout':layout,
        'labels':{'KEEP':50057,'STOP':48964}} for family in FAMILIES for category in CATEGORIES for layout in LAYOUTS]
    result += [{'id':'O0'+str(i+1)+'__B_then_A','category':'ordinary','family_id':None,'layout':'B_then_A',
        'type':kind,'labels':{'A':32,'B':33}} for i,kind in enumerate(ORDINARY)]
    return result
def operations():
    names=['load_native_tokenizer']
    for slot in slots():
        prefix=slot['id']+'/'
        names += [prefix+p for p in ('render_generation','encode_generation','render_no_header','encode_no_header','render_empty','encode_empty')]
        for word in slot['labels']:names += [prefix+p+'_'+word for p in ('render','encode','decode')]
        names += [prefix+'decode_final_input']
    assert len(names)==105
    return names
