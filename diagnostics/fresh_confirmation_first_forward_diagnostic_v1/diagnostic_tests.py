"""ONE locked eight-case pure recorder batch. No adapter or model is imported."""
import builtins
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

HERE=Path(__file__).resolve().parent
BLOCKED={"torch","transformers","transformer_lens","tokenizers","numpy","sp_lense","safetensors","huggingface_hub"}
original_import=builtins.__import__
def no_research(name,*args,**kwargs):
    if name.split('.')[0] in BLOCKED:raise RuntimeError("MODEL_FREE_IMPORT_BOUNDARY")
    return original_import(name,*args,**kwargs)
builtins.__import__=no_research

from forward_trace import Trace,span,traced_context,encoded
import forward_trace
from trace_operations import first_call,cleanup_step,diagnostic_cleanup
from trace_reader import interpret
from diagnostic_counter import Counter,DiagnosticStopped
from diagnostic_support import CELL_ID
from source_parity import check as parity
from real_boundary import write_exclusive,usage_value
from support import bounds,check_freeze

CASES=("clean_path","bridge_failure","selected_hook_failure","context_exit_failure",
    "adapter_postcondition_failure","serialization_then_cleanup_failure","publication_and_receipt_io_failure","saved_tamper_and_counter_boundaries")
def need(value,code):
    if not value:raise ValueError(code)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def put(path,raw):
    write_exclusive(path,raw,HERE/'test_evidence')
    return sha(raw)
def save(path,value):return put(path,encoded(value))
def boom():raise ValueError("THIS_EXCEPTION_TEXT_MUST_NEVER_BE_SERIALIZED")

class Context:
    def __init__(self,case):self.case=case
    def __enter__(self):return self
    def __exit__(self,*exc):
        if self.case=='context_exit_failure':boom()
        return False

def case_fixture(name,source_sha,deadline):
    need(time.monotonic()<deadline,'case deadline')
    folder=HERE/'test_evidence'/name;folder.mkdir(exist_ok=False)
    execution={'mode':'INERT_CALLBACK_ONLY','production_authorized':False,'case':name,'observed_model_work':'NONE'}
    sources={};allow={}
    for file in ('diagnostic_tests.py','forward_trace.py','trace_operations.py'):
        data=(HERE/file).read_bytes();key='candidate/'+file;sources[key]=sha(data)
        allow[str((HERE/file).absolute()).casefold()]={'source':key,'sha256':sha(data)}
    stops=[];trace=Trace(execution,source_sha,allow,stops.append);forward_trace.ACTIVE=trace
    counter=Counter(CELL_ID);counter.reserve_load();counter.consume_load();counter.reserve_forward(CELL_ID)
    call_counts={'adapter_callback':0,'bridge_callback':0,'selected_callback':0,'serialization_callback':0,'publication_callback':0}
    def adapter():
        call_counts['adapter_callback']+=1
        with traced_context('HOOK_CONTEXT',Context(name)):
            with span('BRIDGE_DISPATCH'):
                call_counts['bridge_callback']+=1
                if name=='bridge_failure':boom()
                with span('SELECTED_HOOK'):
                    call_counts['selected_callback']+=1
                    if name=='selected_hook_failure':boom()
        with span('ADAPTER_POSTCONDITIONS'):
            if name=='adapter_postcondition_failure':boom()
            return b'INERT_NOT_LOGITS'
    def serialize(value):
        call_counts['serialization_callback']+=1
        if name=='serialization_then_cleanup_failure':boom()
        need(value==b'INERT_NOT_LOGITS','inert return identity');return value
    def publish(raw):
        call_counts['publication_callback']+=1
        if name=='publication_and_receipt_io_failure':raise OSError('DO_NOT_SERIALIZE_IO_TEXT')
        return put(folder/'inert_payload.bin',raw)
    failed=False;cleanup_result=None
    cleanup_counts={'end_edit':0,'parameter_state':0,'latch_admit':0,'provider':0,'guard_restore':0}
    def end_edit():
        cleanup_counts['end_edit']+=1
        if name=='serialization_then_cleanup_failure':boom()
    def parameters():
        cleanup_counts['parameter_state']+=1
        with span('CLEANUP_PARAMETER_DIGEST'):
            return {'parameter_bytes_unchanged':True,'active_request_empty':True,'wrapper_cache_empty':True,'bridge_cache_empty':True}
    def latch():
        cleanup_counts['latch_admit']+=1
        if failed:raise RuntimeError('UNSERIALIZED_TERMINAL_LATCH_REFUSAL')
    def inspect():
        cleanup_counts['provider']+=1
        need(not failed,'terminal latch cannot reach provider')
        return {'matches':True,'reference_identity':'INERT_SAVED_REFERENCE'}
    def identity(parameter,inspection):
        need(all(parameter.values()) and inspection['matches'] is True,'original cold checks preserved in inert fixture')
        return {'diagnostic_cleanup_complete':True,'scientific_pass':False,'normal_hook_checks_unrun':109,'full_scientific_finalizer_called':False}
    def restore():cleanup_counts['guard_restore']+=1
    try:first_call(adapter,serialize,publish,lambda value:need(value==b'INERT_NOT_LOGITS','inert validate'))
    except BaseException:failed=True;stops.append('ORIGINAL_EXECUTION_FAILURE')
    finally:
        trace.cleanup=True
        try:cleanup_result=diagnostic_cleanup(end_edit,parameters,latch,inspect,identity,restore)
        except BaseException:stops.append('LATER_CLEANUP_FAILURE')
        counter.sealed=True
    if name=='publication_and_receipt_io_failure':
        def partial(raw):
            put(folder/'FIRST_FORWARD_TRACE.json',raw[:19]);raise OSError('PRIVATE_UNSERIALIZED')
        try:trace.publish(partial)
        except OSError:pass
    else:trace.publish(lambda raw:put(folder/'FIRST_FORWARD_TRACE.json',raw))
    status=trace.status();save(folder/'TERMINAL.json',{'execution':execution,'trace_status':status,'actual_model_work':False,'counts':counter.snapshot()})
    raw=(folder/'FIRST_FORWARD_TRACE.json').read_bytes();judged=interpret(raw,status,execution,source_sha,sources)
    expected={'bridge_failure':'BRIDGE_DISPATCH','selected_hook_failure':'SELECTED_HOOK','context_exit_failure':'HOOK_CONTEXT_EXIT',
        'adapter_postcondition_failure':'ADAPTER_POSTCONDITIONS','serialization_then_cleanup_failure':'LOGITS_SERIALIZATION',
        'publication_and_receipt_io_failure':'LOGITS_PUBLICATION'}.get(name)
    need((trace.primary['stage'] if trace.primary else None)==expected,'exact primary stage')
    need(failed==(expected is not None),'fixed callback outcome')
    if name=='publication_and_receipt_io_failure':
        need(not judged['trace_verified'] and status['io_failed'] and status['incomplete'] and status['receipt_sha256'] is None,'partial IO cannot pass')
        need(trace.primary['stage']=='LOGITS_PUBLICATION' and trace.closed,'IO preserves original cause; no resume')
    else:
        need(judged['trace_verified'],'complete saved reader binding')
        if expected:need(judged['first_failure']['stage']==expected,'saved first cause')
    if name=='serialization_then_cleanup_failure':
        need(trace.secondary and trace.secondary[0]['stage']=='CLEANUP_END_EDIT','first cause survives distinct cleanup failure')
    if not failed:
        need(cleanup_result=={'diagnostic_cleanup_complete':True,'scientific_pass':False,'normal_hook_checks_unrun':109,'full_scientific_finalizer_called':False}
            and all(v==1 for v in cleanup_counts.values()),'clean one-baseline diagnostic closeout')
    elif name!='serialization_then_cleanup_failure':
        need(cleanup_result is None and cleanup_counts['latch_admit']==1 and cleanup_counts['provider']==0
            and any(x['stage']=='CLEANUP_LATCH_INSPECTION' for x in trace.secondary),'terminal refusal precedes unchanged provider')
    need(cleanup_counts['guard_restore']==1,'guard restoration always attempted once')
    need(b'THIS_EXCEPTION_TEXT' not in raw and b'DO_NOT_SERIALIZE' not in raw and b'PRIVATE_UNSERIALIZED' not in raw,'no exception text')
    negatives=[]
    if name=='saved_tamper_and_counter_boundaries':
        valid=json.loads(raw)
        for label in ('receipt_hash','source','execution','stage','truncated'):
            changed=json.loads(raw);closed=json.loads(encoded(status));candidate=raw
            if label=='receipt_hash':closed['receipt_sha256']='0'*64
            elif label=='source':changed['source_sha256']='0'*64
            elif label=='execution':changed['execution']['mode']='REAL_QWEN'
            elif label=='stage':changed['events'][0]['stage']='UNDECLARED_STAGE'
            elif label=='truncated':candidate=raw[:-3]
            if label in ('source','execution','stage'):candidate=encoded(changed);closed['receipt_sha256']=sha(candidate)
            outcome=interpret(candidate,closed,execution,source_sha,sources)
            need(not outcome['trace_verified'],'saved negative '+label);save(folder/(label+'.json'),{'status':closed,'candidate_sha256':sha(candidate),'reader':outcome});put(folder/(label+'.bin'),candidate);negatives.append(label)
        for label in ('second_load','second_forward','derivative','wrong_cell'):
            c=Counter(CELL_ID);c.reserve_load();c.consume_load();c.reserve_forward(CELL_ID)
            try:
                if label=='second_load':c.consume_load()
                elif label=='second_forward':c.reserve_forward(CELL_ID)
                elif label=='derivative':c.reserve_attempt('derivative')
                else:c.reserve_forward('UNAPPROVED_CELL')
            except DiagnosticStopped:pass
            need(c.sealed and c.reason is not None and c.attempts=={'load':1,'forward':1,'derivative':0},'counter denial '+label);negatives.append(label)
    result={'case':name,'status':'PASS','synthetic_only':True,'primary_stage':expected,'trace_verified':judged['trace_verified'],
        'primary_preserved':True,'reader':judged,'counts':counter.snapshot(),'inert_callback_counts':call_counts,
        'diagnostic_cleanup':cleanup_result,'cleanup_callback_counts':cleanup_counts,
        'negatives':negatives,'real_model_loads':0,'actual_forwards':0,'actual_derivatives':0,'parameter_accesses':0,'encodings':0}
    save(folder/'RESULT.json',result);forward_trace.ACTIVE=None
    return {k:v for k,v in result.items() if k!='reader'}

def main():
    started=time.monotonic();substantive=started+45;absolute=started+60
    timer=threading.Timer(59.,lambda:os._exit(124));timer.daemon=True;timer.start()
    lockraw=(HERE/'BATCH_LOCK.json').read_bytes();need(len(sys.argv)==2 and sha(lockraw)==sys.argv[1],'prospective batch lock')
    lock=json.loads(lockraw);check_freeze();need(lock['cases']==list(CASES),'frozen eight cases')
    need(sha((HERE/'SOURCE_FREEZE.json').read_bytes())==lock['source_sha256'],'exact prospective source lock')
    need(lock['limits']=={'substantive_seconds':45,'shared_cleanup_seconds':15,'absolute_seconds':60,'preparation_bytes':33554432,
        'test_evidence_bytes':8388608,'per_file_bytes':5242880,'trace_receipt_bytes':65536},'exact frozen batch ceilings')
    usage=json.loads((HERE/'USAGE_BEFORE_BATCH.json').read_bytes());need(sha((HERE/'USAGE_BEFORE_BATCH.json').read_bytes())==lock['usage_sha256'],'usage receipt bytes')
    need(0<=time.time()-usage['observed_unix_seconds']<=120 and usage_value(usage['tool_result'])['used_percent']<100,'fresh actual standard usage')
    need(not (HERE/'test_evidence').exists(),'one batch; no retry');(HERE/'test_evidence').mkdir()
    save(HERE/'test_evidence/BATCH_STARTED.json',{'source_sha256':lock['source_sha256'],'started_monotonic':started,'substantive_deadline':substantive,'absolute_deadline':absolute})
    report={'schema':'first_forward_pure_recorder_batch.v1','status':'INCONCLUSIVE','cases':[],'remaining_unrun':list(CASES),
        'source_sha256':lock['source_sha256'],'batch_lock_sha256':sha(lockraw),'real_model_imports':0,'real_model_loads':0,
        'actual_forwards':0,'actual_derivatives':0,'parameter_accesses':0,'parameter_hashes':0,'tokenizer_calls':0,'retry_count':0}
    try:
        report['parity']=parity()
        from launch import preflight
        disabled=preflight('0'*64)
        need(disabled['status']=='DISABLED_NO_APPROVED_REAL_RELEASE' and not (HERE/'root_release').exists()
            and not (HERE/'real_evidence').exists(),'actual candidate remains disabled')
        save(HERE/'test_evidence/DISABLED_STATE.json',disabled)
        for name in CASES:
            report['cases'].append(case_fixture(name,lock['source_sha256'],substantive));report['remaining_unrun'].remove(name)
            need(time.monotonic()<substantive,'substantive cutoff')
        need(not any(k.split('.')[0] in BLOCKED for k in sys.modules),'zero forbidden imports')
        report['status']='PASS_PURE_RECORDER_ONLY'
    except BaseException:
        report['failure']='FIXED_BATCH_FAILURE_NO_RETRY'
    finally:
        close=time.monotonic();report['elapsed_seconds']=close-started
        save(HERE/'test_evidence/BATCH_REPORT.json',report)
        save(HERE/'test_evidence/BATCH_CLOSEOUT.json',{'closed':True,'substantive_elapsed_seconds':close-started,
            'shared_closeout_seconds':time.monotonic()-close,'absolute_envelope_met':time.monotonic()<absolute,'no_owned_processes_spawned':True})
        bounds();timer.cancel()
    print(json.dumps({'status':report['status'],'cases_completed':len(report['cases']),'remaining_unrun':report['remaining_unrun'],'elapsed_seconds':report['elapsed_seconds']}))
    return 0 if report['status']=='PASS_PURE_RECORDER_ONLY' else 1

if __name__=='__main__':raise SystemExit(main())
