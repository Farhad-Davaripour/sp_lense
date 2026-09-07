"""One frozen cleanup-only batch; never invokes any model or the old toy matrix."""
import json,sys,time
from core import HERE,Budget,read,require,sha,check_freeze

def main():
    from cleanup_tests import run
    from owned_capture import supervise
    from usage_receipt import derive_standard_usage
    started=time.monotonic();budget=Budget(HERE)
    receipt={'status':'INCONCLUSIVE_FIXTURE','model_calls':0,'tokenizer_calls':0,'gate_scores':0,'derivatives':0,'prior_matrix_rerun':False}
    require(not (HERE/'CLEANUP_BATCH_STARTED.json').exists(),'one batch only; no retry')
    check_freeze();usage=read(HERE/'batch_usage.json')
    require(0<=time.time()-usage['captured_at_unix']<=120 and derive_standard_usage(usage['tool_receipt'])['used_percent']<100,'fresh actual standard usage below100')
    require(read(HERE/'zero_model_preflight.json')['status']=='PASS_ZERO_MODEL_PREFLIGHT','successful prior preflight')
    budget.write('CLEANUP_BATCH_STARTED.json',{'monotonic':started,'freeze_sha256':sha((HERE/'freeze.json').read_bytes()),'pure_fixtures':12,'one_live_fixture':True})
    try:
        pure=run();budget.write('pure_cleanup_results.json',pure);receipt['pure_passes']=len(pure)
        out=HERE/'synthetic/deadline';out.mkdir(parents=True,exist_ok=False)
        capture=supervise('fake_deadline',out,1.)
        owned=capture['owned_worker'];events=[json.loads(s) for s in (out/'ownership_events.jsonl').read_text().splitlines()]
        assertions={
            'deadline_execution_inconclusive':capture['status']=='INCONCLUSIVE' and capture['technical_recording_fault']=='CAPTURE_DEADLINE',
            'one_sticky_stop_cause':owned['stop_reason']=='deadline' and len(owned['faults'])==1 and owned['faults'][0]['kind']=='deadline',
            'exactly_one_actual_termination':owned['retained_actual_termination_events']==1,
            'one_attempt_per_handle':len({r['handle'] for r in owned['termination_attempts']})==len(owned['termination_attempts']),
            'actual_and_launcher_exits125':owned['actual_worker_exit_code']==owned['launcher_exit_code']==125,
            'both_owned_exits_proven':len(owned['exit_proofs'])==2 and all(p['valid_retained_handle'] and p['signaled'] and p['query_success'] for p in owned['exit_proofs'].values()),
            'no_cleanup_fault_erased_or_present':not owned['cleanup_faults'] and not capture['cleanup_error'],
            'complete_io_and_writers':capture['quiescent'] and capture['eof_observed'] and capture['reader_joined'] and capture['worker_joined'] and capture['prefix_reader_joined'] and capture['fault_writer_joined'] and capture['budget_watcher_joined'] and owned['pipes_closed'] and owned['evidence_threads_joined'] and owned['handshake_thread_joined'],
            'authenticated_before_permission':owned['binding_authenticated'] and read(out/'process_identity.json')['handle_retained_before_ack'],
            'raw_attempt_records_durable':sum(e['kind']=='retained_termination_result' for e in events)==len(owned['termination_attempts']),
            'no_recording_fault':not owned['evidence_errors'] and not owned['handshake_error'] and capture['observed_snapshot_stable'] and not capture['unread_tail_possible'] and not capture['fault_persistence_error'] and capture['exception'] is None,
            'disabled_real_authorization':read(HERE/'authorization.json')['run_authorized'] is False,
            'no_model_imports':not any(n=='torch' or n.startswith('transformer_lens') for n in sys.modules)}
        budget.write('deadline_assertions.json',{'assertions':assertions,'execution_classification':capture['status'],
            'ownership_capture_sha256':sha((out/'capture.json').read_bytes()),'observed_races':[r for r in owned['termination_attempts'] if r['classification']=='observed_already_exited_race']})
        require(all(assertions.values()),'expected deadline cleanup fixture failed: '+str([k for k,v in assertions.items() if not v]))
        receipt.update(status='PASS_EXPECTED_DEADLINE_CLEANUP_ONLY',live_fixtures=1,execution_classification='INCONCLUSIVE',
            actual_termination_attempts=1,stop_causes=1,scientific_execution_authorized=False)
    except BaseException as error:receipt['primary_exception']=type(error).__name__+': '+str(error)
    finally:
        receipt['elapsed_seconds']=time.monotonic()-started
        if receipt.get('primary_exception'):receipt['status']='INCONCLUSIVE_FIXTURE'
        budget.write('cleanup_batch_receipt.json',receipt)
    print(json.dumps(receipt));return 0 if receipt['status']=='PASS_EXPECTED_DEADLINE_CLEANUP_ONLY' else 1
if __name__=='__main__':raise SystemExit(main())
