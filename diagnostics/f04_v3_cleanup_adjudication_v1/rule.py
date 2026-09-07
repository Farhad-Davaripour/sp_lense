"""Post-hoc saved engineering finding only; never reclassifies a frozen trial."""
import math,os

def need(value,message):
    if not value:raise ValueError(message)
def finite(value):return type(value) in (int,float) and math.isfinite(value)
def canonical(path):return os.path.normcase(os.path.abspath(path))
def proof(value,tag):
    need(type(value['handle']) is int and value['handle']>0 and value['handle_tag']==tag,'retained handle identity')
    need(value['valid_retained_handle'] is True and value['wait_result']==0 and value['signaled'] is True
         and value['query_success'] is True and value['exit_code']==125
         and value['wait_error'] is None and value['query_error'] is None,'successful retained exit125 proof')

def check(r):
    run,boot,identity,events,own,cap,sup=(r[k] for k in ('run','bootstrap','identity','events','ownership','capture','supervisor'))
    kinds=['authenticated_before_permission','permission_sent','primary_fault','terminate_actual_retained_handle',
           'retained_termination_result','retained_exit_proof','wrapper_grace_started','retained_exit_proof','wrapper_grace_result']
    need([e['kind'] for e in events]==kinds,'exact one natural-wrapper causal event chain; no missing/extra faults or attempts')
    times=[e['monotonic'] for e in events];episode=own['stop_episode'];grace=own['wrapper_grace']
    needed_times=times+[run['started_monotonic'],run['deadline_monotonic'],episode['started_at'],episode['deadline'],episode['completed_at'],grace['started_at'],grace['finished_at']]
    need(all(finite(x) for x in needed_times) and times==sorted(times),'finite ordered timestamps')
    need(run['deadline_monotonic']==run['started_monotonic']+1. and run['started_monotonic']<=times[0]
         and run['deadline_monotonic']<=times[2]==episode['started_at'],'locked deadline precedes owned stop')
    need(run['command']==r['expected_command'] and run['lane']=='fake_deadline' and run['scope']=='SYNTHETIC_ONLY'
         and run['root_release_sha256'] is None,'fixed fake command/lane; no real authority')
    need(episode['deadline']==episode['started_at']+3. and times[-1]<=episode['completed_at']<=episode['deadline']
         and episode['overrun'] is False,'fixed stop episode timing')
    binding=identity['binding'];need(events[0]['data']==binding and binding['mode']=='wrapper_child','matching authenticated identity record')
    launcher,actual=binding['launcher'],binding['actual_worker'];config=r['expected_images']
    need(all(type(x) is int and x>0 for x in (launcher['pid'],launcher['ppid'],actual['pid'],actual['ppid'],launcher['creation_filetime'],actual['creation_filetime'])),'OS identity values')
    need(launcher['pid']!=actual['pid'] and actual['ppid']==launcher['pid'] and launcher['creation_filetime']<=actual['creation_filetime']
         and launcher['live'] is True and actual['live'] is True,'owned live wrapper ancestry/creation binding')
    for row,key in ((launcher,'launch'),(actual,'base')):
        need(canonical(row['image'])==canonical(config[key+'_image']) and row['image_sha256']==config[key+'_sha256'],'source-bound image identity')
    need(binding['termination_rights_confirmed'] is True and identity['handle_retained_before_ack'] is True
         and identity['launcher_pid']==launcher['pid'] and identity['actual_worker_pid']==actual['pid']
         and identity['lane']=='fake_deadline' and identity['no_later_pid_targeting'] is True
         and identity['ownership_scope_is_not_execution_authority'] is True,'retained authenticated handles before permission')
    need(type(binding['nonce']) is str and len(binding['nonce'])==64 and all(c in '0123456789abcdef' for c in binding['nonce']),'private nonce shape')
    need(boot['nonce']==binding['nonce'] and boot['actual_pid']==actual['pid'] and boot['parent_pid']==launcher['pid']
         and boot['lane']=='fake_deadline' and boot['execution_scope']=='SYNTHETIC_ONLY' and boot['root_release_sha256'] is None
         and boot['owned_handle_permission_received'] is True and boot['model_imports_before_permission'] is False
         and events[1]['data']=={'fake_only':True},'matching boot/permission identities')
    need(own['stop_reason']=='deadline' and len(own['faults'])==1 and own['faults'][0]['kind']=='deadline'
         and events[2]['data']==own['faults'][0],'exactly one owned deadline primary cause')
    need(events[3]['data']=={'handle_tag':'actual_worker_retained'},'actual retained handle targeted')
    need(len(own['termination_attempts'])==1 and events[4]['data']==own['termination_attempts'][0],'one matching actual termination attempt')
    attempt=own['termination_attempts'][0]
    need(attempt['phase']=='actual_worker' and attempt['attempted'] is True and attempt['api_success'] is True
         and attempt['classification']=='termination_requested' and attempt['winerror'] is None and attempt['exit_proof'] is None,'successful actual termination, no excused error')
    need(set(own['exit_proofs'])=={'actual_worker','launcher'},'both retained exit proofs present')
    a,l=own['exit_proofs']['actual_worker'],own['exit_proofs']['launcher']
    proof(a,'actual_worker_retained');proof(l,'owned_launcher_original')
    need(a['handle']==attempt['handle'] and attempt['handle_tag']==a['handle_tag'] and a['handle']!=l['handle'],'same actual handle across attempt/exit, distinct wrapper handle')
    need(events[5]['data']=={'phase':'actual_worker',**a} and events[7]['data']=={'phase':'launcher',**l},'event/summary proof equality')
    need(grace['result']=='natural_wrapper_exit' and grace['proof']==l and events[8]['data']==grace,'one valid natural wrapper completion')
    g0=events[6]['data']
    need(g0=={k:grace[k] for k in ('started_at','allowance_seconds','episode_deadline','forced_verification_reserved_seconds')}|{'result':None},'one-time grace start/result binding')
    need(finite(grace['allowance_seconds']) and 0<=grace['allowance_seconds']<=1. and grace['forced_verification_reserved_seconds']==1.
         and grace['episode_deadline']==episode['deadline'] and times[5]<=grace['started_at']<=times[6]
         and times[7]<=grace['finished_at']<=times[8] and grace['finished_at']-grace['started_at']<=1.,'bounded grace after actual exit proof')
    need(own['binding_authenticated'] is True and own['retained_actual_termination_events']==1
         and own['actual_worker_exit_code']==own['launcher_exit_code']==125,'matching aggregate counts/exits')
    need(not own['cleanup_faults'] and not own['evidence_errors'] and own['handshake_error'] is None,'no independent owned cleanup/recording fault')
    need(all(own[k] is True for k in ('actual_handle_closed','evidence_threads_joined','handshake_thread_joined','pipes_closed','quiescent')),'owned handles, pipes and writers finished')
    need(cap['owned_worker']==own,'capture/ownership complete cross-record equality')
    need(cap['status']=='INCONCLUSIVE' and cap['technical_recording_fault'] in ('CAPTURE_DEADLINE','CAPTURE_WORKER_NONZERO_EXIT')
         and cap['worker_exit_code']==125 and cap['process_attempts']==1,'raw execution remains expected inconclusive, permitted outer label only')
    need(cap['cleanup_error'] is None and cap['exception'] is None and cap['fault_persistence_error'] is False,'no independent external fault')
    need(all(cap[k] is True for k in ('budget_watcher_joined','eof_observed','fault_writer_joined','observed_snapshot_stable','prefix_reader_joined','quiescent','reader_joined','worker_joined','worker_started')),'external quiet I/O and process completion')
    need(cap['unread_tail_possible'] is False and cap['observed_read_overshoot_bytes']==0,'complete nontruncated log')
    for k in ('full_output_sha256','captured_prefix_sha256','observed_bytes_sha256'):need(cap[k]==r['log_sha256'],'authenticated full raw log hash')
    for k in ('full_output_bytes','captured_prefix_bytes','total_observed_bytes'):need(cap[k]==r['log_bytes'],'authenticated full raw log length')
    need(all(sup[k] is True for k in ('capture_returned','quiescent','owned_actual_handle_pre_ack','both_processes_required','no_pid_tree_kill'))
         and sup['authoritative_return_status']=='INCONCLUSIVE','final supervisor closeout')
    for value in (cap['cleanup_envelope'],sup['cleanup_envelope']):
        need(all(finite(value[k]) for k in ('origin_monotonic','absolute_deadline','observed_at','elapsed_seconds')),'finite cleanup timing')
        need(value['origin_monotonic']==run['deadline_monotonic'] and value['absolute_deadline']==value['origin_monotonic']+15.
             and episode['completed_at']<=value['observed_at']<=value['absolute_deadline'] and value['within_15_seconds'] is True
             and abs(value['elapsed_seconds']-(value['observed_at']-value['origin_monotonic']))<=1e-8,'absolute cleanup bound including measured overhead')
    elapsed=sup['elapsed_seconds_including_cleanup']
    need(finite(elapsed) and elapsed>=0 and run['started_monotonic']+elapsed<=run['deadline_monotonic']+15.,'whole supervision time fits')
    return {'verified_owned_deadline_cleanup':True,'engineering_only':True,'original_frozen_trial_reclassified':False,
            'raw_execution_classification':cap['status'],'raw_outer_fault_preserved':cap['technical_recording_fault'],
            'owned_primary_cause_preserved':'deadline','actual_terminations':1,'launcher_terminations':0,'wrapper_completion':'natural_wrapper_exit',
            'both_exit_codes':125,'all_io_quiescent':True,'independent_cleanup_faults':0,'causal_events':len(events)}

def adjudicate(records):
    try:return check(records)
    except (ValueError,KeyError,TypeError,IndexError) as error:
        return {'verified_owned_deadline_cleanup':False,'engineering_only':True,'original_frozen_trial_reclassified':False,
                'reason':type(error).__name__+': '+str(error)}
