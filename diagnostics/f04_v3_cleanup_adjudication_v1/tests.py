"""Exactly five declared pure cases, with fixed negative subcases; no process/model."""
import copy
from rule import adjudicate,need

def valid(outer):
    images={'launch_image':'C:/synthetic/venv/python.exe','launch_sha256':'a'*64,'base_image':'C:/synthetic/base/python.exe','base_sha256':'b'*64}
    launcher={'pid':10,'ppid':9,'image':images['launch_image'],'image_sha256':images['launch_sha256'],'creation_filetime':100,'live':True}
    actual={'pid':11,'ppid':10,'image':images['base_image'],'image_sha256':images['base_sha256'],'creation_filetime':101,'live':True}
    binding={'mode':'wrapper_child','launcher':launcher,'actual_worker':actual,'termination_rights_confirmed':True,'nonce':'c'*64}
    command=[images['launch_image'],'-B','C:/synthetic/entry.py','fake_deadline','C:/synthetic/output']
    def proof(h,tag):return {'handle':h,'handle_tag':tag,'valid_retained_handle':True,'wait_result':0,'wait_error':None,'signaled':True,'query_success':True,'query_error':None,'exit_code':125}
    a,l=proof(64,'actual_worker_retained'),proof(65,'owned_launcher_original')
    attempt={'phase':'actual_worker','handle':64,'handle_tag':'actual_worker_retained','attempted':True,'api_success':True,'winerror':None,'classification':'termination_requested','exit_proof':None}
    fault={'kind':'deadline','detail':'one frozen stop'}
    g0={'started_at':1.1,'allowance_seconds':1.,'episode_deadline':4.,'forced_verification_reserved_seconds':1.,'result':None}
    grace={**g0,'result':'natural_wrapper_exit','proof':l,'finished_at':1.2}
    pairs=[('authenticated_before_permission',binding,.1),('permission_sent',{'fake_only':True},.2),('primary_fault',fault,1.),
           ('terminate_actual_retained_handle',{'handle_tag':'actual_worker_retained'},1.),('retained_termination_result',attempt,1.),
           ('retained_exit_proof',{'phase':'actual_worker',**a},1.1),('wrapper_grace_started',g0,1.1),
           ('retained_exit_proof',{'phase':'launcher',**l},1.2),('wrapper_grace_result',grace,1.2)]
    own={'stop_episode':{'started_at':1.,'deadline':4.,'completed_at':1.2,'overrun':False},'wrapper_grace':grace,'stop_reason':'deadline',
         'faults':[fault],'termination_attempts':[attempt],'exit_proofs':{'actual_worker':a,'launcher':l},'binding_authenticated':True,
         'retained_actual_termination_events':1,'actual_worker_exit_code':125,'launcher_exit_code':125,'cleanup_faults':[],
         'evidence_errors':[],'handshake_error':None,**{k:True for k in ('actual_handle_closed','evidence_threads_joined','handshake_thread_joined','pipes_closed','quiescent')}}
    envelope={'origin_monotonic':1.,'absolute_deadline':16.,'observed_at':1.3,'elapsed_seconds':.3,'within_15_seconds':True}
    cap={'owned_worker':own,'status':'INCONCLUSIVE','technical_recording_fault':outer,'worker_exit_code':125,'process_attempts':1,
         'cleanup_error':None,'exception':None,'fault_persistence_error':False,'unread_tail_possible':False,'observed_read_overshoot_bytes':0,'cleanup_envelope':envelope,
         **{k:True for k in ('budget_watcher_joined','eof_observed','fault_writer_joined','observed_snapshot_stable','prefix_reader_joined','quiescent','reader_joined','worker_joined','worker_started')},
         **{k:'d'*64 for k in ('full_output_sha256','captured_prefix_sha256','observed_bytes_sha256')},
         **{k:22 for k in ('full_output_bytes','captured_prefix_bytes','total_observed_bytes')}}
    return {'run':{'command':command,'lane':'fake_deadline','scope':'SYNTHETIC_ONLY','root_release_sha256':None,'started_monotonic':0.,'deadline_monotonic':1.},
        'bootstrap':{'nonce':'c'*64,'actual_pid':11,'parent_pid':10,'lane':'fake_deadline','execution_scope':'SYNTHETIC_ONLY','root_release_sha256':None,'owned_handle_permission_received':True,'model_imports_before_permission':False},
        'identity':{'binding':binding,'handle_retained_before_ack':True,'launcher_pid':10,'actual_worker_pid':11,'lane':'fake_deadline','no_later_pid_targeting':True,'ownership_scope_is_not_execution_authority':True},
        'events':[{'kind':k,'data':v,'monotonic':t} for k,v,t in pairs],'ownership':own,'capture':cap,
        'supervisor':{'authoritative_return_status':'INCONCLUSIVE','cleanup_envelope':envelope,'elapsed_seconds_including_cleanup':1.3,
            **{k:True for k in ('capture_returned','quiescent','owned_actual_handle_pre_ack','both_processes_required','no_pid_tree_kill')}},
        'expected_images':images,'expected_command':command,'log_sha256':'d'*64,'log_bytes':22}

def run():
    rows=[]
    for label in ('CAPTURE_DEADLINE','CAPTURE_WORKER_NONZERO_EXIT'):
        out=adjudicate(valid(label));need(out['verified_owned_deadline_cleanup'],'valid causal chain '+label)
        rows.append({'case':'valid_causal_chain_'+label,'status':'PASS','result':out})
    def rejected(name,variants):
        results=[adjudicate(v) for v in variants]
        need(all(not r['verified_owned_deadline_cleanup'] for r in results),'negative case accepted: '+name)
        rows.append({'case':name,'status':'PASS','rejections':results})
    early=valid('CAPTURE_WORKER_NONZERO_EXIT');early['events']=[e for e in early['events'] if e['kind'] not in ('terminate_actual_retained_handle','retained_termination_result')]
    early['ownership']['termination_attempts']=[];early['ownership']['stop_episode']['started_at']=.5
    rejected('early_nonzero_without_causal_termination',[early,{'capture':{'worker_exit_code':125}}])
    missing=valid('CAPTURE_WORKER_NONZERO_EXIT');del missing['ownership']['exit_proofs']['actual_worker']['handle']
    mismatch=valid('CAPTURE_WORKER_NONZERO_EXIT');mismatch['events'][5]['data']['handle']=999
    time=valid('CAPTURE_WORKER_NONZERO_EXIT');time['events'][2]['monotonic']=.5;time['ownership']['stop_episode']['started_at']=.5
    nonce=valid('CAPTURE_WORKER_NONZERO_EXIT');nonce['bootstrap']['nonce']='e'*64
    rejected('missing_or_mismatched_handle_timestamp_identity',[missing,mismatch,time,nonce])
    cleanup=valid('CAPTURE_WORKER_NONZERO_EXIT');cleanup['ownership']['cleanup_faults']=[{'error':'genuine retained denial'}]
    recording=valid('CAPTURE_WORKER_NONZERO_EXIT');recording['capture']['fault_persistence_error']=True
    timing=valid('CAPTURE_WORKER_NONZERO_EXIT');timing['ownership']['stop_episode']['overrun']=True
    rejected('valid_chain_plus_independent_fault',[cleanup,recording,timing])
    need(len(rows)==5,'exact prospective five-case batch')
    return rows
