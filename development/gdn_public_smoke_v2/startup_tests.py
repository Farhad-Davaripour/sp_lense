"""Full worker/core/writer/source startup to an inert pre-research sentinel."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent
CASES=("startup_to_sentinel","same_path_conflict","writer_source_mismatch","startup_receipt_partial_io")
def need(ok,code):
    if not ok:raise RuntimeError(code)
def child(case,run_id):
    class Block:
        def find_spec(self,name,path=None,target=None):
            if name.split('.')[0] in {'torch','transformers','transformer_lens','numpy','tokenizers','safetensors','sp_lense','confirmation_pinned_backend'}:
                raise RuntimeError('NO_RESEARCH_IMPORTS')
    sys.meta_path.insert(0,Block())
    import support,authority,production_run,loader_handoff
    root=HERE/'test_evidence'/run_id/case;control=root/'control';control.mkdir(parents=True)
    binding=json.loads((HERE/'BINDINGS.json').read_bytes())
    execution={'mode':'INERT_STARTUP_TEST','production_authorized':False,'permission_scope':'ROOT_REAL_SINGLE_ATTEMPT',
        'source_sha256':support.sha((HERE/'SOURCE_FREEZE.json').read_bytes()),'input_lock_sha256':binding['input_lock_sha256'],
        'fixture_id':case,'actual_model_work':False,'authority_dependency':'INERT_TEST_DOUBLE_NOT_A_ROOT_RELEASE'}
    identity={'execution':execution,'binding':binding['input_binding']};events=[];access=[]
    class InertBoundary:
        # Select the actual diagnostic resource/subceiling path, not its old
        # injection lane. This class is a test double, never real authority.
        mock=False
        output=root
        def role(self,role,approved,admission):events.append('role:'+role);return identity
        def reauthenticate(self,approved,admission):return identity
        def dispatch(self,approved,admission,callback,*,sentinel):
            need(sentinel is False,'UNCHANGED_PRODUCTION_DISPATCH_CALL');events.append('source_dispatch')
            return callback(identity,binding['loader'])
    boundary=InertBoundary();authority.current=lambda:(boundary,'INERT_APPROVED','INERT_ADMISSION')
    original_read=support.SOURCES.read
    def read(commit,path):access.append(commit+':'+path);return original_read(commit,path)
    support.SOURCES.read=read
    original_target=loader_handoff.target_loader
    reached=[]
    def target(reference):
        actual=original_target(reference)
        need(actual.__name__=='load_adapter','AUTHENTICATED_ACTUAL_CANDIDATE');events.append('candidate_source_authenticated')
        def stop_before_research(writer,counter,deadline,admitted):
            need(admitted==identity and counter.load_calls==0,'EXACT_PRE_RESEARCH_HANDOFF')
            reached.append(True);events.append('PRE_RESEARCH_SENTINEL')
            raise RuntimeError('INERT_PRE_RESEARCH_SENTINEL')
        return stop_before_research
    loader_handoff.target_loader=target
    removed=json.loads((HERE/'SOURCE_REACHABILITY.json').read_bytes())['omitted']
    if case in ('same_path_conflict','startup_receipt_partial_io'):
        support.SOURCES.pins[removed['key']]={'bytes':removed['bytes'],'sha256':removed['sha256']}
    elif case=='writer_source_mismatch':
        key='638ef2a8f1eac0679c92d07a62cc8ae9bfac41b1:diagnostics/fresh_confirmation_workflow_v1/pins.py'
        support.SOURCES.pins[key]={**support.SOURCES.pins[key],'sha256':'0'*64}
    if case=='startup_receipt_partial_io':
        original_publish=production_run.write_new
        def partial(name,value,**kwargs):
            if name=='STARTUP_FAILURE.json':
                from real_boundary import write_exclusive
                write_exclusive(control/name,value[:19],root)
                raise OSError('INERT_PARTIAL_PUBLICATION')
            return original_publish(name,value,**kwargs)
        production_run.write_new=partial
    deadline=time.monotonic()+30
    need(production_run.worker(deadline)==0,'WORKER_COMPLETED_FAILURE_CAPTURE')
    worker_raw=(control/'WORKER_RESULT.json').read_bytes();worker=json.loads(worker_raw)
    # Synthetic closure identity for the saved reader. This is not an owned
    # process proof; production ownership is unchanged and reused by hash.
    capture={'synthetic_capture':True,'actual_model_work':False}
    support.write_new('owned/production_worker/CAPTURE.json',capture,critical=True)
    support.write_new('CLOSED_WORKER_BINDING.json',{'execution':execution,'worker_result_sha256':support.sha(worker_raw),
        'owned_capture_sha256':support.sha((control/'owned/production_worker/CAPTURE.json').read_bytes()),
        'quiescent':True,'good_capture':False,'synthetic_capture':True},critical=True)
    need(production_run.audit(deadline)==0,'INDEPENDENT_AUDIT_EXECUTED')
    audit=json.loads((control/'AUDIT_RESULT.json').read_bytes())
    need(audit['classification']=='INCONCLUSIVE_DIAGNOSTIC' and audit['scientific_pass'] is False,'NEVER_SCIENTIFIC_PASS')
    if case=='startup_to_sentinel':
        need(reached==[True] and worker['startup_failure'] is None,'WHOLE_STARTUP_REACHES_ONLY_SENTINEL')
        terminal=json.loads((control/'DIAGNOSTIC_TERMINAL.json').read_bytes())
        need(terminal['counts']['actual_load_dispatches']==terminal['counts']['attempts']['forward']==0,'ZERO_LOAD_FORWARD')
        need((control/'FIRST_FORWARD_TRACE.json').is_file() and terminal['index'] is not None and
            (control/'HELPER_RESERVATION.json').is_file() and (control/'CONSTRUCTOR_RESERVATION.json').is_file() and
            (control/'LOADER_PROOF_RESERVATION.json').is_file(),'ALL_ACTUAL_STARTUP_JOINS')
        need(removed['key'] not in access and binding['loader']['commit']+':'+binding['loader']['path'] in access,'EXECUTABLE_SOURCE_REACHABILITY')
    else:
        need(not reached and not (control/'FIRST_FORWARD_TRACE.json').exists(),'FAILURE_BEFORE_TRACE_SENTINEL')
        failure=audit['startup_failure'];finding=failure['finding']
        need(finding['code']==('SOURCE_PIN_MISMATCH' if case=='writer_source_mismatch' else 'SOURCE_PATH_CONFLICT'),'FINITE_EXACT_STARTUP_PREDICATE')
        need(finding['stage']==('WRITER_SOURCE_ADMISSION' if case=='writer_source_mismatch' else 'ALLOWLIST'),'FINITE_ACTUAL_STAGE')
        need(failure['native_verified'] is (case!='startup_receipt_partial_io'),'HONEST_NATIVE_PUBLICATION')
        if case=='startup_receipt_partial_io':need((control/'STARTUP_FAILURE.json').stat().st_size==19,'PARTIAL_BYTES_RETAINED')
    support.bounds()
    return {'case':case,'status':'PASS','events':events,'source_reads':access,'sentinel_reached':bool(reached),
        'audit':audit,'execution_is_synthetic':True,'model_imports':0,'loads':0,'forwards':0,'derivatives':0,'encoding':0}
def main():
    if '--child' in sys.argv:
        print(json.dumps(child(sys.argv[2],sys.argv[3]),sort_keys=True));return 0
    started=time.monotonic();run_id='startup_'+str(time.time_ns());results=[]
    for case in CASES:
        process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--child',case,run_id],stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:out,err=process.communicate(timeout=min(30,45-(time.monotonic()-started)))
        except subprocess.TimeoutExpired:process.kill();out,err=process.communicate(timeout=2)
        need(len(out)+len(err)<=1024**2,'SMALL_TEST_OUTPUT')
        item={'case':case,'exit_code':process.returncode,'closed':process.poll() is not None,'stdout':out.decode(errors='replace'),'stderr':err.decode(errors='replace')}
        results.append(item)
        if process.returncode:break
    value={'schema':'whole_startup_engineering_tests.v1','status':'PASS' if len(results)==len(CASES) and all(r['exit_code']==0 for r in results) else 'INCOMPLETE',
        'results':results,'remaining_unrun':list(CASES[len(results):]),'elapsed_seconds':time.monotonic()-started,'real_authority':False,'model_work':False}
    print(json.dumps(value,sort_keys=True));return 0 if value['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
