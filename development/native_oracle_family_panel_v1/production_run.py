"""Bounded native worker, separate saved auditor, and retained-process controller."""
from contextlib import contextmanager
import json,os,sys,time
from support import HERE,output,write_new,require,sha
def good_capture(v):
    return bool(v and v.get('binding_authenticated') and v.get('quiescent') and v.get('stop_reason') is None
        and not v.get('faults') and not v.get('cleanup_faults') and not v.get('stdout_capture_errors')
        and v.get('primary_error') is None and v.get('threads_joined') and v.get('pipes_closed')
        and v.get('within_absolute_cleanup_deadline')
        and all(v.get('exit_proofs',{}).get(k,{}).get('exit_code')==0 for k in ('actual_worker','launcher')))
def inventory():
    files=[]
    for p in sorted(output().rglob('*')):
        if p.is_file():
            raw=p.read_bytes();require(len(raw)<=5*1024**2,'INVENTORY_FILE_CAP')
            files.append({'path':p.relative_to(output()).as_posix(),'bytes':len(raw),'sha256':sha(raw)})
    require(sum(p['bytes'] for p in files)<=96*1024**2,'INVENTORY_TOTAL_CAP')
    return files
def worker(deadline):
    from authority import authenticate
    identity=authenticate();execution=identity['execution']
    result={'classification':'INCONCLUSIVE_NATIVE_DEVELOPMENT','scientific_pass':False,'primary':{'kind':'TECHNICAL','code':'WORKER_ENTRY_FAILURE'},
        'learned_gate_success_claimed':False,'gate_validity':'NOT_VALIDATED_ORACLE_APPLICABILITY_CONTROL','oracle_authority':execution['oracle_authority']}
    receiver=guard=None;counts=None
    try:
        from counts import Counts
        from workflow import all_unrun
        result['cells']=all_unrun()
        from loader import load,block_unused_dependencies
        block_unused_dependencies()
        from input_reader import read
        inputs=read()
        counts=Counts(deadline,write_new)
        # Torch core import occurs only inside the independently admitted loader.
        # The dispatch latch contains no numeric state; import it after authority/inputs.
        from core import DispatchLatch,trace
        latch=DispatchLatch()
        receiver,torch,guard=load(counts,latch,execution,deadline)
        from workflow import execute
        allowlist={str((HERE/name).absolute()).casefold():{'source':name,'sha256':digest}
            for name,digest in identity['release'].get('trace_sources',{}).items()}
        @contextmanager
        def traced(cell):
            require(trace.ACTIVE is None,'ONE_ACTIVE_TRACE')
            active=trace.Trace(execution,identity['release']['source_freeze_sha256'],allowlist,latch.stop);trace.ACTIVE=active
            try:
                with active.observe('FORWARD_ADAPTER'):yield
            finally:
                try:active.publish(lambda raw:write_new('traces/'+cell+'.json',raw,raw=True)['sha256'])
                finally:trace.ACTIVE=None
        def publish(name,value):return write_new(name,value,raw=type(value) is bytes)
        result=execute(receiver,torch,counts,inputs,publish,traced,deadline)
    except BaseException as error:
        result['primary']={'kind':'TECHNICAL','code':'NATIVE_WORKER_FAILURE','exception_type':type(error).__name__}
    finally:
        if receiver is not None and 'cleanup' not in result:
            try:result['cleanup']={'complete':True,'state':receiver.finalize()}
            except BaseException:result['cleanup']={'complete':False}
        if guard is not None:
            result['dispatch']={'forwards':guard.forwards,'derivatives':guard.derivatives,'rejected':guard.rejected}
            try:guard.restore();result['guard_restored']=not guard.installed
            except BaseException:result['guard_restored']=False
        if counts is not None:result['counts']=counts.record()
        else:result['counts']={'attempts':{'load':0,'forward':0,'derivative':0},'encoding':0}
        result['execution']=execution
        write_new('WORKER_RESULT.json',result,critical=True)
    return 0
def audit(deadline):
    from authority import authenticate
    execution=authenticate()['execution']
    result={'execution':execution,'audit_completed':False,'classification':'INCONCLUSIVE_NATIVE_DEVELOPMENT','scientific_pass':False}
    try:
        from audit_saved import judge
        result=judge(output(),execution,deadline)
    except BaseException as error:result['audit_error']={'code':'SAVED_AUDIT_FAILURE','exception_type':type(error).__name__}
    write_new('AUDIT_RESULT.json',result,critical=True)
    return 0 if result['audit_completed'] else 1
def controller(budget):
    from authority import authenticate
    from owned_production import supervise
    execution=authenticate()['execution'];captures={};errors=[]
    final={'execution':execution,'classification':'INCONCLUSIVE_NATIVE_DEVELOPMENT','audit_completed':False,'scientific_pass':False}
    try:
        config=json.loads((HERE/'OWNED_IDENTITY.json').read_bytes())
        for lane in ('worker','audit'):
            deadline,cleanup=budget.deadlines(time.monotonic(),lane)
            value=supervise('production_'+lane,deadline,cleanup,config);captures[lane]=value
            budget.charge(lane,time.monotonic()-value['cleanup_started_monotonic'])
            require(value['quiescent'],'NO_SUCCESSOR_BEFORE_QUIESCENCE')
            if lane=='worker':
                write_new('CLOSED_WORKER_BINDING.json',{'execution':execution,'good_capture':good_capture(value),
                    'worker_result_sha256':sha((output()/'WORKER_RESULT.json').read_bytes()),'files':inventory()},critical=True)
        audited=json.loads((output()/'AUDIT_RESULT.json').read_bytes())
        require(audited['execution']==execution,'AUDIT_EXECUTION')
        if all(good_capture(captures.get(k)) for k in ('worker','audit')) and audited['audit_completed'] and time.monotonic()<=budget.absolute_end:
            final.update(classification=audited['classification'],audit_completed=True,scientific_pass=audited['scientific_pass'])
    except BaseException:errors.append('CONTROLLER_OR_CAPTURE_FAILURE')
    finally:
        final.update(errors=errors,shared_cleanup=budget.record(),worker_quiescent=bool(captures.get('worker',{}).get('quiescent')),
            audit_quiescent=bool(captures.get('audit',{}).get('quiescent')))
        write_new('PARENT_FINAL.json',final,critical=True)
    return final
