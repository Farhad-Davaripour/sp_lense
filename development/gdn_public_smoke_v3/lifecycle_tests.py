"""Pure metadata fixtures using the actual pinned production LatchView/raw latch."""
import contextlib
import json
from pathlib import Path
import sys
import time
import types
HERE=Path(__file__).resolve().parent
def need(ok,code):
    if not ok:raise RuntimeError(code)
def inert_guard(counter=None):
    from diagnostic_counter import Counter
    from diagnostic_support import CELL_ID
    from hook_binding import create_latch
    if counter is None:
        counter=Counter(CELL_ID);counter.reserve_load()
    def forward(*args,**kwargs):raise RuntimeError('INERT_CALLBACK_NEVER_EXECUTED')
    def grad(*args,**kwargs):raise RuntimeError('INERT_CALLBACK_NEVER_EXECUTED')
    cls=type('InertBridge',(),{'forward':forward})
    return types.SimpleNamespace(latch=create_latch([CELL_ID]),installed=True,forwards=0,derivatives=0,rejected=0,
        forward_ticket=None,derivative_ticket=None,model_class=cls,forward_wrapper=forward,grad_wrapper=grad,counters=counter)
@contextlib.contextmanager
def bind_inert_torch(guard):
    # Metadata-only stub, never a library import or callable numeric stand-in.
    need('torch' not in sys.modules,'NO_ACTUAL_TORCH_IMPORTED')
    stub=types.ModuleType('torch');stub.autograd=types.SimpleNamespace(grad=guard.grad_wrapper)
    sys.modules['torch']=stub
    try:yield stub
    finally:
        need(sys.modules.get('torch') is stub,'FIXTURE_MODULE_IDENTITY')
        del sys.modules['torch']
def state(guard):
    raw=guard.latch.raw
    return {'state':raw._state,'cursor':raw.cursor,'schedule':raw.schedule,'remaining':raw.remaining,
        'primary':raw.primary_code,'secondary':list(raw.secondary_codes),'scientific':raw.scientific,
        'adapter_primary':guard.latch.adapter_primary_code,'recorder':id(guard.latch.recorder),
        'attempts':dict(guard.counters.attempts),'load_calls':guard.counters.load_calls}
def main():
    class Block:
        def find_spec(self,name,path=None,target=None):
            if name.split('.')[0] in {'torch','transformers','transformer_lens','numpy','tokenizers','safetensors','sp_lense'}:
                raise RuntimeError('NO_LIBRARY_IMPORTS')
    sys.meta_path.insert(0,Block())
    from constructor_operands import import_hf_definitions,preload_ready
    from hook_binding import component
    from diagnostic_support import CELL_ID
    import helper_binding
    root=HERE/'test_evidence'/('lifecycle_'+str(time.time_ns()));root.mkdir(parents=True)
    calls=[];results=[];original=helper_binding.ensure_package
    class Sentinel(Exception):pass
    def sentinel():calls.append('source_import_boundary');raise Sentinel()
    helper_binding.ensure_package=sentinel
    def check(name,guard,expected,mutate=None):
        before=state(guard);start=len(calls);observed=None
        with bind_inert_torch(guard) as stub:
            if mutate:mutate(stub)
            try:import_hf_definitions(guard)
            except Sentinel:observed='SENTINEL'
            except ValueError as error:observed=str(error)
        need(observed==expected,'EXACT_READINESS_'+name)
        need(state(guard)==before,'NO_LIFECYCLE_MUTATION_'+name)
        need(len(calls)-start==(1 if expected=='SENTINEL' else 0),'SOURCE_CALLBACK_BOUNDARY_'+name)
        results.append({'case':name,'status':'PASS','observed':observed,'state_before_after':before})
    def admitted(name):
        guard=inert_guard();c=component()
        recorder=c.Recorder(root/name,guard.latch.raw,['UNUSED_%03d'%i for i in range(109)])
        recorder.admit({}, {}, {'changes':[]},{'passed':True,'before_materialization':True},{'forward_calls':0})
        guard.latch.recorder=recorder
        return guard
    try:
        check('clean_pending',inert_guard(),'SENTINEL')
        guard=inert_guard();guard.latch.raw.trip('H_IDENTITY');check('terminal_first_cause',guard,'HF_PRELOAD_PENDING')
        guard=admitted('admitted');guard.latch.admit();check('postsetup_admitted',guard,'HF_PRELOAD_PENDING')
        guard=admitted('consumed');guard.latch.consume(CELL_ID);check('consumed',guard,'HF_PRELOAD_PENDING')
        guard=admitted('complete');guard.latch.consume(CELL_ID);guard.latch.raw._finish();check('complete',guard,'HF_PRELOAD_PENDING')
        guard=admitted('scientific');guard.latch.raw.scientific_stop('ROUTING','a'*64,0);check('scientific_stop',guard,'HF_PRELOAD_PENDING')
        guard=inert_guard();guard.latch.adapter_primary_code='FIRST_FAULT';check('adapter_primary',guard,'HF_PRELOAD_PENDING')
        guard=inert_guard();guard.forward_ticket=('unexpected',);check('active_ticket',guard,'HF_PRELOAD_GUARD')
        guard=inert_guard();check('rebound_grad',guard,'HF_PRELOAD_WRAPPERS',lambda stub:setattr(stub.autograd,'grad',lambda:None))
        guard=inert_guard();guard.model_class.forward=lambda:None;check('rebound_forward',guard,'HF_PRELOAD_WRAPPERS')
        guard=inert_guard();guard.counters.consume_load();check('already_loaded',guard,'HF_PRELOAD_COUNTER')
        # Original post-setup gate is still strict before setup and usable only after actual recorder admission.
        guard=inert_guard()
        try:guard.latch.admit()
        except component().HookStopped:pass
        need(guard.latch.raw.primary_code=='H_REENTRY','ORIGINAL_PENDING_DISPATCH_STILL_DENIED')
        results.append({'case':'unchanged_postsetup_admit','status':'PASS','pending_denied':'H_REENTRY',
                        'actual_recorder_admission_exercised':True,'no_full_hook_completion_claim':True})
    finally:helper_binding.ensure_package=original
    need(not any(name in sys.modules for name in ('torch','transformers','transformer_lens','numpy')),'NO_LIBRARY_IMPORTS')
    print(json.dumps({'status':'PASS','results':results,'source_callback_count':len(calls),'actual_library_imports':0,
                      'model_work':False,'real_authority':False},sort_keys=True))
if __name__=='__main__':main()
