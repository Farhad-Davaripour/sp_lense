"""Single owned loader dispatch; immutable pinned Qwen loader remains unchanged."""
import time
from real_boundary import need
from support import SOURCES


def load_adapter(writer,counters,deadline,admitted):
    from authority import current
    boundary,approved,admission=current()
    identity=boundary.role("writer",approved,admission)
    need(identity==admitted and identity["execution"]["permission_scope"]=="ROOT_REAL_SINGLE_ATTEMPT","D_REAL_LOADER_AUTHORITY")
    need(writer.root.resolve()==(boundary.output/"evidence"/"attempt").resolve(),"D_WRITER_OUTPUT_ROOT")
    need(counters.attempts=={"load":1,"forward":0,"derivative":0},"D_LOADER_ACCOUNTING")
    need(time.monotonic()<deadline,"D_LOADER_DEADLINE")
    model=None
    def invoke(current_identity,target):
        nonlocal model
        from loader_handoff import target_loader,run_candidate
        checked=target_loader(target)
        from diagnostic_counter import LoadSources
        checked.__globals__["SOURCES"]=LoadSources(SOURCES,counters)
        model=run_candidate(writer,counters,deadline,current_identity,checked)
        return model
    try:
        return boundary.dispatch(approved,admission,invoke,sentinel=False)
    except BaseException:
        from helper_binding import abort_load
        import sys
        # A failed boundary receipt after load never permits the first forward.
        if model is not None:
            model.latch.stop("LOADER_BOUNDARY_CLOSEOUT")
            try: model.end_edit()
            finally:
                abort_load(writer,sys.exception())
                model.guard.restore()
        else:abort_load(writer,sys.exception())
        raise
