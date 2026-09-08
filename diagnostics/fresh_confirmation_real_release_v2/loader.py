"""Single owned loader dispatch; immutable pinned Qwen loader remains unchanged."""
import time
from real_boundary import need


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
        from pinned import module,COMMIT,PREFIX
        need(target["commit"]==COMMIT and target["path"]==PREFIX+"loader.py","D_PINNED_LOADER_TARGET")
        checked=module("loader")
        model=checked.load_adapter(writer,counters,deadline,current_identity)
        return model
    try:
        return boundary.dispatch(approved,admission,invoke,sentinel=False)
    except BaseException:
        # A failed boundary receipt after load never permits the first forward.
        if model is not None:
            model.latch.stop("LOADER_BOUNDARY_CLOSEOUT")
            try: model.end_edit()
            finally: model.guard.restore()
        raise
