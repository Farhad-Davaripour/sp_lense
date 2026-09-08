"""Only pre-import exclusive role gates differ from the reviewed v2 controller."""
from pinned import install
install(globals(),"production_run")

_checked_controller,_checked_worker,_checked_audit=controller,worker,audit


def controller(outer_cutoff):
    from authority import current
    from real_boundary import write_exclusive,encoded
    boundary,approved,admission=current()
    identity=boundary.reauthenticate(approved,admission)
    write_exclusive(boundary.control/"CONTROLLER_ENTRY.json",encoded({"execution":identity["execution"],
        "observed_model_work":"NONE_BEFORE_OWNED_LAUNCH"}),boundary.root)
    return _checked_controller(outer_cutoff)


def worker(deadline):
    from authority import current
    boundary,approved,admission=current()
    boundary.role("worker",approved,admission)  # BOOTSTRAP proves prior retained permission.
    return _checked_worker(deadline)  # First possible engine/torch import is inside this call.


def audit(deadline):
    from authority import current
    boundary,approved,admission=current()
    boundary.role("audit",approved,admission)
    return _checked_audit(deadline)
