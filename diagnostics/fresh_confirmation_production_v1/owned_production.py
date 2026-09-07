"""Only envelope/receipt scope differs from the checked retained-handle owner."""
from support import HERE, SOURCES, require


def source():
    raw = SOURCES.read("23ce46f1e5177791751b08bd15de07f9d836ecf8","diagnostics/fresh_confirmation_failure_delta_v1/supervisor.py").decode()
    old = '"source_freeze_sha256":sha((HERE/"SOURCE_FREEZE.json").read_bytes())}'
    new = '"source_freeze_sha256":sha((HERE/"SOURCE_FREEZE.json").read_bytes()), "production_release_sha256":__import__("json").loads((HERE/"PRODUCTION_AUTHORIZATION.json").read_bytes())["root_release_sha256"]}'
    require(raw.count(old) == 1, "one separate authority envelope addition")
    raw = raw.replace(old,new).replace('"fake_only":True','"fake_only":False').replace('"real_model_process":False','"real_model_process":lane == "production_worker"')
    # Bounded finite capture diagnostics, without arbitrary exception repr/str.
    raw = raw.replace('type(exc).__name__+": "+str(exc)','"CAPTURE_EXCEPTION"')
    raw = raw.replace('{"type":type(exc).__name__,"message":str(exc)}','{"code":"CAPTURE_EXCEPTION"}')
    return raw


def supervise(lane, deadline, cleanup_deadline, identity):
    from production_admission import admit_production
    admit_production("worker" if lane == "production_worker" else "audit")
    namespace = {"__file__":str(HERE/"owned_production_bound.py")}
    exec(compile(source(),namespace["__file__"],"exec"),namespace)
    return namespace["supervise"](lane,deadline,cleanup_deadline,identity)
