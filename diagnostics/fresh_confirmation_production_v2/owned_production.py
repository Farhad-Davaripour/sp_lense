"""Retained-handle owner with explicit authority and a shared cleanup bound."""
from support import HERE,SOURCES,require


def source():
    raw=SOURCES.read("23ce46f1e5177791751b08bd15de07f9d836ecf8","diagnostics/fresh_confirmation_failure_delta_v1/supervisor.py").decode()
    raw=raw.replace('"fake_only":True','"execution":__import__("authority").authenticate()["execution"]')
    raw=raw.replace('"real_model_process":False','"execution":__import__("authority").authenticate()["execution"]')
    old='"source_freeze_sha256":sha((HERE/"SOURCE_FREEZE.json").read_bytes())}'
    require(raw.count(old)==1,"exact ownership envelope delta")
    raw=raw.replace(old,'"source_freeze_sha256":sha((HERE/"SOURCE_FREEZE.json").read_bytes()), "execution":__import__("authority").authenticate()["execution"]}')
    raw=raw.replace('pair = owned.OwnedPair(launcher,actual,event)','pair = owned.OwnedPair(launcher,actual,event)\n        pair.cleanup_bound = absolute_cleanup_deadline')
    raw=raw.replace('type(exc).__name__+": "+str(exc)','"CAPTURE_EXCEPTION"').replace('{"type":type(exc).__name__,"message":str(exc)}','{"code":"CAPTURE_EXCEPTION"}')
    raw=raw.replace('elapsed_seconds=time.monotonic()-started,cleanup_seconds=time.monotonic()-cleanup_started,',
        'elapsed_seconds=time.monotonic()-started,cleanup_started_monotonic=min(cleanup_started,pair.stop_started_at) if pair is not None and pair.stop_started_at is not None else cleanup_started,\n            cleanup_seconds=time.monotonic()-(min(cleanup_started,pair.stop_started_at) if pair is not None and pair.stop_started_at is not None else cleanup_started),')
    raw=raw.replace('ownership_core_unchanged=True','ownership_identity_validation_unchanged=True,cleanup_bound_delta=True')
    return raw


def supervise(lane,deadline,cleanup_deadline,identity):
    from authority import authenticate
    authenticate()
    scope={"__file__":str(HERE/"owned_bound_v2.py")}
    exec(compile(source(),scope["__file__"],"exec"),scope)
    return scope["supervise"](lane,deadline,cleanup_deadline,identity)
