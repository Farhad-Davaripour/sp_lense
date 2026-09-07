"""Explicit delta over the checked actual-input engine; no synthetic fallback."""
import sys
import types
from support import HERE, ROOT, SOURCES, bounds, require, sha

RUNTIME_COMMIT = "2e11335f1036532b6f79fb2927dd9616c8a09915"
RUNTIME = "diagnostics/fresh_confirmation_runtime_v1/"
WORKFLOW_COMMIT = "638ef2a8f1eac0679c92d07a62cc8ae9bfac41b1"
WORKFLOW = "diagnostics/fresh_confirmation_workflow_v1/"


def once(source, old, new):
    require(source.count(old) == 1, "unique real adapter source boundary")
    return source.replace(old,new)


def adapted_sources():
    previous = SOURCES.load("checked_runtime_source_adapter",RUNTIME_COMMIT,RUNTIME+"bind_runtime.py")
    engine,judge = previous.adapted_sources()
    engine = once(engine,"from fake_backend import make_backend","from real_adapter import make_backend")
    engine = once(engine,'require(mode == "normal", "success-only fake candidate")',
                  'require(mode == "production", "only separately admitted production lane")')
    engine = once(engine,'model = make_backend(science.artifact_raw, mode)',
                  'model = make_backend(writer, counters, deadline)')
    engine = once(engine,'        cursor += 1',
                  '        model.latch.admit()\n        model.latch.consume(cell["cell_id"])\n        cursor += 1')
    engine = once(engine,'sha(b"fake-unchanged-nonfinal-state")','model.capture["unselected_sha256"]')
    engine = once(engine,'g = torch.autograd.grad(z[50057] - z[48964], h)[0]','g = model.gradient(z)')
    engine = once(engine,'"finite current fake gradient"','"finite actual current gradient"')
    anchor = 'writer.write_row(f"rows/{counters.attempts[\'forward\']:03d}.json", encoded(row))'
    engine = once(engine,anchor,
        'require(model.capture["input_int64_le_sha256"] == prompt["input_int64_le_sha256"] and model.capture["final_input_index"] == prompt["final_input_index"], "actual dispatched tensor proof")\n'
        '            if baseline is not None:\n'
        '                require(row["unselected_sha256"] == baseline[0]["unselected_sha256"], "measured nonfinal states unchanged on every replay/update")\n'
        '            '+anchor)
    anchor = 'state["cell_status"][cell["cell_id"]] = "FAILED"\n            raise\n\n    def identity'
    engine = once(engine,anchor,
        'state["cell_status"][cell["cell_id"]] = "FAILED"\n'
        '            model.latch.stop("CELL_FAILURE")\n            raise\n'
        '        finally:\n            model.clear_capture()\n\n    def identity')
    engine = once(engine,'initial_weight = sha(model.weight.detach().numpy().tobytes())','initial_weight = model.initial_digest')
    engine = engine.replace('model.cache = None','model.clear_capture()')
    engine = once(engine,'require(model.active is None and model.cache is None and model.weight.requires_grad, "cold request state")',
                  'model.start_request(rid)')
    engine = once(engine,'model.active = rid\n                model.weight.requires_grad_(False)','model.begin_edit()')
    start = engine.index('                finally:\n                    model.weight.requires_grad_(True)')
    end = engine.index('            state["request_status"][rid] = "DONE"',start)
    engine = engine[:start]+'''                finally:
                    model.end_edit()
                    if not writer.sticky_failure:
                        measured = model.parameter_state()
                        event("cleanup_events.jsonl", {"request_id": rid,
                            "flags_restored": measured["parameter_flags_restored"],
                            "gradients_absent": measured["parameter_gradients_absent"],
                            "parameter_versions_unchanged": measured["parameter_versions_unchanged"],
                            "parameter_identities_unchanged": measured["parameter_identities_unchanged"]})
            model.finish_request()
'''+engine[end:]
    start = engine.index('        if model is not None:\n            model.weight.requires_grad_(True)')
    end = engine.index('        if gate is not None:',start)
    engine = engine[:start]+'''        if model is not None:
            try:
                final_identity = model.finalize()
                hook_capture = model.recorder.finish()
                state["cleanup_complete"] = all(v for v in final_identity.values() if type(v) is bool)
                state["adapter_accounting"] = {"forwards":model.guard.forwards,"derivatives":model.guard.derivatives,
                    "rejected_dispatches":model.guard.rejected,"dispatch_failed":model.latch.failed}
                writer.write_source("runtime_adapter.json",encoded({"metadata":model.runtime_metadata,
                    "initial_parameter_sha256":model.initial_digest,"final_identity":final_identity,
                    "hook_checks":model.recorder.checks,"hook_capture":hook_capture,"hook_controller_status":model.recorder.status(),"input_bound":True,"fake_backend":False}))
            except BaseException:
                model.latch.stop("FINAL_IDENTITY_FAILURE")
                state["status"] = "INCONCLUSIVE_STUDY"
                state["technical_failures"].append({"code":"FINAL_IDENTITY_FAILURE"})
            finally:
                model.guard.restore()
'''+engine[end:]
    engine = engine.replace('"synthetic_only_no_real_hooks": True','"measured_real_adapter_identity": True')
    engine = engine.replace('"SYNTHETIC_ONLY"','"PRODUCTION_PREPARATION_ONLY"')
    engine = engine.replace('PASS_SYNTHETIC_ONLY','PASS_STUDY').replace('FAIL_SYNTHETIC_ONLY','FAIL_STUDY').replace('INCONCLUSIVE_SYNTHETIC_ONLY','INCONCLUSIVE_STUDY')
    engine = engine.replace('SYNTHETIC_ONLY fixed workflow started','PRODUCTION candidate fixed workflow started')
    engine = once(engine,'{"type": type(error).__name__, "message": str(error)}','{"code":"WORKER_EXCEPTION"}')
    engine = once(engine,'state["technical_failures"].append({"code":"WORKER_EXCEPTION"})',
                  'state["technical_failures"].append({"code":"WORKER_EXCEPTION"})\n        if model is not None: model.latch.stop("WORKER_EXCEPTION")')
    judge = once(judge,'from judge_binding import bound_plan','from judge_binding import bound_plan\nfrom saved_runtime import verify_runtime_identity')
    judge = once(judge,'require(state["cleanup_complete"] is True, "fake cleanup declaration required, not real ownership proof")',
                  'verify_runtime_identity(root,state,io)')
    judge = judge.replace('PASS_SYNTHETIC_ONLY','PASS_STUDY').replace('FAIL_SYNTHETIC_ONLY','FAIL_STUDY').replace('INCONCLUSIVE_SYNTHETIC_ONLY','INCONCLUSIVE_STUDY')
    require(not any(x in engine for x in ('from fake_backend','model.weight','model.active','model.cache','fake-unchanged','Fake(')),
            "all known fake-only engine placeholders removed")
    return engine,judge


def compiled(name, raw):
    module = types.ModuleType(name)
    module.__file__ = str(HERE/(name+".py"))
    sys.modules[name] = module
    exec(compile(raw,module.__file__,"exec"),module.__dict__)
    return module


def bind(*, injection_test=False, include_engine=True):
    pins = SOURCES.load("pins",WORKFLOW_COMMIT,WORKFLOW+"pins.py")
    pins.HERE,pins.ROOT,pins.subprocess = HERE,ROOT,SOURCES
    original = pins.io_components
    def components():
        writer,reader = original()
        sys.modules["binding"].subprocess = reader.subprocess = SOURCES
        return writer,reader
    pins.io_components = components
    area = SOURCES.load("area",WORKFLOW_COMMIT,WORKFLOW+"area.py")
    if injection_test:
        area.area_bounds = bounds
    else:
        from support import REAL_EVIDENCE,production_bounds
        area.EVIDENCE,area.area_bounds = REAL_EVIDENCE,production_bounds
    from hook_binding import bind_writer
    area.WorkflowWriter = bind_writer(area.WorkflowWriter)
    engine,judge = adapted_sources()
    return (compiled("production_engine",engine) if include_engine else None),compiled("production_saved_judge",judge)
