"""Integration-only delta over the pinned checked adapter and saved science."""
import sys
import types
from support import HERE,ROOT,SOURCES,bounds,require,run_dir

V1="6573e3b5ce71c7a40f9d8a5bc9815b0154ebf129"
BASE="diagnostics/fresh_confirmation_production_v1/"
WORKFLOW_COMMIT="638ef2a8f1eac0679c92d07a62cc8ae9bfac41b1"
WORKFLOW="diagnostics/fresh_confirmation_workflow_v1/"


def once(source,old,new):
    require(source.count(old)==1,"unique integration source boundary: "+old[:70])
    return source.replace(old,new)


def compiled(name,raw):
    module=types.ModuleType(name)
    module.__file__=str(HERE/(name+".py"))
    sys.modules[name]=module
    exec(compile(raw,module.__file__,"exec"),module.__dict__)
    return module


def adapted_sources():
    old=SOURCES.load("checked_v1_adapter_binding",V1,BASE+"bind_production.py")
    engine,judge=old.adapted_sources()
    engine=once(engine,'from real_adapter import make_backend','from real_adapter import make_backend\nfrom authority import authenticate\nfrom runtime_closeout import close_runtime')
    engine=once(engine,'state = {"execution_mode": "PRODUCTION_PREPARATION_ONLY",','state = {"execution": authenticate()["execution"],')
    engine=once(engine,'             "real_supervisors_executed": False, "real_hook_capacity_verified": False}',
        '             "owned_process_evidence": "EXTERNAL_CONTROLLER_REQUIRED"}')
    engine=once(engine,'        event("scientific_failures.jsonl", record)\n        raise ScientificStop(kind)',
        '        event("scientific_failures.jsonl", record)\n        model.recorder.raw.scientific_stop(kind.upper(),sha(encoded(record)))\n        raise ScientificStop(kind)')
    engine=once(engine,'        except BaseException:\n            if state["cell_status"]',
        '        except ScientificStop:\n            raise\n        except BaseException:\n            if state["cell_status"]')
    engine=once(engine,'            current_rid = rid = spec["request_id"]\n            state["request_status"][rid] = "STARTED"',
        '            rid = spec["request_id"]')
    engine=once(engine,'            model.start_request(rid)',
        '            model.start_request(rid)\n            current_rid = rid\n            state["request_status"][rid] = "STARTED"')
    start=engine.index('        if model is not None:\n            try:\n                final_identity')
    end=engine.index('    return capture',start)
    engine=engine[:start]+'''        # Non-I/O status/accounting is captured even if any final writer fails.
        state["attempts"] = dict(counters.attempts)
        state["cursor"] = cursor
        state["fresh_routes"] = gate.calls if gate else 0
        capture = close_runtime(model,gate,writer,state,event,encoded)
'''+engine[end:]
    # Successful audit execution is not synonymous with scientific PASS.
    judge=once(judge,'verify_runtime_identity(root,state,io)','hook_result = verify_runtime_identity(root,state,io,capture)\n    if not hook_result["evidence_valid"]: technical.append("runtime_identity_or_hook_incomplete")')
    judge=once(judge,'"real_run_authorized":False,"real_model_result":False,"real_hook_capacity_verified":False,"real_supervisors_executed":False}',
        '"execution":state["execution"],"audit_completed":True,"hook_result":hook_result}')
    return engine,judge


def bind(*,include_engine=True):
    pins=SOURCES.load("pins",WORKFLOW_COMMIT,WORKFLOW+"pins.py")
    pins.HERE,pins.ROOT,pins.subprocess=HERE,ROOT,SOURCES
    original=pins.io_components
    def components():
        writer,reader=original()
        sys.modules["binding"].subprocess=SOURCES
        # Explicit new schema and independently authenticated execution field.
        raw=SOURCES.read("33f9f7b3b85325334da22abdc82d9db10f53a01e","diagnostics/semantic_confirmation_io_v1/reader.py").decode()
        raw=raw.replace('"sp_lense.confirmation_io_index.v1"','"sp_lense.confirmation_io_index.v2"')
        raw=once(raw,'demand(index["settings"] == expected_settings and index["real_run_authorized"] is False, "independently fixed resource settings")',
            'demand(index["settings"] == expected_settings, "independently fixed resource settings")\n    from authority import independent_execution\n    independent_execution(index["execution"])')
        reader=compiled("confirmation_saved_reader_v2",raw)
        reader.ROOT,reader.subprocess=ROOT,SOURCES
        return writer,reader
    pins.io_components=components
    area=SOURCES.load("area",WORKFLOW_COMMIT,WORKFLOW+"area.py")
    area.EVIDENCE=run_dir()/"evidence"
    area.EVIDENCE.parent.mkdir(parents=True,exist_ok=True)
    area.area_bounds=bounds
    from hook_binding import bind_writer
    area.WorkflowWriter=bind_writer(area.WorkflowWriter)
    engine,judge=adapted_sources()
    return (compiled("production_engine_v2",engine) if include_engine else None),compiled("production_saved_judge_v2",judge)
