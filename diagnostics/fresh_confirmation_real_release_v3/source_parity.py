"""Exact full-engine/scoring source parity and reviewed handoff source bindings."""
import ast
import json
from support import HERE,SOURCES,require,sha
def check():
    lock=json.loads((HERE/"INFRASTRUCTURE_PARITY.json").read_bytes())
    for row in lock["identical_files"]:
        require((HERE/row["namespace_file"]).read_bytes()==SOURCES.read(row["commit"],row["path"]),"unchanged reviewed file: "+row["namespace_file"])
    from bind_production import adapted_sources
    engine,judge=adapted_sources()
    require(sha(engine.encode())==lock["engine_sha256"] and sha(judge.encode())==lock["judge_sha256"],"exact full final-study engine and scientific judge bytes")
    binding=json.loads((HERE/"BINDINGS.json").read_bytes())
    require(binding["production_ceiling"]=={"forwards":180,"derivatives":48,"loads":1,"tokens":160,"hook_checks":109,"evidence_bytes":301989888,"file_bytes":5242880},"unchanged real final-study capacities")
    require(binding["production_seconds"]=={"worker":1800,"audit":180,"cleanup":15},"unchanged1995-second final controller")
    old=json.loads(SOURCES.read("518be64d39159539bbbaad4d5a675a97b5c43ef1","diagnostics/fresh_confirmation_real_release_v2/BINDINGS.json"))
    for key in ("input_binding","input_lock_sha256","runtime_spec_sha256","frozen_runtime_reference","resource_contract_sha256"):
        require(binding[key]==old[key],"exact authority/source/input/runtime/resource binding")
    base=SOURCES.read("518be64d39159539bbbaad4d5a675a97b5c43ef1","diagnostics/fresh_confirmation_real_release_v2/loader.py").decode()
    expected=base.replace('        from pinned import module,COMMIT,PREFIX\n        need(target["commit"]==COMMIT and target["path"]==PREFIX+"loader.py","D_PINNED_LOADER_TARGET")\n        checked=module("loader")\n        model=checked.load_adapter(writer,counters,deadline,current_identity)',
        '        from loader_handoff import target_loader,run_candidate\n        checked=target_loader(target)\n        model=run_candidate(writer,counters,deadline,current_identity,checked)')
    require((HERE/"loader.py").read_text()==expected,"only selected loader/context handoff changes in real loader boundary")
    for name in ("support.py","real_boundary.py"):
        source=SOURCES.read("518be64d39159539bbbaad4d5a675a97b5c43ef1","diagnostics/fresh_confirmation_real_release_v2/"+name).decode()
        require((HERE/name).read_text()==source.replace("fresh_confirmation_real_release_v2","fresh_confirmation_real_release_v3").replace("fresh_confirmation_real_attempt_002","fresh_confirmation_real_attempt_003"),"authority/clock/output delta is identity only")
    require(not any((HERE/name).exists() for name in ("setup_counter.py","setup_budget.py","setup_core.py","setup_loader.py")),"startup-only dispatch/control components not carried")
    for name in ("loader_handoff.py","saved_runtime.py","handoff_tests.py","weight_reader.py"):
        require(not any(isinstance(n,ast.Assert) for n in ast.walk(ast.parse((HERE/name).read_text()))),"new mandatory joins and expectations survive optimization")
    return {"status":"FULL_SCIENTIFIC_SOURCE_PARITY","engine_sha256":sha(engine.encode()),"judge_sha256":sha(judge.encode()),
        "identical_reviewed_files":len(lock["identical_files"]),"original14_predicates_and_digest_retained":True,
        "full_scientific_counters_controller_and_schedule_unchanged":True,"startup_zero_forward_controller_carried":False,
        "input_lock_sha256":binding["input_lock_sha256"],"resource_contract_sha256":binding["resource_contract_sha256"],
        "scientific_change":False}
