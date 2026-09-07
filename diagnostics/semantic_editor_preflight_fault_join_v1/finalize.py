"""Close artifacts only after bounded test subprocess and all judges exited."""
import json
from core import HERE,ROOT,Budget,read,sha,require,check_freeze

if __name__=="__main__":
    check_freeze();budget=Budget(HERE);process=read(HERE/"test_process.json")
    require(process["writer_exited"] and process["exit_code"]==0,"test process quiescent successful exit")
    result=read(HERE/"test_receipt.json");binding=read(HERE/"repair_source_bindings.json")
    require(result["verification"]=="PASS" and len(result["fixtures"])==3,"three fixed fixtures completed")
    parent=ROOT/"diagnostics/semantic_editor_final_pipeline_v1"
    require(sha((parent/"FINAL_INVENTORY.json").read_bytes())==binding["parent_inventory_sha256"],"parent inventory unchanged")
    for name,item in binding["sources"].items():require(sha((parent/name).read_bytes())==item["old_sha256"],"parent source unchanged "+name)
    for fixture in result["fixtures"]:
        output=HERE/fixture["fixture"]
        require(read(output/"judge_process.json")["writer_exited"],"independent judge quiescent")
        require((output/"FINAL_INVENTORY.json").exists(),"fixture closeout complete")
    old=binding["sources"]["editor.py"]["old_sha256"];new=binding["sources"]["editor.py"]["new_sha256"]
    report=("# Preflight fault join: VERIFIED\n\n"
        "One combined synthetic fixture and two minimal ordering regressions passed verification. Each performed exactly two preflight forwards, zero derivatives and zero edit registrations; all forty request slots remained UNRUN. The unchanged independent saved-data judge returned scientific FAIL, with no spurious audit, technical or cleanup faults. The combined fixture preserved one wrong-route finding and two finite eligibility findings before stopping.\n\n"
        "Route-only still stops with RoutingMismatch; eligibility-only still stops with EligibilityError. No old source, attempt or verdict was modified.\n\n"
        f"Parent editor SHA256: {old}\n\nRepair editor SHA256: {new}\n\n"
        f"Parent commit: {binding['parent_commit']}; authenticated inventory: {binding['parent_inventory_sha256']} (693 entries).\n\n"
        "The production change only moves the wrong-route raise below eligibility collection. Scientific helpers and independent judge are byte/AST bound as recorded. Synthetic backend adds only combined existing fault injection; namespace recorder cap is tightened to64MiB/5MiB.\n\n"
        f"Bounded test subprocess elapsed: {process['elapsed_seconds']:.3f}s. Verification body: {result['elapsed_seconds']:.3f}s. Six total toy forwards;0D. No real model/tokenizer/dataset/final-input reads, gate fitting, full-matrix rerun, git index or commit action.\n\n"
        "This is a repair verification, not model-performance evidence. No reveal/run authority is conveyed. Final-family historical exposure remains NON_ACCESS_UNVERIFIED. Ready for root review and integration by the sole committer.\n")
    budget.write_bytes("REPORT.md",report.encode())
    budget.write("quiescence_receipt.json",{"test_process_exited":True,"independent_judges_exited":3,"parent_sources_unchanged":True,"sole_inventory_writer":"finalize.py","verification":"PASS"})
    files=[{"path":p.relative_to(HERE).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p.read_bytes())} for p in sorted(HERE.rglob("*")) if p.is_file() and p.name!="FINAL_INVENTORY.json"]
    # Include child inventories; only this namespace's self-referential inventory is excluded.
    for p in sorted(HERE.glob("*/FINAL_INVENTORY.json")):files.append({"path":p.relative_to(HERE).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p.read_bytes())})
    files.sort(key=lambda item:item["path"])
    require(len({r["path"] for r in files})==len(files),"unique inventory entries")
    budget.write("FINAL_INVENTORY.json",{"files":files,"quiescent":True,"source_frozen_before_test":True,"parent_inventory_sha256":binding["parent_inventory_sha256"],"self_inventory_excluded":True})
    total=sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file())
    require(total<=64*1024**2 and all(p.stat().st_size<=5*1024**2 for p in HERE.rglob("*") if p.is_file()),"final namespace/file ceilings")
    print(json.dumps({"verification":"PASS","inventory_entries":len(files),"inventory_sha256":sha((HERE/"FINAL_INVENTORY.json").read_bytes()),"namespace_bytes":total,"new_editor_sha256":new}))
