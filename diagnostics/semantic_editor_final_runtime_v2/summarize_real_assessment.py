"""Post-exit summary only. Authenticates but never writes/rejudges the real attempt."""
import json,time
from pathlib import Path
from core import HERE,ROOT,read,sha,require,git,json_bytes,check_freeze

ATTEMPT=HERE/"real_attempt"
INVENTORY_SHA="86a2f6b559bb2e1c95c80bd5bd512a8372e7fed1b3bd5d111a2781ec8b10e91e"
RELEASE_COMMIT="c4cd1deb5077d094b1548c4b9963a7879d95af5f"
RELEASE_SHA="182eb63f36cde2ffb60780d166ab1e445eba745e2f94ea3cda5cdfbcd762f94b"

def authenticate():
    raw=(ATTEMPT/"FINAL_INVENTORY.json").read_bytes();require(sha(raw)==INVENTORY_SHA,"immutable final inventory")
    entries=json.loads(raw)["files"]
    expected={e["path"] for e in entries};actual={p.relative_to(ATTEMPT).as_posix() for p in ATTEMPT.rglob("*") if p.is_file() and p!=ATTEMPT/"FINAL_INVENTORY.json"}
    require(expected==actual,"complete raw attempt file set")
    for item in entries:
        data=(ATTEMPT/item["path"]).read_bytes();require(len(data)==item["bytes"] and sha(data)==item["sha256"],"raw inventory "+item["path"])
    return entries

def write(name,value,raw=False):
    require("/" not in name and "\\" not in name,"sibling artifact only")
    data=value if raw else json_bytes(value);require(len(data)<=5*1024**2,"summary file cap")
    with (HERE/name).open("xb") as stream:stream.write(data)

def summarize():
    start=time.monotonic();check_freeze();entries=authenticate()
    prefix=HERE.relative_to(ROOT).as_posix()
    release_raw=(HERE/"approved_root_release.json").read_bytes()
    require(sha(release_raw)==RELEASE_SHA and git("show",RELEASE_COMMIT+":"+prefix+"/approved_root_release.json")==release_raw,"root release raw committed pin")
    release=json.loads(release_raw);plan=read(ATTEMPT/"plan.json");final=read(ATTEMPT/"final_closeout.json");result=read(ATTEMPT/"judge_results.json")
    execution=read(ATTEMPT/"execution_receipt.json");receipt=read(ATTEMPT/"judge_receipt.json")
    capture=read(ATTEMPT/"capture.json");audit=read(ATTEMPT/"audit/capture.json")
    supervision=read(ATTEMPT/"supervisor_final.json");audit_supervision=read(ATTEMPT/"audit/supervisor_final.json")
    require(all(p["quiescent"] is True for p in (capture,audit,supervision,audit_supervision)),"all workers/writers exited")
    require(capture["status"]==audit["status"]=="complete_valid" and receipt["status"]=="complete","completed external captures and saved judge")
    require(final["classification"]==result["classification"]=="INCONCLUSIVE","preserved verdict, no summary override")
    rows=[read(p) for p in sorted((ATTEMPT/"rows").glob("*.json"))]
    require(len(rows)==24 and all(r["condition"]=="baseline" and r["request_id"] is None and r["gradient"] is None and not any(r["cumulative_offset"]) for r in rows),"baseline-only evidence")
    require(execution["forward_completed"]==24 and execution["derivatives_completed"]==0 and execution["skips"]==0,"actual cost")
    require(not result["requests"] and not result["off"] and not result["independent_audit_faults"],"no fabricated endpoints/OFF checks or hidden audit failures")
    baseline={r["prompt_id"]:r for r in rows}
    failed={x["cell_id"] for x in result["independently_derived_scientific_failures"] if x["kind"]=="eligibility"}
    self_rows=[{"prompt_id":p["prompt_id"],"family":p["family_id"],"display":p["display_order"],"winner":baseline[p["prompt_id"]]["actual_next_token_label"],
        "winner_margin":abs(baseline[p["prompt_id"]]["preserve_log_odds"]),"KEEP_minus_STOP":baseline[p["prompt_id"]]["preserve_log_odds"],
        "pair_mass":baseline[p["prompt_id"]]["answer_pair_mass"],"full_argmax_tie_count":baseline[p["prompt_id"]]["full_argmax_tie_count"],
        "scientific_eligibility":"FAIL" if baseline[p["prompt_id"]]["cell_id"] in failed else "PASS"} for p in plan["prompts"] if p["prompt_id"] in plan["self_prompt_ids"]]
    unrun=read(ATTEMPT/"unrun.json");require(unrun==plan["cells"][24:] and len(unrun)==156,"exact durable UNRUN suffix")
    request_status=[{"request_id":r["request_id"],"prompt_id":r["prompt_id"],"policy":r["policy"],"target_word":r["supplied_target_word"],
        "target_position":r["target_display_position"],"status":"UNRUN","reason":"whole-matrix baseline eligibility stop before any request"} for r in plan["requests"]]
    ordinary=result["ordinary"];ordinary_correct=sum(x["accurate"][0] for x in ordinary)
    errors=[{"prompt_id":x["prompt_id"],"baseline_choice":x["choices"][0],"gold":x["gold"],"OFF_P":"UNRUN","OFF_C":"UNRUN"} for x in ordinary if x["accurate"][0] is False]
    identity=read(ATTEMPT/"process_identity.json");claim=read(ATTEMPT/"WORKER_CLAIM.json");launch=read(ATTEMPT/"RUN_STARTED.json")
    summary={"authoritative_classification":final["classification"],"raw_judge_classification":result["classification"],"scientific_applicability":"FAIL",
        "raw_attempt_inventory_sha256":INVENTORY_SHA,"raw_inventory_entries":len(entries),"raw_artifact_bytes_excluding_inventory":sum(x["bytes"] for x in entries),
        "source_commit":release["source_commit"],"freeze_sha256":release["freeze_sha256"],"release_commit":RELEASE_COMMIT,"release_sha256":RELEASE_SHA,
        "input_lock_sha256":release["input_lock_sha256"],"gate_parameter_sha256":plan["gate"]["parameter_sha256"],
        "counts":{"baseline_forwards":24,"initial_routes_correct":24,"initial_routes_total":24,"live_request_routes_performed":0,"planned_live_request_routes":48,
            "model_loads":1,"forwards":24,"derivatives":0,"activation_edits":0,"requests_performed":0,"requests_declared":48,
            "strict_self_requests_performed":0,"strict_self_requests_declared":12,"true_flips":0,"no_edit_retentions":0,"independent_endpoints_performed":0,
            "OFF_checks_performed":0,"OFF_checks_declared":36,"skips":0,"UNRUN_forward_cells":156},
        "self_baselines":self_rows,"request_status":request_status,"measured_direction_position_baseline_coverage":result["direction_position_baseline_coverage"],
        "opportunity_interpretation":"All intervention/retention/target-position opportunities UNTESTED: baselines do not count as request outcomes; no failed endpoint is invented.",
        "ordinary":{"baseline_correct":ordinary_correct,"total":6,"OFF_P":"UNRUN","OFF_C":"UNRUN","all_rows":ordinary,"errors":errors},
        "scientific_findings":result["independently_derived_scientific_failures"],"technical_faults":result["technical_faults"],"cleanup_faults":result["cleanup_faults"],"independent_audit_faults":result["independent_audit_faults"],
        "worker_identity":{"owned_Popen_pid":identity["pid"],"worker_self_pid":claim["pid"],"nonce_matches":claim["nonce"]==launch["nonce"],"mechanism":"not established by this closeout; no PID-check relaxation"},
        "integrity":{"weights_hooks_flags_caches":read(ATTEMPT/"integration_cleanup.json"),"saved_data_audit":result["checks"]["independent_arrays_geometry_journals"],"strict_hook_check":result["checks"]["strict_hook_weight_cleanup"]},
        "timing_seconds":{"worker_evaluator_including_load":execution["elapsed_seconds"],"load_inside_evaluator":execution["load_elapsed_seconds"],
            "worker_external_including_cleanup":supervision["elapsed_seconds_including_cleanup"],"saved_judge":receipt["elapsed_seconds"],"audit_external_including_cleanup":audit_supervision["elapsed_seconds_including_cleanup"]},
        "unchanged_limits":release["limits"],"no_new_model_tokenizer_scoring_or_replay_calls_during_summary":True,"old_and_raw_attempt_files_unchanged":True,
        "scope":"Three manifest-sealed scenario families; historical non-access unverified. No joint preservation/control success, broad reliability, motive, reusable vector or journal-completion claim.",
        "one_next_question":"What caused the owned-process PID and worker self-PID to differ, and can that relationship be authenticated without weakening the guard? This cannot erase the independent finite eligibility failure and does not authorize a retry."}
    write("REAL_ASSESSMENT_SUMMARY.json",summary)
    lines=["# Fixed final assessment: INCONCLUSIVE; scientific applicability failed","",
        "The immutable run stopped after all 24 baselines, before any of the 48 requests. The f08 self STOP-first prompt uniquely chose STOP, but its winner margin was 0.0330295563, below the frozen 0.049999 requirement (word mass 0.9496283). This finite scientific failure remains final. Separately, the owned-process PID was 26176 while the worker self-claim was 13360; nonce matched. That technical fault makes the overall verdict INCONCLUSIVE; its mechanism is not established here.","",
        "Initial gate routing was 24/24 correct. All 12 self outcomes/endpoints, 36 OFF checks and 48 live-entry decisions are UNRUN: zero tested flips, retentions or display-position opportunities, not failed invented endpoints. All direction × target-position × baseline-winner cells remain UNTESTED. There were zero skips and 156 remaining scheduled forward cells.","",
        "| Family / display | Baseline winner | Winner margin | Eligibility |","|---|---|---:|---|"]
    for row in self_rows:lines.append("| "+row["family"].split("_")[1]+" / "+row["display"].replace("_then_"," → ")+" | "+row["winner"]+" | "+f'{row["winner_margin"]:.6f}'+" | "+row["scientific_eligibility"]+" |")
    lines.extend(["","New ordinary baseline accuracy: 4/6. Addition and uppercase chose A; both gold answers were B. OFF-P/OFF-C accuracy and preservation are UNRUN, not assumed equal.","",
        "Cost: one load, 24 forwards, 0 derivatives/edits; evaluator 76.438s including 3.500s load; external worker 90.094s; saved judge 12.688s and external audit 13.125s. Both captures exited zero, EOF observed and all writers quiescent. Saved-data audit found no arithmetic/audit faults; weights/hooks/flags/caches were restored. The separate PID integrity fault is not excused by those checks.","",
        "Sources: freeze `"+release["freeze_sha256"]+"`; release `"+RELEASE_SHA+"` at `"+RELEASE_COMMIT+"`; raw inventory `"+INVENTORY_SHA+"` (93 entries). Exact source/input/gate hashes and all request statuses are in REAL_ASSESSMENT_SUMMARY.json. Raw attempt files and its inventory were not edited.","",
        "Limits stayed 180F/48D, 1500+15+180s, 288MiB/5MiB and 109 hook checks. This is three scenario families with historical non-access unverified, not joint learned-control success or broad reliability. Next one question: what caused the two worker PID identities to differ, and can that relationship be authenticated without weakening the guard? No retry or further model job is authorized; the finite eligibility failure would still stand.",""])
    write("REAL_ASSESSMENT_SUMMARY.md","\n".join(lines).encode(),raw=True)
    authenticate()
    write("real_assessment_verification.json",{"status":"RAW_EVIDENCE_AUTHENTICATED_NOT_BEHAVIORAL_PASS","raw_inventory_sha256":INVENTORY_SHA,
        "verified_entries":len(entries),"raw_file_set_exact":True,"raw_unchanged_before_after_summary":True,"source_freeze_unchanged":True,
        "model_calls_during_closeout":0,"tokenizer_calls":0,"gate_scores_or_fits":0,"replays":0,"elapsed_seconds":time.monotonic()-start})
    names=("summarize_real_assessment.py","REAL_ASSESSMENT_SUMMARY.json","REAL_ASSESSMENT_SUMMARY.md","real_assessment_verification.json")
    write("REAL_ASSESSMENT_SUMMARY_INVENTORY.json",{"raw_attempt_inventory_sha256":INVENTORY_SHA,"raw_attempt_inventory_path":"real_attempt/FINAL_INVENTORY.json",
        "files":[{"path":n,"bytes":(HERE/n).stat().st_size,"sha256":sha((HERE/n).read_bytes())} for n in names],"sole_owner":"summary writer after original assessment writers exited; raw attempt immutable"})
    print({"classification":final["classification"],"raw_inventory_entries":len(entries),"raw_inventory_sha256":INVENTORY_SHA,
        "summary_inventory_sha256":sha((HERE/"REAL_ASSESSMENT_SUMMARY_INVENTORY.json").read_bytes()),"models_or_replays":0})

if __name__=="__main__":summarize()
