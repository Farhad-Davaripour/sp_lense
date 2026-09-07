"""ONE <=120-second fake-only workflow batch; retained receipts, no restart."""
import hashlib
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def write_new(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    if len(raw) > 5*1024**2:
        raise ValueError("preparation per-file ceiling")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def check_freeze():
    frozen = json.loads((HERE / "SOURCE_FREEZE.json").read_bytes())
    for name, expected in frozen["source_sha256"].items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest() != expected:
            raise ValueError("frozen source mismatch: "+name)
    return frozen


def main():
    started = time.monotonic()
    deadline = started+120
    frozen = check_freeze()
    write_new(HERE/"BATCH_STARTED.json", {"single_batch":True,"maximum_seconds":120,
        "source_freeze_sha256":hashlib.sha256((HERE/"SOURCE_FREEZE.json").read_bytes()).hexdigest(),
        "real_run_authorized":False,"usage_percent_before_batch":json.loads((HERE/"USAGE_BEFORE_BATCH.json").read_bytes())["used_percent"]})
    receipt = {"status":"INCONCLUSIVE_PREPARATION_ONLY","cases":[],"real_model_calls":0,"tokenizer_calls":0,
               "dataset_reads":0,"author_submission_reads":0,"real_supervisors_executed":False,
               "real_hook_capacity_verified":False,"milestone_credit":False,"error":None}
    captures = {}
    try:
        from workflow import execute
        from judge import judge
        from area import EVIDENCE, area_bounds
        from pins import READS, require
        modes = (("success","normal"),("routing_prefix","routing"),("finite_eligibility_prefix","finite_eligibility"),
                 ("endpoint_corruption_prefix","endpoint_corruption"),("partial_write_prefix","partial_write"))
        for name, mode in modes:
            require(time.monotonic() < deadline, "one targeted batch deadline")
            capture = execute(name,mode,deadline)
            captures[name] = capture
            # External hash is saved before independent adjudication; no changed
            # expected hash may hide endpoint/file corruption.
            write_new(HERE/("CAPTURE_"+name+".json"),capture)
            result = judge(EVIDENCE/name,capture,deadline)
            if mode == "normal":
                require(result["classification"] == "PASS_SYNTHETIC_ONLY" and result["io_status"] == "COMPLETE", "full synthetic success")
                require({k:result[k] for k in ("forward_attempts","forward_completed","derivatives","loads","routes","self_endpoints","strict_flips","retentions","off_identities")}
                        == dict(zip(("forward_attempts","forward_completed","derivatives","loads","routes","self_endpoints","strict_flips","retentions","off_identities"),(96,96,6,1,72,12,6,6,36))), "exact success counts")
                require(result["cell_counts"] == {"DONE":96,"SKIPPED":84} and result["request_counts"] == {"DONE":48}, "success no UNRUN")
                require(all(v == {"tested":6,"correct":4} for v in result["ordinary_accuracy"].values()), "unchanged fake ordinary accuracy including errors")
            elif mode in ("routing","finite_eligibility"):
                require(result["classification"] == "FAIL_SYNTHETIC_ONLY" and result["forward_attempts"] == result["routes"] == 1
                        and result["derivatives"] == 0 and result["request_counts"] == {"UNRUN":48}
                        and result["cell_counts"] == {"DONE":1,"UNRUN":179}, "scientific prefix leaves all requests UNRUN")
            elif mode == "endpoint_corruption":
                require(result["classification"] == "INCONCLUSIVE_SYNTHETIC_ONLY" and result["forward_attempts"] == 28
                        and result["derivatives"] == 1 and result["routes"] == 25 and result["self_endpoints"] == 1
                        and result["strict_flips"] == result["retentions"] == 0
                        and result["request_counts"] == {"FAILED":1,"UNRUN":47}
                        and result["cell_counts"] == {"DONE":28,"SKIPPED":6,"UNRUN":146}, "corrupt replay not successful flip/retention")
            else:
                require(result["classification"] == "INCONCLUSIVE_SYNTHETIC_ONLY" and result["io_status"] == "INCOMPLETE"
                        and result["forward_attempts"] == 1 and result["forward_completed"] == result["routes"] == result["derivatives"] == 0
                        and result["request_counts"] == {"UNRUN":48} and result["cell_counts"] == {"FAILED":1,"UNRUN":179}, "partial attempted write retained and remaining UNRUN")
                index = json.loads((EVIDENCE/name/capture["path"]).read_bytes())
                partial = [x for x in index["reconciliation"]["files"] if x.get("logit")]
                require(len(partial) == 1 and partial[0]["actual_bytes"] == 4096 and partial[0]["expected_bytes"] == 993280
                        and not partial[0]["complete"] and capture["remaining_closeout_bytes"] > 0, "partial raw bytes and closeout reserve honest")
            receipt["cases"].append({"name":name,"test_status":"PASS","capture":capture,"judge":result})
            print(json.dumps({"case":name,"status":"PASS","elapsed_seconds":time.monotonic()-started}),flush=True)
            area_bounds()
        require(len(receipt["cases"]) == 5 and time.monotonic() < deadline, "five bounded targeted cases")
        check_freeze()
        receipt["source_bindings"] = dict(READS)
        receipt["storage"] = area_bounds()
        receipt["status"] = "PASS_PREPARATION_ONLY"
    except BaseException as error:
        receipt["error"] = {"type":type(error).__name__,"message":str(error)}
    finally:
        receipt["elapsed_seconds"] = time.monotonic()-started
        receipt["captures"] = captures
        write_new(HERE/"TEST_REPORT.json",receipt)
    print(json.dumps({"status":receipt["status"],"cases_passed":len(receipt["cases"]),"elapsed_seconds":receipt["elapsed_seconds"],"error":receipt["error"]}),flush=True)
    return 0 if receipt["status"] == "PASS_PREPARATION_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
