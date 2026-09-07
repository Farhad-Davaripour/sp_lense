"""Exactly one bounded stdlib fabricated-fixture batch; no real-input entry point."""
import hashlib
import io
import json
import sys
import time
import traceback
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent

def write(name, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    if len(raw) > 1024*1024:
        raise RuntimeError("receipt file cap")
    with (HERE/name).open("xb") as stream:
        stream.write(raw)

def main():
    started = time.monotonic()
    write("BATCH_STARTED.json", {"batch":"fresh_confirmation_input_fabricated_001",
                                "one_attempt":True,"real_input_access_authorized":False})
    record = {"status":"FAIL","model_calls":0,"tokenizer_calls":0,"author_submissions_read":0,
              "scenario_candidates_authored":0,"synthetic_fixtures_only":True}
    output = io.StringIO()
    try:
        freeze = json.loads((HERE/"SOURCE_FREEZE.json").read_bytes())
        for row in freeze["files"]:
            path = HERE/row["path"]
            if path.stat().st_size != row["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise RuntimeError("frozen source mismatch:"+row["path"])
        sys.path.insert(0, str(HERE))
        import fixtures
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(fixtures.PacketCheckerTests)
        result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
        record.update(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                      status="PASS" if result.wasSuccessful() else "FAIL")
    except BaseException as exc:
        record.update(exception=type(exc).__name__+": "+str(exc),traceback=traceback.format_exc())
    finally:
        record["elapsed_seconds"] = time.monotonic()-started
        record["test_output"] = output.getvalue()
        if record["elapsed_seconds"] > 60:
            record["status"]="FAIL"
            record["budget_fault"]="60-second batch limit exceeded; no extension"
        record["all_tests_passed_does_not_admit_cohort"]=True
        write("TEST_RECEIPT.json", record)
    print(json.dumps({k:v for k,v in record.items() if k not in ("test_output","traceback")}))
    return 0 if record["status"] == "PASS" else 1

if __name__ == "__main__":
    sys.exit(main())
