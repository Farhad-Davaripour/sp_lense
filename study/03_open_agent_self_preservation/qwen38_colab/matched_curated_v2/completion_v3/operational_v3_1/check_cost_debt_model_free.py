"""Pure debt/time admission tests; no model imports or child processes."""
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
from admission import plan
def rejects(fn):
    try:fn()
    except (ValueError,RuntimeError):return True
    raise AssertionError("Unsafe plan admitted")
def main(output):
    p=Path(output).resolve();p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise RuntimeError("Use a fresh artifact output")
    a=plan(1000,1100,phase_incurred_units=.82)
    b=plan(1000,1100,phase_incurred_units=.90)
    tests={"debt_does_not_backdate_current_setup":a["setup_elapsed_seconds"]==b["setup_elapsed_seconds"]==100,
      "debt_shortens_future_deadline":b["phase_deadline_monotonic"]<a["phase_deadline_monotonic"],
      "cumulative_cap_not_reset":b["maximum_cumulative_units"]==164.92 and b["phase_units"]==16,
      "observed_debt_fully_covered":rejects(lambda:plan(0,0,phase_incurred_units=.81)),
      "old1051_fit_rejected":rejects(lambda:plan(0,0,fit_seconds=1051)),
      "unchanged_fit_eval_export":(b["fit_seconds"],b["eval_seconds"],b["export_seconds"])==(2700,3600,900),
      "five_minute_current_setup_fits":plan(1000,1300)["admitted"],
      "dynamic_setup_bound_before600":plan(0,0)["effective_setup_limit_seconds"]<600,
      "late_dynamic_setup_rejected":rejects(lambda:plan(0,590)),
      "debt2_rejects_whole_pipeline_even_before_setup":rejects(lambda:plan(0,0,phase_incurred_units=2)),
      "rate8_rejects_budget":rejects(lambda:plan(0,0,rate=8)),
      "startup_overhead_changes_fit_admission":rejects(lambda:plan(0,0,seconds_per_update=45,startup_seconds=175.41824))}
    result={"passed":all(tests.values()),"checks":tests,"example_debt90":b,
            "real_model_Torch_GPU_API_or_child_calls":0}
    p.write_bytes((json.dumps(result,indent=2)+"\n").encode());print(json.dumps(result))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);a=p.parse_args();main(a.output)
