"""Whole-pipeline time/cost admission; standard library only."""
import math
PHASE_UNITS=16.;RATE=6.77;OVERALL_UNITS=200.
SETUP_MAX=600.;FIT_MIN=2700.;EVAL_MIN=3600.;EXPORT_MIN=900.;MARGIN_UNITS=.5
OBSERVED_PRIOR_DEBT=.82
def finite(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError("Invalid numeric admission field: "+name)
    return float(value)
def observed_model_fit_estimate(seconds_per_update=19.4715888889,startup_seconds=175.41824,final_audit_seconds=120.):
    for name,value in locals().copy().items():finite(value,name)
    if seconds_per_update<=0 or startup_seconds<0 or final_audit_seconds<0:raise ValueError("Invalid observed fit timing")
    return {"updates":56,"seconds_per_update":seconds_per_update,"startup_seconds":startup_seconds,
        "final_audit_seconds":final_audit_seconds,
        "estimated_seconds":56*seconds_per_update+startup_seconds+final_audit_seconds,
        "completion_guarantee":False,"source":"Observed interrupted B timing plus explicit final-audit allowance"}
def plan(phase_started,now,cumulative_before=148.92,phase_units=PHASE_UNITS,rate=RATE,
         setup_limit=SETUP_MAX,fit_seconds=FIT_MIN,eval_seconds=EVAL_MIN,export_seconds=EXPORT_MIN,
         margin_units=MARGIN_UNITS,overall_units=OVERALL_UNITS,phase_incurred_units=.90,
         seconds_per_update=19.4715888889,startup_seconds=175.41824,final_audit_seconds=120.):
    values=locals().copy()
    for name,value in values.items():finite(value,name)
    if phase_units!=16 or overall_units!=200:raise ValueError("Frozen phase/overall authorization differs")
    if rate<=0:raise ValueError("Actual observed rate must be positive")
    estimate=observed_model_fit_estimate(seconds_per_update,startup_seconds,final_audit_seconds)
    if fit_seconds<estimate["estimated_seconds"]:raise ValueError("Fit below observed startup plus56updates and final-audit estimate")
    if not 0<=cumulative_before or cumulative_before+phase_units>overall_units:
        raise ValueError("Phase exceeds cumulative authorization")
    if setup_limit<=0 or setup_limit>600:raise ValueError("Setup window exceeds10minutes")
    if fit_seconds<2700:raise ValueError("Fit must admit at least45minutes including loading/audits/saves")
    if eval_seconds<3600:raise ValueError("Evaluation must admit at least60minutes including actual Jev decisions")
    if export_seconds<900:raise ValueError("Export/release must reserve at least15minutes")
    if margin_units<.5:raise ValueError("Whole-pipeline discretionary margin must be at least0.5unit")
    if phase_incurred_units<OBSERVED_PRIOR_DEBT or phase_incurred_units>=phase_units:
        raise ValueError("Prior phase debt must cover observed0.82units and fit within16units")
    if now<phase_started:raise ValueError("Clock precedes phase allocation")
    elapsed=now-phase_started
    if elapsed>setup_limit:raise RuntimeError("Setup10minute bound exceeded; no optimizer admitted")
    maximum_seconds=(phase_units-phase_incurred_units)/rate*3600
    margin_seconds=margin_units/rate*3600
    reserved=fit_seconds+eval_seconds+export_seconds+margin_seconds
    dynamic_setup_limit=min(setup_limit,maximum_seconds-reserved)
    if dynamic_setup_limit<0 or elapsed>dynamic_setup_limit:
        raise RuntimeError("Prior cost debt plus current setup cannot admit fit, full evaluation, export and margin")
    return {"admitted":True,"phase_started_monotonic":phase_started,"admitted_monotonic":now,
        "phase_units":phase_units,"phase_incurred_units":phase_incurred_units,
        "future_phase_units":phase_units-phase_incurred_units,
        "rate_units_per_hour":rate,"cumulative_before":cumulative_before,
        "current_cumulative_upper_before_allocation":cumulative_before+phase_incurred_units,
        "cost_debt_separate_from_setup_clock":True,
        "maximum_cumulative_units":cumulative_before+phase_units,
        "setup_elapsed_seconds":elapsed,"setup_limit_seconds":setup_limit,
        "effective_setup_limit_seconds":dynamic_setup_limit,
        "setup_deadline_monotonic":phase_started+dynamic_setup_limit,
        "fit_seconds":fit_seconds,"eval_seconds":eval_seconds,"export_seconds":export_seconds,
        "minimum_margin_units":margin_units,"phase_deadline_monotonic":phase_started+maximum_seconds,
        "planned_minimum_units_including_debt_and_current_setup":phase_incurred_units+(elapsed+fit_seconds+eval_seconds+export_seconds)*rate/3600,
        "remaining_seconds":maximum_seconds-elapsed,
        "estimate_is_completion_guarantee":False,"A_generation_required":False,
        "observed_model_fit_estimate":estimate}
def require_eval_and_export(plan_value,now):
    left=plan_value["phase_deadline_monotonic"]-now
    minimum=plan_value["eval_seconds"]+plan_value["export_seconds"]+plan_value["minimum_margin_units"]/plan_value["rate_units_per_hour"]*3600
    if left<minimum:raise RuntimeError("No complete evaluation/export window remains; B evaluation not started")
    return left
