"""Model-free whole-campaign admission; the distributed template never authorizes a run."""
import math
from datetime import datetime, timezone


def utc(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Admission timestamps require an explicit UTC offset')
    return parsed.astimezone(timezone.utc)


def validate_admission(cfg, freeze, now=None):
    now = now or datetime.now(timezone.utc)
    budget = freeze['campaign_budget']
    assert cfg['full_fixed_batch_admitted'] is True
    assert cfg['same_family_case_count'] == 10 and cfg['paired_condition_count'] == 20
    cap = cfg['authorized_campaign_cap_units']
    assert math.isfinite(cap) and cap == budget['required_new_phase_cap_units'] == 19
    prior_actual = cfg['phase_prior_actual_debit_units']
    prior_conservative = cfg['phase_prior_conservative_debit_units']
    assert 0 <= prior_actual <= prior_conservative < cap
    available_cap = cap - prior_conservative
    assert available_cap >= budget['conservative_seconds'] * budget['conservative_planning_units_per_hour'] / 3600 + budget['margin_units']
    assert cfg['human_phase_allowance_relaxed'] is True
    assert cfg['carried_Research3_actual_units'] >= round(budget['Research3_actual_before'] + prior_actual, 8)
    assert cfg['carried_Research3_conservative_units'] >= round(budget['Research3_conservative_before'] + prior_conservative, 8)
    assert cfg['carried_Research3_conservative_units'] + available_cap <= budget['Research3_ceiling']
    attempts, successes = cfg['carried_actual_API_attempts'], cfg['carried_actual_API_successes']
    assert 104 <= successes <= attempts < 2048
    assert attempts + 2 * budget['projected_Jev_calls'] <= 2048
    cost = cfg['carried_API_input_estimated_USD']
    assert cost >= budget['old_API_input_estimated_USD']
    assert cost + budget['API_input_estimate_bound'] - budget['old_API_input_estimated_USD'] <= 1
    assert cfg['broker_enforces_shared_totals'] is True
    allocation, receipt = utc(cfg['allocation_requested_utc']), utc(cfg['preallocation_receipt_utc'])
    assert receipt <= allocation <= now
    assert (allocation - receipt).total_seconds() <= 600
    assert cfg['preallocation_credit_units'] >= available_cap
    assert cfg['preallocation_sessions'] == 0 and cfg['preallocation_usage_units_per_hour'] == 0
    assert cfg['allocated_gpu_type'] == 'A100'
    assert cfg['allocated_GPU_sessions'] == 1 and cfg['worker_count'] == 1
    rate = cfg['allocated_units_per_hour']
    assert 0 < rate <= budget['conservative_planning_units_per_hour']
    assert cfg['packed_base_pin'] == freeze['packed_base_pin']
    assert cfg['predeclared_projected_episode_tokens'] == freeze['projected_episode_output_tokens']
    assert cfg['predeclared_projected_Jev_events'] == budget['projected_Jev_calls']
    reserve = cfg['export_reserve_seconds']
    assert reserve >= budget['reserved_seconds']['integrity'] + budget['reserved_seconds']['export_release']
    required = math.ceil(budget['conservative_seconds'] - budget['reserved_seconds']['setup'])
    assert cfg['worker_seconds'] >= required - reserve
    # Use the frozen conservative rate even when the observed allocated rate is lower.
    remaining = (available_cap - budget['margin_units']) * 3600 / budget['conservative_planning_units_per_hour'] - (now - allocation).total_seconds()
    assert remaining >= required, 'Insufficient complete window; do not begin a partial fixed campaign'
    return min(cfg['worker_seconds'], remaining - reserve)
