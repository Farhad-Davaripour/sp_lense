"""Pre-query campaign estimate from verified neural timing receipts; no model loading."""
import json
from pathlib import Path
from prepare import HERE,ARTIFACT,sha,write_new


def main():
    native=Path('C:/Users/farha/Research3/artifacts/sentence_control_v1/closeout_v6/verified/native')
    previous=native/'prior_attempts/qwen38_sentence_diagnostic_v3_20261005T131745Z_f9733a50'
    measured={}
    for root,name in ((previous,'q38_dev_lease_ledger_self_complete_control'),(previous,'q38_dev_lease_ledger_self_complete_sentence'),(previous,'s72111_ordered_ticket_51_self_unfinished_control'),(native,'s72111_ordered_ticket_51_self_unfinished_sentence')):
        path=root/'episodes'/f'{name}.json';d=json.loads(path.read_bytes());turns=d['initial_turns']+d['threat_turns']
        seconds=sum(sum(x['model_seconds'] for x in json.loads((root/t['token_record_path']).read_bytes())['token_records']) for t in turns)
        measured[name]={'output_tokens':sum(len(t['token_ids']) for t in turns),'model_seconds':seconds,'episode_sha256':sha(path)}
    annotations=[json.loads(p.read_bytes()) for p in (native/'annotations').glob('*.json')]
    service=sum(x['service_seconds'] for x in annotations)/len(annotations)
    max_service=max(x['service_seconds'] for x in annotations)
    lease_model=sum(v['model_seconds'] for k,v in measured.items() if 'lease_ledger' in k)
    pending_model=sum(v['model_seconds'] for k,v in measured.items() if 'ordered_ticket' in k)
    # Conservative unmeasured proxies: one-step pending uses ordered timings;
    # each negative uses completed-work positive timings and event count.
    model_projection=6*lease_model+4*pending_model
    events=6*31+4*52
    overhead={'setup':600,'load':300,'integrity':300,'export_release':600}
    expected_seconds=model_projection+events*service+sum(overhead.values())
    conservative_seconds=model_projection*1.25+events*max_service+sum(overhead.values())
    rate=6.77; margin=.5
    expected_units=expected_seconds*rate/3600+margin
    conservative_units=conservative_seconds*rate/3600+margin
    cap=float(__import__('math').ceil(conservative_units))
    remaining=4.59
    receipt={'basis':'verified pilot end-to-end timing, not provider-only latency','measured':measured,
       'mean_end_to_end_service_seconds':service,'max_end_to_end_service_seconds':max_service,
       'projected_episode_model_seconds':model_projection,'projected_Jev_calls':events,
       'unmeasured_one_step_and_negatives_use_conservative_existing_family_proxies':True,
       'reserved_seconds':overhead,'conservative_planning_units_per_hour':rate,'last_observed_GPU_rate_historical':5.3,
       'margin_units':margin,'expected_seconds':expected_seconds,'conservative_seconds':conservative_seconds,
       'expected_units':expected_units,'required_new_phase_cap_units':cap,'carried_phase_allowance_remaining':remaining,
       'full_fixed_batch_fits_current_allowance':False,'additional_allowance_requested':round(cap-remaining,2),
       'Research3_actual_before':166.33,'Research3_conservative_before':166.61,'Research3_ceiling':200,
       'conservative_with_requested_cap':166.61+cap,'old_API_attempts':104,'old_API_successes':104,
       'API_attempts_remaining':1944,'old_API_input_estimated_USD':.010242876,'API_input_estimate_bound':round(2*events*8192*.042/1e6+.010242876,6),
       'no_neural_GPU_Jev_queries_in_estimate':True,'no_purchase_or_budget_reset':True}
    assert cap+166.61<=200 and 2*events<=1944 and receipt['API_input_estimate_bound']<1
    write_new(HERE/'BUDGET_ESTIMATE.json',receipt)
    print(json.dumps({k:receipt[k] for k in ('expected_units','required_new_phase_cap_units','additional_allowance_requested','full_fixed_batch_fits_current_allowance','projected_Jev_calls')},separators=(',',':')))


if __name__=='__main__':main()
