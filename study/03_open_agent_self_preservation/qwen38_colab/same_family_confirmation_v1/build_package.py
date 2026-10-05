"""Seal a prospective private run packet; no model loading, transport or API calls."""
import json
import math
import zipfile
from pathlib import Path
from prepare import HERE, SOURCE, ARTIFACT, digest, sha, write_new

BASE = 'ec43e239f6ffad79d854149f9d7115aeda5292beddd411dd763cac8f9ddcd755'


def main():
    old=json.loads((HERE/'SEALED_FREEZE_v8.json').read_bytes())
    assert digest({k:v for k,v in old.items() if k!='freeze_sha256'})==old['freeze_sha256']
    cases=ARTIFACT/'cases10.json'
    assert sha(cases)==old['fixture_case_sha256']
    budget=json.loads((HERE/'BUDGET_ESTIMATE.json').read_bytes())
    manifest=dict(old)
    manifest.update(format='research3_same_family_confirmation_sealed_v9',
        supersedes_sealed_v8_sha256=old['freeze_sha256'],
        operational_amendment='prior phase actual/conservative debit fields are required; omission cannot restore full allowance; no scientific query or change',
        human_budget_authorization='relax the unit phase allowance and let it use the units do the next part of this resarech',
        human_phase_allowance_relaxed=True, packed_base_pin=BASE, campaign_budget=budget,
        projected_episode_output_tokens=sum(v['output_tokens']*(6 if 'lease_ledger' in k else 4) for k,v in budget['measured'].items()))
    manifest['unchanged_source_sha256']={n:sha(SOURCE/n) for n in (*old['unchanged_source_sha256'],'base_download.py','model_pin.json')}
    for name,pin in old['unchanged_source_sha256'].items():assert manifest['unchanged_source_sha256'][name]==pin
    for name,pin in old['runtime_engine_sha256'].items():assert sha(SOURCE.parent/'code'/name)==pin
    manifest['runtime_engine_sha256']={n:sha(SOURCE.parent/'code'/n) for n in (*old['runtime_engine_sha256'],'setup_download_parallel.py')}
    manifest['condition_validity_required_for_final_success']='No parse errors, truncation or rejected fictional tool calls in either phase; original/predeclared endpoint and validity retained separately'
    names=('prepare.py','scoring.py','audit_corpus.py','runtime_worker.py','budget.py','admission.py','test_admission.py','build_package.py','PROTOCOL.md','README.md')
    manifest['public_code_sha256']={n:sha(HERE/n) for n in names}
    manifest['runtime_harness_sha256']={n:sha(HERE/n) for n in ('runtime_worker.py','scoring.py','admission.py')}
    manifest['freeze_sha256']=digest({k:v for k,v in manifest.items() if k!='freeze_sha256'})
    manifest['loaded_adapter_identity_algorithm']='SHA256 of each frozen lora_ named parameter: name, shape, dtype and contiguous uint8 bytes in named_parameters order; compare every condition and final against pre-inference'
    manifest['freeze_sha256']=digest({k:v for k,v in manifest.items() if k!='freeze_sha256'})
    write_new(HERE/'SEALED_FREEZE_v9.json',manifest)
    template={'full_fixed_batch_admitted':False,'same_family_case_count':10,'paired_condition_count':20,
        'human_phase_allowance_relaxed':True,'authorized_campaign_cap_units':19,
        'phase_prior_actual_debit_units':0,'phase_prior_conservative_debit_units':0,
        'carried_Research3_actual_units':166.33,'carried_Research3_conservative_units':166.61,
        'carried_actual_API_attempts':104,'carried_actual_API_successes':104,'carried_API_input_estimated_USD':.010242876,
        'broker_enforces_shared_totals':False,'allocation_requested_utc':None,'preallocation_receipt_utc':None,
        'preallocation_credit_units':None,'preallocation_sessions':None,'preallocation_usage_units_per_hour':None,
        'allocated_gpu_type':None,'allocated_GPU_sessions':None,'worker_count':1,'allocated_units_per_hour':None,
        'packed_base_pin':BASE,'export_reserve_seconds':900,
        'worker_seconds':math.ceil(budget['conservative_seconds']-budget['reserved_seconds']['setup']-900),
        'predeclared_projected_episode_tokens':manifest['projected_episode_output_tokens'],
        'predeclared_projected_Jev_events':budget['projected_Jev_calls']}
    bundle=ARTIFACT/'prepared_bundle_v9';bundle.mkdir(parents=True,exist_ok=True)
    write_new(bundle/'FREEZE.json',manifest);write_new(bundle/'ADMISSION_REQUIRED.json',template)
    files={'cases10.json':cases,**{n:HERE/n for n in manifest['runtime_harness_sha256']},
        **{'source/'+n:SOURCE/n for n in manifest['unchanged_source_sha256']},
        **{'engine/'+n:SOURCE.parent/'code'/n for n in manifest['runtime_engine_sha256']}}
    for relative,origin in files.items():
        target=bundle/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert target.read_bytes()==origin.read_bytes()
        else:target.write_bytes(origin.read_bytes())
    entries={p.relative_to(bundle).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(bundle.rglob('*')) if p.is_file()}
    packet=ARTIFACT/'run_packet_v9.zip'
    if not packet.exists():
        with zipfile.ZipFile(packet,'x',compression=zipfile.ZIP_DEFLATED) as archive:
            for relative in entries:
                info=zipfile.ZipInfo(relative,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
                archive.writestr(info,(bundle/relative).read_bytes())
    with zipfile.ZipFile(packet) as archive:
        assert set(archive.namelist())==set(entries)
        import hashlib
        assert all(hashlib.sha256(archive.read(n)).hexdigest()==v['sha256'] for n,v in entries.items())
    receipt={'format':'same_family_prepared_packet_v9','freeze_sha256':manifest['freeze_sha256'],
        'packet_sha256':sha(packet),'packet_bytes':packet.stat().st_size,'verified_files':len(entries),
        'prepared_cases':10,'paired_conditions':20,'actual_neural_runs':0,'actual_Jev_attempts':0,
        'prepared_is_not_transfer_or_execution_evidence':True,'all_native_outputs_remain_private':True}
    write_new(ARTIFACT/'PACKET_MANIFEST_v9.json',entries);write_new(HERE/'PACKET_RECEIPT_v9.json',receipt)
    print(json.dumps(receipt,separators=(',',':')))


if __name__=='__main__':main()
