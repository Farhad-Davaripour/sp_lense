"""Verify all native repair bytes and independently recompute its chain check."""
import hashlib,io,json,shutil,sys,zipfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'14_cached_state_response_v1')]
from lossless_arrays import read_bundle
from maths import checks,wn
BASE=Path('C:/Users/farha/Research4/artifacts/fresh_memory_boundary_response_v1')
DOWNLOADS=Path('C:/Users/farha/Downloads')
PRIVATE=Path('G:/My Drive/sp_lense/research4/fresh_memory_boundary_response_v1')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def arrays(z,name):
    with np.load(io.BytesIO(z.read(name)),allow_pickle=False) as a:return {n:a[n].copy() for n in a.files}

def main():
    with zipfile.ZipFile(DOWNLOADS/'R4_NATIVE_SEED_REPAIR_TRANSFER_20261014.zip') as z:
        assert z.testzip() is None;transfer=json.loads(z.read('REPAIR_TRANSFER_RECEIPT_20261014.json'))
    assert transfer['native']['sha256']=='af1e34a1c6c869d78e97acf7ef436d758407ac32fc6e4a276b4f5a136820c775'
    assert transfer['native']['bytes']==76789098 and transfer['native']['phase17_calls']==8
    out=BASE/'repair_evaluated_20261014';out.mkdir(exist_ok=False)
    verified=[]
    for name,size,digest,manifest_name in [
        ('R4_NATIVE_SEED_REPAIR_20261014.zip',76789098,transfer['native']['sha256'],'REPAIR_EXPORT_HASHES.json'),
        (transfer['findings']['name'],transfer['findings']['bytes'],transfer['findings']['sha256'],'CONTENTS_SHA256.json')]:
        source=DOWNLOADS/name;dest=BASE/name
        assert source.stat().st_size==size and sha(source)==digest
        assert not dest.exists();shutil.copyfile(source,dest)
        with zipfile.ZipFile(dest) as z:
            assert z.testzip() is None
            manifest=json.loads(z.read(manifest_name))
            for n,h in manifest.items():
                assert hashlib.sha256(z.read(n)).hexdigest()==h
                if n.endswith('.json') or n.startswith('repair_points/'):
                    target=out/n;assert target.resolve().is_relative_to(out.resolve());target.parent.mkdir(parents=True,exist_ok=True)
                    if not target.exists():target.write_bytes(z.read(n))
                    assert hashlib.sha256(target.read_bytes()).hexdigest()==h
        verified.append({'name':name,'host':str(dest),'bytes':size,'sha256':digest,'all_members_SHA256_and_CRC_verified':True})
    (BASE/'REPAIR_TRANSFER_RECEIPT_20261014.json').write_text(json.dumps(transfer,indent=2)+'\n')
    p15=Path('C:/Users/farha/Research4/artifacts/cached_memory_transport_v1/R4_MEMORY_TRANSPORT_NATIVE_20261014.zip')
    with zipfile.ZipFile(p15) as z:
        ref=arrays(z,'probes/r14_cal_window_d0_q0.npz');target=arrays(z,'coefficients/r14_cal_window_d0.npz')
        schema=json.loads(z.read('STATE_SCHEMA.json'));base=arrays(z,'contexts/r14_cal_window_baseline.npz')
    def loader(reference_id,name):
        assert reference_id=='parent15:probes/r14_cal_window_d0_q0.npz';return ref[name]
    decoded={p.stem:read_bundle(p,loader) for p in (out/'repair_points').glob('*.npz')}
    assert len(decoded)==7
    for name in ['shared_seed','owned_seed']:
        assert all(np.array_equal(v,ref[n]) for n,v in decoded[name].items())
    result=json.loads((out/'REPAIR_RESULTS.json').read_text());coarse,fine=result['coarse'],result['fine']
    components=decoded['chain_J_r1'];plus=decoded[f'chain_H_s{fine}_1'];minus=decoded[f'chain_H_s{fine}_-1']
    predicted={n:components[n+'_first'].astype(np.float64)+(plus[n+'_first'].astype(np.float64)-minus[n+'_first'])/(2*fine) for n in schema}
    ck=checks(predicted,{n:target[n+'_second'] for n in schema},schema,.10)
    assert all(v['pass'] for v in ck.values())
    maximum_report_difference=max(abs(v['relative']-result['chain_groups'][g]['relative']) for g,v in ck.items())
    assert maximum_report_difference<1e-12
    # An independent route through the separately stored full-vocabulary slopes.
    # FP32 output-head rounding makes this a second route, not exact head equality.
    q=components['q'].astype(np.float64)+(plus['q'].astype(np.float64)-minus['q'])/(2*fine)
    independent_output_relative=wn(base['logits'],q-target['r'])/max(wn(base['logits'],target['r']),1e-30)
    assert independent_output_relative<.10
    mutation=json.loads((out/'NATIVE_SHARED_SEED_MUTATION.json').read_text())
    assert len(mutation['changed_fields'])==18 and all(mutation['before'][n]!=mutation['after'][n] for n in mutation['changed_fields'])
    assert result['full_control_pass'] and result['archived_q0_exact_replay'] and result['single_call_outputs_unchanged']
    audit={'decoded_lossless_native_points':len(decoded),'decoded_arrays':sum(len(v) for v in decoded.values()),
        'all_array_original_bit_hashes_verified':True,'state_chain_relative_report_max_difference':maximum_report_difference,
        'independent_full_output_slope_chain_relative':independent_output_relative,
        'native_reported_output_chain_relative':result['weighted_logit_chain_relative'],
        'all_seven_state_groups_pass':True,'caller_mutation_fields':18,'model_calls':0,
        'classification':'independent arithmetic on archived old-calibration repair; fresh held confirmation remains pending'}
    (out/'INDEPENDENT_REPAIR_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    PRIVATE.mkdir(exist_ok=True)
    for row in verified:
        dest=PRIVATE/row['name']
        assert not dest.exists()
        with Path(row['host']).open('rb') as source,dest.open('xb') as output:shutil.copyfileobj(source,output,8*1024**2)
        assert dest.stat().st_size==row['bytes'] and sha(dest)==row['sha256']
        with zipfile.ZipFile(dest) as z:assert z.testzip() is None
        row.update(private_Drive_path=str(dest),private_byte_and_CRC_readback_verified=True)
    receipt={'host_transfer_complete':True,'private_full_readback_verified':True,'provider_metadata_verified':False,
        'private_parent':'1C2uSUGzDOrm5CEbqJjDP6o_oxo-OJTIE','files':verified,'audit':audit,
        'phase17_calls':8,'remaining_phase17_calls':292,'phase17_prior_charged_seconds':transfer['prior_phase17_seconds']+transfer['postprocessing_seconds'],
        'fresh_model_calls':0,'model_worker_exited':True,'GPU_allocations':0}
    (BASE/'REPAIR_FINAL_BACKUP_RECEIPT_20261014.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
