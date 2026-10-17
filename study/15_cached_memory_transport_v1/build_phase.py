"""Freeze the actual memory-transport continuation before any T2 outcomes."""
import ast,hashlib,json,shutil,textwrap,zipfile
from datetime import datetime,timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=Path('C:/Users/farha/Research4/artifacts/cached_memory_transport_v1')
ROOT=BASE/'prepared_20261014'
PARENT=Path('C:/Users/farha/Research4/artifacts/cached_state_response_v1')
RAW=PARENT/'R4_CACHED_STATE_NATIVE_20261014_COLAB_CPU_RESUME1.zip'
PARENT_SHA='55717a3be9d192ac984187dcfc42e061fd7763a157e53f943185ca08bdaed3a0'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    proof=json.loads((PARENT/'COLAB_DURABLE_RAW_BACKUP_RECEIPT_20261014.json').read_text(encoding='utf-8'))
    assert proof['complete'] and proof['private_Drive_provider_metadata_verified'] and proof['files'][0]['sha256']==PARENT_SHA
    ROOT.mkdir(parents=True,exist_ok=False);src=ROOT/'source';src.mkdir()
    with zipfile.ZipFile(RAW) as z:
        parent_manifest=json.loads(z.read('EXPORT_HASHES.json'))
        used=['INPUTS.json','DIRECTIONS.npz','source/base_native.py','source/native.py','source/model_pin.json','source/PLAN.json']
        for n in used:
            data=z.read(n);assert hashlib.sha256(data).hexdigest()==parent_manifest[n]
            target=ROOT/n
            if n=='source/PLAN.json':target=ROOT/'PARENT_PLAN.json'
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        parent_plan=json.loads(z.read('source/PLAN.json'))
        receipt=json.loads(z.read('STATUS.json'));assert receipt['model_calls']==166 and receipt['status']=='native_complete'
    for n in ['transport_native.py','worker.py','analyze.py']:shutil.copyfile(HERE/n,src/n)
    shutil.copyfile(HERE.parent/'14_cached_state_response_v1/maths.py',src/'maths.py')
    original=(HERE.parent/'14_cached_state_response_v1/worker.py').read_text(encoding='utf-8')
    tree=ast.parse(original)
    funcs={n.name:n for statement in tree.body if isinstance(statement,ast.Try) for n in statement.body if isinstance(n,ast.FunctionDef)}
    (src/'reused_helpers.py').write_text('\n\n'.join(ast.unparse(funcs[n]) for n in ['save_value','save_probe','primal_ok','jet_from','save_jet','forecasts'])+'\n',encoding='utf-8')
    selection=original.split('    SELECTION=None;candidate_checks=[]\n',1)[1].split('    def jet_from(q,entry):',1)[0]
    (src/'reused_numerical_selection.py').write_text('SELECTION=None;candidate_checks=[]\n'+textwrap.dedent(selection),encoding='utf-8')
    plan=json.loads((HERE/'PLAN.json').read_text(encoding='utf-8'))
    plan['status']='frozen before native execution; parent preserved, T2 outcomes unseen'
    plan['amendment']={
      'reason':'Use the fixed300-call continuation budget to test complete memory transport, chain-rule components and predicted direction interactions; restore correct EOS handling from the pinned model and tokenizer.',
      'prepared_old_plan_retained':True,'T2_outcomes_seen':0,
      'held_primary_unchanged':'All32 signed primary targets on the four known held questions at a new prediction position remain required, with unchanged strengths and accuracy limits.',
      'calibration_endpoint_reallocation':'The proposed16 calibration targets are diagnostic only and did not select strengths or methods. They fund direct chain controls and a full mixed-plane forecast test. Any spare cal targets are locked before observing endpoints.',
      'parent_probe_availability':'If calibration selects the finer pair requiring16 missing held first-step probes, collect these natively and count them. Reduce optional interaction coverage, never held primary coverage or the300-call ceiling.',
      'calibration_only_allocation':True}
    plan['parent_root']='/content/research4/artifacts/cached_state_response_v1/colab_cpu_20261014_resume1'
    plan['parent_archive_sha256']=PARENT_SHA
    plan['parent_archive_private_id']='1jXhhHZD9qgrAJPode7q0PIMU8q5FRCYY'
    plan['checkpoint']='/content/r4_checkpoint';plan['parent_seconds']=622.9496870979992;plan['parent_calls']=166
    plan['revision']=parent_plan['revision'];plan['system']=parent_plan['system']
    plan['numeric_rules']=parent_plan['numerical_gates'];plan['primary_rules']=parent_plan['primary_support']
    plan['primary']['signed_endpoints']=32;plan['primary']['calibration_finite_targets']='optional; allocation locked before endpoints'
    plan['input']+=' Treat both pinned config EOS248044 and tokenizer EOS248046 as terminal; force760 in either case, and disclose all fallbacks.'
    plan['chain_control']={'question':'r14_cal_window','direction':0,'equation':'r2 = J_memory*r1 + H_memory[q1,q1]',
      'native_components':'One forward-AD seed r1 at normal M1; two slopes on each synthetic straight-memory path M1+/-h*q1, tangent q1. Same original decoder operations.',
      'checks':'10% relative or1e-6 absolute per original state group and weighted full logits; record H coarse/fine agreement. These auxiliary memories are synthetic local derivative probes, not free-running histories.',
      'common_pair_calls':5,'finer_pair_calls':3,'failure':'Retain failure and all primary forecasts; include the chain check as a primary-support condition, without tuning.'}
    plan['cubic_secondary']={'classification':'Prospective new-position test motivated by study16 saved-data lead; known questions and older directions, not fresh wording',
      'equation':'Y2(0)+alpha*q2+alpha^2*r2/2+alpha^3*t2/6',
      'third':'[q2(+h)+q2(-h)-2q2(0)]/h^2 from the same slope probes; fine estimate only',
      'calibration_lock':'Requires all groups slope_difference and coarse/fine third agreement<=10% or1e-6 absolute on all calibration jets; weighted logits<=10%. Never choose on held endpoints.',
      'held_failure':'Keep frozen cubic predictions and flag any numerical failure; never exclude or retune.',
      'gates':'Same signed cost/logit/group/field limits and comparator improvements as primary quadratic; outcomes reported separately',
      'extra_model_calls':0}
    plan['cubic_secondary']['stronger_full_state_gate']='Require all7 original state groups<=10% and every resolved recorded field<=15%, including attention/MLP and interface fields; unchanged upstream fields are controls, not accuracy evidence.'
    plan['mixed_secondary']={'calibration_question':'r14_cal_window','held_question':'r14_test_notes',
      'sum_first':'Assemble q2_sum(0)=(q20+q21)/sqrt2; check inherited first-step sum tangent linearity and new sum finite-difference slopes.',
      'sum_probes':'Four new second-step slope calls at the existing first-step sum +/-coarse/fine memories per context:8 calls.',
      'cross':'H01_total=r2_sum-0.5*(r20+r21), including both first-step memory interaction and second-step response interaction',
      'targets':'One calibration sum+0.125, and held sum/difference/weighted directions both signs0.125:7 calls; all plane coefficients frozen first.',
      'directions':'sum=(v0+v1)/sqrt2; difference=(v0-v1)/sqrt2; weighted=(2*v0+v1)/sqrt5',
      'forecast':'Full quadratic plane versus no-cross diagonal comparator; difference and weighted directions not differentiated at T2',
      'gates':'Inherited per-endpoint output/state bounds on all6 held targets and>=25% median weighted-error improvement over diagonal; no per-target improvement required',
      'scope':'Only one held question at a new position; no full-space or semantic interaction claim. No cubic forecast of unprobed mixed directions without a full third-order plane tensor.',
      'finer_pair_fallback':'One held positive sum target scored against a frozen diagonal model only; no full interaction-support claim.'}
    plan['budget'].update(primary_actual_calls=32,planned_primary_additional_calls=114,planned_total_calls=280,
      possible_missing_first_step_probes=16,common_pair_chain_calls=5,common_pair_plane_probes=8,common_pair_plane_targets=7,
      common_pair_total=300,finer_pair_chain_calls=3,finer_pair_single_mixed_target=1,finer_pair_total=300,
      optional_stress_calls=0,maximum_total_calls=300)
    plan['optional_cubic_output_remedy']={'enabled':False,'replaced_by':'prospective cubic secondary at unchanged primary strengths'}
    plan['readout_extra_controls']={'previous_KV':'Exact primal and tangent carry-through checks; separate current-KV-only error bound<=10% avoids masking new errors with copied history',
      'injection':'Current b11 input must be exactly unchanged; steering delivery is nonzero changed incoming memory.',
      'timing':'Measure actual value/AD calls and deterministic full-field/logit/softmax forecast evaluation. Conditional reuse timing, not end-to-end generation/compression speedup.'}
    (ROOT/'PREPARED_ORIGINAL_PLAN.json').write_bytes((HERE/'PLAN.json').read_bytes())
    (src/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    (ROOT/'PARENT_IDENTITY.json').write_text(json.dumps({'sha256':PARENT_SHA,'private_provider_id':plan['parent_archive_private_id'],
      'manifest_sha256':hashlib.sha256(json.dumps(parent_manifest,sort_keys=True).encode()).hexdigest(),
      'parent_calls':166,'parent_charged_seconds':plan['parent_seconds'],'parent_raw_backup_complete':True},indent=2)+'\n',encoding='utf-8')
    for p in src.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    files=sorted(p for p in ROOT.rglob('*') if p.is_file())
    (ROOT/'EXECUTION_FREEZE.json').write_text(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'new_model_calls_before_freeze':0,
      'parent_calls':166,'T2_outcomes_seen':0,'sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files}},indent=2)+'\n',encoding='utf-8')
    archive=BASE/'R4_MEMORY_TRANSPORT_SOURCE_20261014.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as z:
        for p in ROOT.rglob('*'):
            if p.is_file():z.write(p,'colab_cpu_20261014/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps({'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'new_model_calls':0,'combined_forward_ceiling':300}))

if __name__=='__main__':main()
