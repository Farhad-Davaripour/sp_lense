"""Finish prospective source/input/runtime lock; no model, tokenizer or scores."""
import ast,json,time
from core import HERE,ROOT,Budget,read,require,sha
from inputs import build_plan
from admission import environment
def main():
    started=time.monotonic();budget=Budget(HERE)
    require(not (HERE/'freeze.json').exists(),'one prospective final source lock')
    plan=build_plan();require((len(plan['cells']),len(plan['derivative_cells']))==(42,16),'exact conditional ceilings')
    require(len({c['cell_id'] for c in plan['cells']})==42 and sum(c['optional'] for c in plan['cells'])==32,'unique request-aware entire schedule')
    environment(plan)
    parent=ROOT/'diagnostics/windows_owned_worker_supervision_v2';raw=(parent/'FINAL_INVENTORY.json').read_bytes()
    require(sha(raw)=='31c4e13812a4934b97feffc177bae489adadf7e77aedbff9751127003cbd0f61','reviewed owned successor inventory')
    entries={e['path']:e for e in json.loads(raw)['files']}
    for name in ('owned.py','native.py','new_pure_tests.json','reuse_receipt.json','batch_result.json'):
        value=(parent/name).read_bytes();require(sha(value)==entries[name]['sha256'] and len(value)==entries[name]['bytes'],'owned parent artifact')
    for name in ('owned.py','native.py'):require((HERE/name).read_bytes()==(parent/name).read_bytes(),'byte-identical owned core')
    require(len(read(parent/'new_pure_tests.json'))==9 and read(parent/'reuse_receipt.json')['reused_pure_count']==12,'inherited evidence only')
    require(read(parent/'batch_result.json')['status']=='PASS_FAKE_ONLY','owned parent verification')
    budget.write('parent_reuse_receipt.json',{'owned_commit':'1077a38fa140d7dac9f3bf293f700d4a08150c8c','inventory_sha256':sha(raw),
        'reused_pure_identity_fixtures':12,'reused_verdict_fixtures':9,'prior_suites_rerun':False,'owned_native_byte_identical':True,
        'scientific_editor_and_saved_scorers_byte_identical':True})
    budget.write('production_plan.json',plan)
    lengths=[p['prompt_length'] for p in plan['alignment'].values()]
    record_bound=42*(248320*4+1024+262144)+32*1024**2
    require(record_bound<96*1024**2 and lengths==[133,133],'fixed input and conservative storage bound')
    budget.write('resource_proof.json',{'schedule':'2 preflights + 4 fresh entries + 4*(8 updates + 1 endpoint) = 42F / 16D',
        'fresh_routes':6,'endpoints':4,'off_checks':0,'strict_hook_checks':13,'lengths':lengths,
        'conservative_record_bytes':record_bound,'formula':'42*(248320*4+1024+262144)+16MiB hooks+16MiB logs/source/receipts',
        'future_limits':plan['limits'],'file_bytes':5*1024**2,'prep_bytes':32*1024**2,'prep_invoked_seconds':300,
        'runtime_calibration':'Prior40F16D194.516s and42F3D128.641s;300s is finite headroom, no extension or fake throughput claim',
        'conditional_retention_reductions_are_not_the_ceiling':True,'base_runtime_change_explicitly_bound':True})
    for path in HERE.glob('*.py'):ast.parse(path.read_bytes(),filename=path.name)
    names=sorted(p.name for p in HERE.iterdir() if p.is_file() and p.name not in ('authorization.json','approved_root_release.json','freeze.json'))
    budget.write('freeze.json',{'source_sha256':{name:sha((HERE/name).read_bytes()) for name in names},
        'scope':'f04 self-only entire pair and owned supervision, fake preparation only','production_authorized':False,
        'fake_batch':'eight pure changed-path checks, one complete four-request synthetic traversal and independent audit, one live deadline capture',
        'one_batch_no_retry':True})
    print(json.dumps({'status':'PROSPECTIVE_LOCKED','source_files':len(names),'elapsed_seconds':time.monotonic()-started,
        'freeze_sha256':sha((HERE/'freeze.json').read_bytes()),'input_lock_sha256':sha((HERE/'input_lock.json').read_bytes()),'models':0,'tokenizers':0,'real_gate_scores':0}))
if __name__=='__main__':main()
