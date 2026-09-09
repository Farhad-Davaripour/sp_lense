"""Storage-only revision checks; fake marker/tokenizer fixtures, never real encoding."""
import json,sys,time
from pathlib import Path
import test_prepare as old
from storage import Publisher,entry_failure,error_fields,PREPARATION_BYTES,TERMINAL_RESERVE,TERMINAL_BYTES,OWNER_BYTES,COMBINED_BYTES,FILE_BYTES,jb,need
from prepare_core import execute
from plan import PREPARATION
HERE=Path(__file__).resolve().parent
def used(p):return sum(x.stat().st_size for x in p.iterdir() if x.is_file())
def fill(pub,base,total,*,critical=False):
    i=0
    while used(base)<total:
        amount=min(FILE_BYTES,total-used(base));pub.write('padding_'+str(i),b'x'*amount,raw=True,critical=critical);i+=1
def main():
    root=HERE/'test_evidence'/('storage_'+str(time.time_ns()));root.mkdir(parents=True);reports=[]
    need(PREPARATION['total_bytes']==COMBINED_BYTES and PREPARATION['owner_reserved_bytes']==OWNER_BYTES
        and PREPARATION['preparation_bytes']==PREPARATION_BYTES and PREPARATION['terminal_file_bytes']==TERMINAL_BYTES,'FIXED_PARTITION_METADATA')
    base=root/'boundary';base.mkdir();pub=Publisher(base);fill(pub,base,PREPARATION_BYTES-TERMINAL_RESERVE)
    need(used(base)==PREPARATION_BYTES-TERMINAL_RESERVE,'EXACT_NONCRITICAL_BOUNDARY')
    try:pub.write('one_more',b'x',raw=True)
    except ValueError as e:need(str(e)=='PREPARATION_TOTAL_CAP','NONCRITICAL_ONE_BYTE_REASON')
    else:raise RuntimeError('NONCRITICAL_OVERFLOW')
    need(not (base/'one_more').exists(),'NO_OVERFLOW_WRITE')
    pub.write('RESULT.json',b' '*TERMINAL_BYTES,raw=True,critical=True)
    pub.write('remaining_critical',b'x'*(TERMINAL_RESERVE-TERMINAL_BYTES),raw=True,critical=True)
    need(used(base)==PREPARATION_BYTES,'EXACT_CRITICAL_BOUNDARY')
    try:pub.write('critical_extra',b'x',raw=True,critical=True)
    except ValueError as e:need(str(e)=='PREPARATION_TOTAL_CAP','CRITICAL_ONE_BYTE_REASON')
    else:raise RuntimeError('CRITICAL_OVERFLOW')
    need(not (base/'critical_extra').exists(),'CRITICAL_NO_WRITE')
    reports.append({'case':'exact_noncritical_and_critical_partition_one_byte_refusal','status':'PASS','preparation_bytes':used(base)})
    empty=root/'terminal';empty.mkdir();terminal=Publisher(empty)
    try:terminal.write('RESULT.json',b'x'*(TERMINAL_BYTES+1),raw=True,critical=True)
    except ValueError as e:need(str(e)=='BOUNDED_TERMINAL_RESULT','TERMINAL_BOUND_REASON')
    else:raise RuntimeError('TERMINAL_OVERSIZE')
    need(not (empty/'RESULT.json').exists(),'TERMINAL_REFUSED_BEFORE_WRITE')
    reports.append({'case':'terminal_8193_refused_before_write','status':'PASS'})
    Huge=type('X'*100000,(Exception,),{});error=Huge('A'*1000000)
    fields=error_fields(error);need(fields=={'error_type':'Exception','error_code':'PREPARATION_FAILURE'},'BOUNDED_HUGE_ERROR')
    class Broken(Exception):
        def __str__(self):raise RuntimeError('BROKEN_STR')
    need(error_fields(Broken())['error_code']=='PREPARATION_FAILURE','BROKEN_ERROR_STRING')
    terminal.write('ADMISSION.json',{'synthetic_only':True})
    fill(terminal,empty,PREPARATION_BYTES-TERMINAL_RESERVE)
    entry_failure(empty,error);entry=json.loads((empty/'RESULT.json').read_bytes())
    need(entry['status']=='PREPARATION_ENTRY_FAILURE' and entry['unrun_operations']==313
        and len((empty/'RESULT.json').read_bytes())<512 and used(empty)<=PREPARATION_BYTES,'ENTRY_FAILURE_INSIDE_RESERVED_SPACE')
    reports.append({'case':'large_exception_type_message_and_entry_failure_terminal_reserve','status':'PASS'})
    partial=root/'entry_after_journal';partial.mkdir();Publisher(partial).write('operations.jsonl',{'ordinal':1,'status':'STARTED'})
    entry_failure(partial,error);unknown=json.loads((partial/'RESULT.json').read_bytes())
    need(unknown['attempted_operations'] is None and unknown['unrun_operations'] is None
        and unknown['operation_counts_status']=='UNKNOWN_SEE_RETAINED_JOURNAL','NO_FALSE_ZERO_AFTER_JOURNAL')
    reports.append({'case':'entry_failure_with_journal_preserves_unknown_counts','status':'PASS'})
    lock=old.fixture();clock=old.Clock();tok=old.FakeTokenizer(lock)
    result=execute(lock,root/'complete24',lambda:tok,180.,allow_synthetic=True,
        template_sha256=old.sha(tok.chat_template.encode()),clock=clock)
    need(result['status']=='PASS' and result['completed_operations']==313 and result['completed_cases']==24,'UNCHANGED_FAKE313_INTERFACE')
    need((root/'complete24/RESULT.json').stat().st_size<=TERMINAL_BYTES,'SUCCESS_RESULT_BOUND')
    def fail():raise error
    failed=execute(lock,root/'large_failure',fail,180.,allow_synthetic=True,clock=old.Clock())
    need(failed['status']=='FAIL' and failed['attempted_operations']==1 and failed['unrun_operations']==312
        and failed['error_code']=='PREPARATION_FAILURE' and failed['error_type']=='Exception'
        and not (root/'large_failure/inputs.json').exists() and (root/'large_failure/RESULT.json').stat().st_size<1024,'CORE_TERMINAL_HUGE_FAILURE')
    reports.append({'case':'fake313_success_and_large_failure_without_method_change','status':'PASS'})
    owner_dir=HERE.parent/'native_final_preparation_owner_v1';sys.path.insert(0,str(owner_dir))
    from owner import owner_write,OWNER_CAP,PREPARATION_CAP
    own=root/'owner';own.mkdir()
    for name,n in (('ADMISSION.json',4096),('stdout.bin',8192),('stderr.bin',8192),('CLOSURE.json',12288)):
        owner_write(own,name,b'x'*n,raw=True)
    need(used(own)==OWNER_CAP==OWNER_BYTES and PREPARATION_CAP==PREPARATION_BYTES
        and used(base)+used(own)==COMBINED_BYTES,'EXACT_COMBINED16MIB')
    original=(own/'ADMISSION.json').read_bytes()
    try:owner_write(own,'ADMISSION.json',b'x',raw=True)
    except ValueError as e:need(str(e)=='OWNER_PREWRITE_TOTAL_CAP','OWNER_ONE_BYTE_REASON')
    else:raise RuntimeError('OWNER_OVERFLOW')
    need((own/'ADMISSION.json').read_bytes()==original,'OWNER_OVERFLOW_NO_OVERWRITE')
    fresh=root/'owner_file';fresh.mkdir()
    try:owner_write(fresh,'stdout.bin',b'x'*8193,raw=True)
    except ValueError as e:need(str(e)=='OWNER_ARTIFACT_CAP','OWNER_FILE_REASON')
    else:raise RuntimeError('OWNER_CAPTURE_OVERFLOW')
    need(not (fresh/'stdout.bin').exists(),'OWNER_CAPTURE_NO_WRITE')
    reports.append({'case':'owner_exact32KiB_and_combined_exact16MiB_one_byte_refusals','status':'PASS'})
    need(not any(n.split('.')[0] in old.BLOCKED for n in sys.modules),'NO_REAL_PROVIDERS')
    result={'status':'PASS','groups':reports,'group_count':len(reports),'real_tokenizer_calls':0,'model_calls':0,
        'actual_cohort_access':False,'historical_reports_unchanged':True,'test_root':str(root)}
    (HERE/'TEST_STORAGE_RESULTS.json').write_bytes(jb(result));print(json.dumps(result,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
