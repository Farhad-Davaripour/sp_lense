"""Private owned-process bootstrap. No model/gate imports before admission."""
import json,os,sys,threading,time
from pathlib import Path
from core import HERE,Budget,read,require,sha
from owned import strict_json
from native import source

def handshake(lane,output):
    watchdog=threading.Timer(9.,lambda:os._exit(124));watchdog.daemon=True;watchdog.start()
    try:
        raw=sys.stdin.buffer.readline(8193);require(len(raw)<=8192 and raw.endswith(b'\n'),'challenge timeout/pipe loss')
        nonce=strict_json(raw)['nonce'];require(isinstance(nonce,str) and len(nonce)==64,'private nonce')
        kernel,_=source.api();identity=source.identity(kernel.GetCurrentProcess())
        print(json.dumps({'kind':'claim','nonce':nonce,'pid':os.getpid(),'ppid':os.getppid(),'sys_executable':sys.executable,
            'base_executable':sys._base_executable,'os_identity':identity}),flush=True)
        raw=sys.stdin.buffer.readline(8193);require(len(raw)<=8192 and raw.endswith(b'\n'),'permission timeout/pipe loss')
        envelope=strict_json(raw)
        require(envelope['ownership_only']=={'may_load':True,'nonce':nonce,'scope':'FAKE_WORK_ONLY'},'ownership attestation is not production permission')
        require(envelope['lane']==lane and envelope['output']==str(output),'exact lane/output')
        real=lane in ('worker','audit')
        if real:
            require(envelope['execution_scope']=='PRODUCTION_F03_V2_NONSELF' and envelope['root_release_sha256'] is not None,'separate production execution authority')
            from admission import admit_production,admit_audit
            plan,release=admit_production() if lane=='worker' else admit_audit()
            require(envelope['root_release_sha256']==read(HERE/'authorization.json')['root_release_sha256'],'root pin independently admitted')
            require(output==(HERE/'real_attempt' if lane=='worker' else HERE/'real_attempt/audit'),'fixed real output')
        else:
            require(lane in ('synthetic_worker','synthetic_audit','fake_deadline') and envelope['execution_scope']=='SYNTHETIC_ONLY'
                    and envelope['root_release_sha256'] is None and output.is_relative_to(HERE/'synthetic'),'fake lane cannot enable real execution')
        Budget(output).write('BOOTSTRAP.json',{'actual_pid':os.getpid(),'parent_pid':os.getppid(),'nonce':nonce,'lane':lane,
            'execution_scope':envelope['execution_scope'],'root_release_sha256':envelope['root_release_sha256'],
            'owned_handle_permission_received':True,'model_imports_before_permission':False})
        return envelope
    finally:watchdog.cancel()

def main():
    require(len(sys.argv)==3,'fixed lane and owned output');lane=sys.argv[1];output=Path(sys.argv[2]).resolve()
    envelope=handshake(lane,output)
    if lane=='fake_deadline':
        timer=threading.Timer(9.,lambda:os._exit(124));timer.daemon=True;timer.start()
        print('FAKE_READY_NO_LOADER',flush=True);os.close(1);os.close(2);threading.Event().wait(9.);return 124
    if lane=='worker':
        from run import production_worker
        production_worker();return 0
    if lane=='audit':
        from judge import finish_judge
        return finish_judge(output.parent,synthetic=False)
    if lane=='synthetic_worker':
        from fake_batch import synthetic_worker
        synthetic_worker(output);return 0
    if lane=='synthetic_audit':
        from judge import finish_judge
        return finish_judge(output.parent,synthetic=True)
    raise ValueError('unrecognized bootstrap lane')
if __name__=='__main__':raise SystemExit(main())
