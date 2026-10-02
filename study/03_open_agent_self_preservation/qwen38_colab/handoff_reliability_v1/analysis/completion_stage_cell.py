# Complete only unfinished groups from the unchanged frozen diagnostic plan.
import os,shutil
PRIOR_DIAGNOSTIC_ROOT=ROOT
previous_result=PRIOR_DIAGNOSTIC_ROOT/'reports/RESULT.json'
if not (previous_result.exists() and json.loads(previous_result.read_text()).get('completed')):
    assert json.loads((PRIOR_DIAGNOSTIC_ROOT/'reports/CONTROLLER_RECEIPT.json').read_text())['worker_exited']
    COMPLETION_STARTED=time.monotonic()
    COMPLETION_CAP_UNITS=8
    COMPLETION_RESERVE_UNITS=2
    COMPLETION_RATE=6.77
    COMPLETION_BALANCE_OBSERVED=114.02
    PRIOR_SPEND_CONSERVATIVE=89.30
    assert COMPLETION_CAP_UNITS<=min(COMPLETION_BALANCE_OBSERVED,200-PRIOR_SPEND_CONSERVATIVE)
    COMPLETION_ROOT=PRIOR_DIAGNOSTIC_ROOT.parent/('qwen38_handoff_completion_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])
    COMPLETION_ROOT.mkdir(exist_ok=False)
    shutil.copytree(PRIOR_DIAGNOSTIC_ROOT/'source',COMPLETION_ROOT/'source',ignore=shutil.ignore_patterns('__pycache__'))
    COMPLETION_DRIVER_URL='https://raw.githubusercontent.com/Farhad-Davaripour/sp_lense/f14a1636ef289ae77ab7890e9d27288355ce25d0/study/03_open_agent_self_preservation/qwen38_colab/handoff_reliability_v1/analysis/completion_worker.py'
    COMPLETION_DRIVER_SHA='bbdd2f96bfb540ae62d473cf551b4c2f8c1c0f3a287315b781b0c5d1294c5de3'
    from urllib.request import urlopen
    with urlopen(COMPLETION_DRIVER_URL,timeout=120) as response: completion_driver=response.read()
    assert hashlib.sha256(completion_driver).hexdigest()==COMPLETION_DRIVER_SHA
    helper=COMPLETION_ROOT/'completion/worker.py';helper.parent.mkdir()
    helper.write_bytes(completion_driver)
    elapsed=time.monotonic()-COMPLETION_STARTED
    seconds=min(3180,(COMPLETION_CAP_UNITS-COMPLETION_RESERVE_UNITS)/COMPLETION_RATE*3600-elapsed)
    assert seconds>=1800,'Completion not admitted with meaningful work/export reserve'
    (COMPLETION_ROOT/'reports').mkdir()
    (COMPLETION_ROOT/'COMPLETION_STAGE_FREEZE.json').write_text(json.dumps({
        'prior_root':str(PRIOR_DIAGNOSTIC_ROOT),'driver_revision':'f14a1636ef289ae77ab7890e9d27288355ce25d0',
        'driver_sha256':COMPLETION_DRIVER_SHA,'frozen_source_sha256':hashlib.sha256((COMPLETION_ROOT/'source/SOURCE_FREEZE.json').read_bytes()).hexdigest(),
        'authorized_total_units':200,'prior_spend_conservative':PRIOR_SPEND_CONSERVATIVE,
        'account_observed':COMPLETION_BALANCE_OBSERVED,'rate':COMPLETION_RATE,
        'stage_cap_units':COMPLETION_CAP_UNITS,'reserve_units':COMPLETION_RESERVE_UNITS,
        'main_seconds':seconds,'maximum_resident_model_workers':1,'parameter_updates':0,
        'rule':'Only missing fixed groups; complete groups reused regardless outcome; no new cases or second fresh repeat'},indent=2))
    completion_env=os.environ.copy()
    completion_env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_TELEMETRY='1',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0')
    completion_log=COMPLETION_ROOT/'training/logs/worker.log';completion_log.parent.mkdir(parents=True)
    with completion_log.open('w') as log:
        process=subprocess.Popen([sys.executable,str(helper),'--prior-root',str(PRIOR_DIAGNOSTIC_ROOT),
            '--root',str(COMPLETION_ROOT),'--model',str(PRIOR_DIAGNOSTIC_ROOT/'model'),'--max-seconds',str(seconds)],
            stdout=log,stderr=subprocess.STDOUT,env=completion_env,cwd=COMPLETION_ROOT/'source',start_new_session=True)
        started=time.monotonic();cursor=0
        try:
            while process.poll() is None and time.monotonic()-started<seconds:
                time.sleep(15)
                text=completion_log.read_text(errors='replace')
                if len(text)>cursor:print(text[cursor:],end='',flush=True);cursor=len(text)
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=30)
        finally:
            if process.poll() is None:process.kill();process.wait(timeout=30)
        print(completion_log.read_text(errors='replace')[cursor:],end='',flush=True)
    (COMPLETION_ROOT/'reports/CONTROLLER_RECEIPT.json').write_text(json.dumps({'worker_exited':True,'exit_code':process.returncode,'seconds':time.monotonic()-started,'cap_seconds':seconds,'maximum_resident_model_workers':1},indent=2))
    print('COMPLETION_WORKER_EXITED',process.returncode,COMPLETION_ROOT,flush=True)
    DIAGNOSTIC_EXPORT_ROOTS=[PRIOR_DIAGNOSTIC_ROOT,COMPLETION_ROOT]
    ROOT=COMPLETION_ROOT
