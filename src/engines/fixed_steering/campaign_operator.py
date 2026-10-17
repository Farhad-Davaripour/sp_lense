"""Trusted-parent study orchestration. Credential stays in the supplied broker.

A caller must admit the entire window from current credit/rate and retained API
usage. This module never discovers credentials, changes allowances, buys credit,
or retries an incomplete provider/model job. It preserves all scheduled outcomes.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from campaign_classifier import save
from campaign_analysis import analyze


class StudyOperator:
    def __init__(self,root,source,model,adapter,broker,environment,deadline):
        self.root,self.source,self.model,self.adapter=map(Path,(root,source,model,adapter))
        self.broker,self.environment,self.deadline=broker,environment,deadline
        self.processes=[]
        self.fault=None
        self.stop=threading.Event()
        self.broker_thread=None

    def status(self,phase,**data):
        save(self.root/'reports/STUDY_STATUS.json',{'phase':phase,'utc_seconds':time.time(),
            'seconds_remaining':round(self.deadline-time.monotonic()),
            'cumulative_API':self.broker.totals(),**data})

    def guard(self,reserve=0):
        if self.fault:
            raise RuntimeError('Trusted broker fault; no automatic recovery')
        if time.monotonic()+reserve>=self.deadline:
            raise TimeoutError('Admitted complete-study window ended')

    def pump(self):
        while not self.stop.is_set():
            try:
                self.broker.pump()
            except Exception as error:
                self.fault={'type':type(error).__name__,'no_automatic_retry':True}
                save(self.root/'reports/STUDY_BROKER_FAULT.json',self.fault)
                self.stop.set()
                return
            self.stop.wait(.1)

    def launch(self,mode,adapter=False):
        self.guard(900)
        seconds=int(self.deadline-time.monotonic()-900)
        command=[sys.executable,str(self.source/'campaign_worker.py'),'--root',str(self.root),
            '--model',str(self.model),'--mode',mode,'--max-seconds',str(seconds)]
        if adapter:
            command+=['--adapter',str(self.adapter)]
        label=mode+('_A' if adapter else '_primary')
        path=self.root/(label+'.log')
        if path.exists():
            raise ValueError('Never silently rerun/replace an earlier phase log')
        log=path.open('wb')
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
            env=self.environment,start_new_session=True)
        self.processes.append((process,log,label))
        save(self.root/'reports'/('STARTED_'+label+'.json'),{'pid':process.pid,'max_seconds':seconds,
            'mode':mode,'secondary_A':bool(adapter),'no_credential_in_worker_environment':True})
        return process

    def wait_exit(self,process):
        while process.poll() is None:
            self.guard(900)
            time.sleep(.2)
        if process.returncode!=0:
            raise RuntimeError('Model worker exited incompletely; preserve its fault receipt')

    def wait_file(self,path,process):
        while not path.exists():
            self.guard(900)
            if process.poll() is not None:
                raise RuntimeError('Model worker exited before expected receipt')
            time.sleep(.2)
        return json.loads(path.read_bytes())

    def run(self):
        self.broker.deadline=self.deadline
        self.broker_thread=threading.Thread(target=self.pump,daemon=True)
        self.broker_thread.start()
        try:
            self.status('development_replay_and_calibration')
            calibrator=self.launch('calibrate')
            self.wait_exit(calibrator)
            calibration=self.root/'reports/CALIBRATION_COMPLETE.json'
            if not calibration.exists():
                raise ValueError('Missing real calibration/closed-loop development receipt')
            freeze=self.root/'steering/INTERVENTION_FREEZE.json'
            intervention_sha=hashlib.sha256(freeze.read_bytes()).hexdigest()
            # Seal exactly the executed methods before opening confirmation.
            names=('campaign_worker.py','campaign_generation.py','campaign_classifier.py',
                   'campaign_scoring.py','campaign_analysis.py','actuators.py')
            source_freeze={'sha256':{n:hashlib.sha256((self.source/n).read_bytes()).hexdigest() for n in names},
                           'intervention_sha256':intervention_sha,'confirmation_started':False}
            save(self.root/'reports/CONFIRMATION_SOURCE_FREEZE.json',source_freeze)
            self.status('loading_primary_and_secondary_A',intervention_sha256=intervention_sha)
            primary=self.launch('serve')
            self.wait_file(self.root/'reports/READY_primary.json',primary)
            secondary=self.launch('serve',adapter=True)
            self.wait_file(self.root/'reports/READY_A.json',secondary)
            cases={c['id']:c for c in json.loads((self.root/'confirmation_cases.json').read_bytes())}
            schedule=json.loads((self.root/'confirmation_order.json').read_bytes())
            completed=0
            for row in schedule:
                for condition in row['conditions']:
                    self.guard(900)
                    if hashlib.sha256(freeze.read_bytes()).hexdigest()!=intervention_sha:
                        raise ValueError('Intervention changed after confirmation began')
                    for name,expected in source_freeze['sha256'].items():
                        if hashlib.sha256((self.source/name).read_bytes()).hexdigest()!=expected:
                            raise ValueError('Frozen confirmation source changed')
                    channel='A' if condition=='A_no_jev' else 'primary'
                    job_name=f'{completed:03d}_{channel}.json'
                    save(self.root/'jobs'/channel/job_name,{'case':cases[row['case_id']],
                        'condition':condition,'intervention_sha256':intervention_sha})
                    self.status('confirmation',completed_conditions=completed,total_conditions=96,
                                current_case=row['case_id'],current_condition=condition)
                    receipt=self.wait_file(self.root/'job_receipts'/job_name,secondary if channel=='A' else primary)
                    if not receipt['completed']:
                        raise ValueError('Incomplete scheduled rollout is not a success')
                    completed+=1
            (self.root/'jobs/STOP').write_text('All 96 scheduled conditions finished.\n')
            self.wait_exit(primary)
            self.wait_exit(secondary)
            results=analyze(self.root)
            if not results['all_96_rollouts_complete']:
                raise ValueError('Scheduled confirmation outcomes missing')
            self.status('analysis_ready_export_and_final_review_pending',completed_conditions=96,
                confirmation_complete=True,final_closeout_complete=False)
        except Exception as error:
            self.status('incomplete',failure={'type':type(error).__name__,'message':str(error)[:240]},
                no_automatic_rerun=True)
            raise
        finally:
            self.stop.set()
            for process,log,label in self.processes:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL)
                        process.wait(timeout=10)
                log.close()
            if self.broker_thread:
                self.broker_thread.join(timeout=35)
            self.broker.close()
