"""Explicit-workspace CLI operations. Credentials remain transient environment values."""
import argparse,base64,io,json,os,re,subprocess,tarfile,tempfile,hashlib
from pathlib import Path
from .io_utils import dump,read,sha
HOST='https://dbc-ff9672a4-ba5f.cloud.databricks.com'
PREFIX='/Workspace/Shared/research4-semantic-gate'
GIT_FOLDER_ID=316719975858080
CLI=os.environ.get('DATABRICKS_CLI_PATH','C:/Users/farha/.codex/tools/databricks-cli/1.21.0/databricks.exe')
def cli(args,cwd=None,allow_fail=False,json_output=True):
    if os.environ.get('DATABRICKS_HOST')!=HOST or not os.environ.get('DATABRICKS_TOKEN'):raise ValueError('Use the explicit workspace and transient token')
    p=subprocess.run([CLI,*args,*(['--output','json'] if json_output else [])],cwd=cwd,capture_output=True,text=True)
    if p.returncode:
        error=re.sub(r'dapi[0-9a-fA-F]+','[REDACTED]',p.stderr)
        if allow_fail:return {'error':error,'exit_code':p.returncode}
        raise RuntimeError(error)
    if not json_output:return p.stdout
    return json.loads(p.stdout) if p.stdout.strip() else {}
def git(repo,*args):return subprocess.check_output(['git',*args],cwd=repo).decode().strip()

def workspace_path(path):
    # Repos API returns /Shared/... even when its input is /Workspace/Shared/....
    return path[len('/Workspace'):] if path.startswith('/Workspace/') else path
def provision(artifact):
    artifact=Path(artifact);artifact.mkdir(parents=True,exist_ok=True)
    existing=cli(['schemas','get','workspace.research4'],allow_fail=True)
    if 'error' in existing:
        p=artifact/'schema.json';dump(p,{'catalog_name':'workspace','name':'research4','comment':'Research 4 semantic state-aware adapter pilot only'})
        cli(['schemas','create','--json','@'+str(p)])
    cli(['workspace','mkdirs',PREFIX+'/experiments'])
def prepare_release(repo,artifact):
    repo=Path(repo);artifact=Path(artifact);commit=git(repo,'rev-parse','HEAD');branch=git(repo,'branch','--show-current')
    if branch!='study/04-semantic-state-gated-lora':raise ValueError('Refuse to deploy an unrelated branch')
    remote=git(repo,'ls-remote','origin','refs/heads/'+branch).split()[0]
    if remote!=commit:raise ValueError('Push the exact local source commit before deployment')
    subprocess.check_call(['git','fetch','origin',branch],cwd=repo,stdout=subprocess.DEVNULL)
    if git(repo,'rev-parse','FETCH_HEAD')!=commit:raise ValueError('Remote source moved during release preparation')
    stage=artifact/('release_'+commit);stage.mkdir(exist_ok=False)
    data=subprocess.check_output(['git','archive',commit+':research4'],cwd=repo)
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        for m in t.getmembers():
            dest=(stage/m.name).resolve()
            if stage.resolve() not in dest.parents and dest!=stage.resolve():raise ValueError('Unsafe release archive path')
            if m.isdir():dest.mkdir(parents=True,exist_ok=True)
            elif m.isfile():dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(t.extractfile(m).read())
    files=[p for folder in ['src','config','ddl'] for p in (stage/folder).rglob('*') if p.is_file()]
    release={'commit':commit,'branch':branch,'repository':'https://github.com/Farhad-Davaripour/sp_lense.git','files':{p.relative_to(stage).as_posix():sha(p) for p in files},'notebooks_verified_separately_after_deployment':True}
    dump(stage/'RELEASE.json',release)
    return stage,release
def deploy(repo,artifact):
    artifact=Path(artifact);provision(artifact);stage,release=prepare_release(repo,artifact);commit=release['commit'];code=PREFIX+'/deployed/'+commit
    # A separate Git folder records a genuine remote pull when public Git access is supported.
    folders=cli(['repos','list','--path-prefix',workspace_path(PREFIX+'/source')],allow_fail=True)
    # Shared Git-CLI folders can be readable by ID yet absent from Repos list.
    known=cli(['repos','get',str(GIT_FOLDER_ID)],allow_fail=True)
    repo_id=GIT_FOLDER_ID if (workspace_path(known.get('path',''))==workspace_path(PREFIX+'/source') and known.get('url')==release['repository']) else None
    for r in folders if isinstance(folders,list) else folders.get('repos',[]):
        if workspace_path(r.get('path',''))==workspace_path(PREFIX+'/source'):repo_id=r['id']
    if repo_id is None:
        p=artifact/'git_folder.json';dump(p,{'url':release['repository'],'provider':'gitHub','path':workspace_path(PREFIX+'/source')})
        created=cli(['repos','create','--json','@'+str(p)],allow_fail=True);repo_id=created.get('id')
    if repo_id:
        pulled=cli(['repos','update',str(repo_id),'--branch',release['branch']],allow_fail=True)
        info=cli(['repos','get',str(repo_id)],allow_fail=True)
        if info.get('head_commit_id')!=commit:raise ValueError('Databricks Git folder did not pull the exact pushed commit')
        git_receipt={'mode':'Databricks Git folder pull','repo_id':repo_id,'head_commit_id':info['head_commit_id']}
    else:git_receipt={'mode':'Exact pushed Git commit fetched into isolated release archive; deployed through workspace sync','git_folder_unavailable':True}
    variables='source_commit='+commit+',code_path='+code
    validation=cli(['bundle','validate','--strict','--target','pilot','--var',variables],cwd=stage)
    cli(['bundle','deploy','--auto-approve','--target','pilot','--var',variables],cwd=stage,json_output=False)
    summary=cli(['bundle','summary','--target','pilot','--var',variables],cwd=stage)
    jobs=summary.get('resources',{}).get('jobs',{});job=jobs['semantic_gate_pilot'];job_id=str(job.get('id') or job.get('job_id'))
    deployed=cli(['jobs','get',job_id]);spec=deployed['settings']
    if spec.get('tags',{}).get('source_commit')!=commit:raise ValueError('Job version tag mismatch')
    for task in spec['tasks']:
        if any(k in task for k in ['new_cluster','existing_cluster_id','job_cluster_key']):raise ValueError('Unexpected classic compute')
        if task.get('compute',{}).get('hardware_accelerator'):raise ValueError('Unexpected GPU allocation')
        p=task['notebook_task']['notebook_path'];dest=artifact/(task['task_key']+'_deployed.py')
        cli(['workspace','export',p,'--file',str(dest),'--format','SOURCE'],json_output=False)
        original=stage/'notebooks'/('preflight.py' if task['task_key']=='fresh_cpu_admission' else 'pilot.py')
        normalize=lambda s:'\n'.join(line.rstrip() for line in s.replace('\r\n','\n').splitlines()).strip()
        if normalize(dest.read_text(encoding='utf-8'))!=normalize(original.read_text(encoding='utf-8')):raise ValueError('Deployed notebook source differs')
    verification_id=str(jobs['evidence_verify']['id']) if 'evidence_verify' in jobs else None
    if verification_id:
        verify_job=cli(['jobs','get',verification_id]);task=verify_job['settings']['tasks'][0]
        dest=artifact/'verification_deployed.py';cli(['workspace','export',task['notebook_task']['notebook_path'],'--file',str(dest),'--format','SOURCE'],json_output=False)
        if normalize(dest.read_text(encoding='utf-8'))!=normalize((stage/'notebooks/audit.py').read_text(encoding='utf-8')):raise ValueError('Verification notebook source differs')
    recovery_id=str(jobs['oracle_recovery_development']['id']) if 'oracle_recovery_development' in jobs else None
    if recovery_id:
        recovery_job=cli(['jobs','get',recovery_id]);task=recovery_job['settings']['tasks'][0]
        if any(k in task for k in ['new_cluster','existing_cluster_id','job_cluster_key']) or task.get('compute',{}).get('hardware_accelerator'):raise ValueError('Recovery job must use serverless CPU')
        dest=artifact/'recovery_deployed.py';cli(['workspace','export',task['notebook_task']['notebook_path'],'--file',str(dest),'--format','SOURCE'],json_output=False)
        if normalize(dest.read_text(encoding='utf-8'))!=normalize((stage/'notebooks/recovery.py').read_text(encoding='utf-8')):raise ValueError('Recovery notebook source differs')
    semantic_id=str(jobs['semantic_readout_refit']['id']) if 'semantic_readout_refit' in jobs else None
    if semantic_id:
        semantic_job=cli(['jobs','get',semantic_id]);task=semantic_job['settings']['tasks'][0]
        if any(k in task for k in ['new_cluster','existing_cluster_id','job_cluster_key']) or task.get('compute',{}).get('hardware_accelerator'):raise ValueError('Semantic fit must use serverless CPU')
        dest=artifact/'semantic_refit_deployed.py';cli(['workspace','export',task['notebook_task']['notebook_path'],'--file',str(dest),'--format','SOURCE'],json_output=False)
        if normalize(dest.read_text(encoding='utf-8'))!=normalize((stage/'notebooks/semantic_refit.py').read_text(encoding='utf-8')):raise ValueError('Semantic fitting notebook differs from pushed source')
    receipt={'source_commit':commit,'local_release':str(stage),'workspace_code_path':code,'job_id':job_id,'verification_job_id':verification_id,'recovery_job_id':recovery_id,'semantic_refit_job_id':semantic_id,'bundle_summary':summary,'git_pull':git_receipt,'notebook_sources_verified':True,'serverless_CPU_only':True,'workspace':HOST}
    dump(artifact/'DEPLOYMENT.json',receipt);return receipt
def submit(artifact):
    artifact=Path(artifact);deployment=read(artifact/'DEPLOYMENT.json');active=cli(['jobs','list-runs','--active-only'])
    active=active if isinstance(active,list) else active.get('runs',[])
    if len(active)>=4:raise RuntimeError('Shared concurrency reserve: leave slots available for Research 3')
    if any(str(r.get('job_id'))==deployment['job_id'] for r in active):raise RuntimeError('A Research 4 run is already active')
    p=artifact/'run_now.json';dump(p,{'job_id':int(deployment['job_id'])});run=cli(['jobs','run-now','--no-wait','--json','@'+str(p)])
    receipt={'run_id':run['run_id'],'job_id':deployment['job_id'],'source_commit':deployment['source_commit'],'active_runs_at_admission':[{'run_id':r.get('run_id'),'run_name':r.get('run_name')} for r in active],'GPU_allocations':0,'workspace':HOST}
    dump(artifact/'SUBMISSION.json',receipt);return receipt
def verify(artifact):
    artifact=Path(artifact);deployment=read(artifact/'DEPLOYMENT.json');pilot=read(artifact/'SUBMISSION.json')
    active=cli(['jobs','list-runs','--active-only']);active=active if isinstance(active,list) else active.get('runs',[])
    owned={deployment['job_id'],str(deployment['verification_job_id'])}
    if len(active)>=4 or any(str(r.get('job_id')) in owned for r in active):raise RuntimeError('Shared/Research4 concurrency admission failed')
    p=artifact/'verify_now.json';dump(p,{'job_id':int(deployment['verification_job_id']),'job_parameters':{'pilot_source_commit':pilot['source_commit'],'pilot_run_id':str(pilot['run_id'])}})
    r=cli(['jobs','run-now','--no-wait','--json','@'+str(p)]);result={'run_id':r['run_id'],'audit_source_commit':deployment['source_commit'],'pilot_source_commit':pilot['source_commit'],'pilot_run_id':pilot['run_id'],'GPU_allocations':0};dump(artifact/'VERIFICATION_SUBMISSION.json',result);return result

def recovery_submit(artifact,mode):
    artifact=Path(artifact);deployment=read(artifact/'DEPLOYMENT.json');target=artifact/('RECOVERY_'+mode.upper()+'_SUBMISSION.json')
    if target.exists():raise RuntimeError('This bounded stage was already submitted; inspect its retained result')
    active=cli(['jobs','list-runs','--active-only']);active=active if isinstance(active,list) else active.get('runs',[])
    owned={deployment['job_id'],str(deployment['verification_job_id']),str(deployment['recovery_job_id'])}
    if len(active)>=4 or any(str(r.get('job_id')) in owned for r in active):raise RuntimeError('Shared/Research4 concurrency admission failed')
    probe_run=''
    if mode=='qualification':
        previous=read(artifact/'RECOVERY_PROBE_SUBMISSION.json');summary=read(artifact/'PROBE_SUMMARY.json')
        if not summary['interface_qualified'] or previous['source_commit']!=deployment['source_commit']:raise RuntimeError('No same-source qualifying training probe')
        terminal=cli(['jobs','get-run',str(previous['run_id'])])
        if terminal['state'].get('result_state')!='SUCCESS':raise RuntimeError('Training interface probe has not completed successfully')
        probe_run=str(previous['run_id'])
    p=artifact/('recovery_'+mode+'_now.json');dump(p,{'job_id':int(deployment['recovery_job_id']),'job_parameters':{'mode':mode,'probe_run_id':probe_run}})
    r=cli(['jobs','run-now','--no-wait','--json','@'+str(p)])
    receipt={'run_id':r['run_id'],'job_id':deployment['recovery_job_id'],'source_commit':deployment['source_commit'],'mode':mode,'probe_run_id':probe_run,'GPU_allocations':0,'active_runs_at_admission':[{'run_id':r.get('run_id'),'run_name':r.get('run_name')} for r in active]}
    dump(target,receipt);return receipt
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['deploy','submit','verify','status','recovery-probe','recovery-qualification','recovery-status']);p.add_argument('--repo',default=str(Path(__file__).resolve().parents[2]));p.add_argument('--artifact',required=True);a=p.parse_args()
    if a.command=='deploy':r=deploy(a.repo,a.artifact)
    elif a.command=='submit':r=submit(a.artifact)
    elif a.command=='verify':r=verify(a.artifact)
    elif a.command in ['recovery-probe','recovery-qualification']:r=recovery_submit(a.artifact,a.command.split('-',1)[1])
    elif a.command=='recovery-status':
        folder=Path(a.artifact);p=folder/'RECOVERY_QUALIFICATION_SUBMISSION.json'
        if not p.exists():p=folder/'RECOVERY_PROBE_SUBMISSION.json'
        r=cli(['jobs','get-run',str(read(p)['run_id'])])
    else:r=cli(['jobs','get-run',str(read(Path(a.artifact)/'SUBMISSION.json')['run_id'])])
    print(json.dumps(r,indent=2))
if __name__=='__main__':main()
