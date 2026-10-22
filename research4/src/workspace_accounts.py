"""One operator implementation, two explicitly authorized workspace accounts.

Secrets live only in the ignored repository-root .env or process environment.
Remote scientific workers receive a public, manifest-bound workspace descriptor.
"""
import argparse,json,os,re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ENV_FILE=ROOT.parent/'.env'


def profiles():return json.loads((ROOT/'config/workspace_accounts.json').read_text())


def env_values(path):
    values={}
    if not Path(path).exists():return values
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        line=line.strip()
        if not line or line.startswith('#'):continue
        key,separator,value=line.partition('=');key=key.strip();value=value.strip()
        if not separator or not re.fullmatch('[A-Z_][A-Z0-9_]*',key) or key in values:raise ValueError('Invalid or duplicate .env key; values withheld')
        if value.startswith('"'):
            value=json.loads(value)
            if not isinstance(value,str):raise ValueError('Expected a string .env value')
        elif len(value)>=2 and value[0]==value[-1]=="'":value=value[1:-1]
        values[key]=value
    return values


def account_name(value):
    name=value.strip().lower().replace(' ','_')
    if name=='free':name='free_tier'
    if name not in profiles()['accounts']:raise ValueError('Select work or free tier')
    return name


def selected(account=None,environ=None,env_file=ENV_FILE):
    env=os.environ if environ is None else environ;values=env_values(env_file);config=profiles()
    name=account_name(account or env.get('RESEARCH4_DATABRICKS_ACCOUNT') or values.get('RESEARCH4_DATABRICKS_ACCOUNT') or config['default_account'])
    return {'account':name,**config['accounts'][name]}


def command_environment(account=None,environ=None,env_file=ENV_FILE):
    env=dict(os.environ if environ is None else environ);values=env_values(env_file);profile=selected(account,env,env_file);prefix=profile['credential_prefix']
    declared=env.get(prefix+'_HOST',values.get(prefix+'_HOST',profile['host']))
    if declared.rstrip('/')!=profile['host']:raise ValueError('Account host does not match its authorized workspace')
    token=env.get(prefix+'_TOKEN') or values.get(prefix+'_TOKEN')
    # Preserve old explicitly addressed scripts without allowing their generic
    # token to be silently paired with the other account's host.
    if not token and env.get('DATABRICKS_HOST','').rstrip('/')==profile['host']:token=env.get('DATABRICKS_TOKEN')
    if not token:raise ValueError('Selected account credential is missing from .env/environment')
    env.pop('DATABRICKS_CONFIG_PROFILE',None)
    env.update(DATABRICKS_HOST=profile['host'],DATABRICKS_TOKEN=token,DATABRICKS_AUTH_TYPE='pat')
    return env


def redact(text,environ):
    value=re.sub(r'dapi[0-9a-fA-F]+','[REDACTED]',text)
    for key,secret in environ.items():
        if (key.endswith('_TOKEN') or key=='DATABRICKS_TOKEN') and secret:value=value.replace(secret,'[REDACTED]')
    return value


def switch(account,env_file=ENV_FILE):
    name=account_name(account);path=Path(env_file);values=env_values(path)
    # Validate both identity and credential before changing the active selector.
    command_environment(name,environ={},env_file=path)
    lines=path.read_text(encoding='utf-8-sig').splitlines() if path.exists() else []
    replacement='RESEARCH4_DATABRICKS_ACCOUNT='+name
    lines=[replacement if line.partition('=')[0].strip()=='RESEARCH4_DATABRICKS_ACCOUNT' else line for line in lines]
    if 'RESEARCH4_DATABRICKS_ACCOUNT' not in values:lines.append(replacement)
    # Updating the existing inode retains its Windows ACL. No credentials are
    # printed, copied to profile files, or changed during account switching.
    with path.open('w',encoding='utf-8',newline='\n') as stream:
        stream.write('\n'.join(lines)+'\n');stream.flush();os.fsync(stream.fileno())
    return selected(name,environ={},env_file=path)


def verify_runtime_workspace(root,host):
    path=Path(root)/'inputs/WORKSPACE.json';actual=host.rstrip('/')
    if path.exists():
        descriptor=json.loads(path.read_text());profile=selected(descriptor['account'],environ={},env_file=Path(root)/'nonexistent.env')
        if descriptor['host']!=profile['host'] or descriptor['organization']!=profile['organization'] or actual!=profile['host']:raise ValueError('Runtime workspace differs from its frozen descriptor')
    else:
        if actual!=profiles()['accounts']['free_tier']['host']:raise ValueError('Work-account execution requires a frozen workspace descriptor')
        profile=profiles()['accounts']['free_tier'];profile={'account':'free_tier',**profile}
    return {k:profile[k] for k in ['account','label','host','organization']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['status','switch']);parser.add_argument('account',nargs='?');args=parser.parse_args()
    if args.action=='switch' and not args.account:parser.error('switch requires work or "free tier"')
    profile=switch(args.account) if args.action=='switch' else selected()
    print(json.dumps({k:profile[k] for k in ['account','label','host','organization']}))
