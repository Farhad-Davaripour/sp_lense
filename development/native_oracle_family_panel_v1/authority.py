"""Explicit root oracle authority; implementation alone never admits a model."""
import json,os,time
from pathlib import Path
from support import HERE,ATTEMPT,output,require,sha,check_freeze,write_new
from input_reader import require_oracle,validate
LIMITS={'loads':1,'forwards':72,'derivatives':16,'encoding':0,'worker_seconds':900,'audit_seconds':90,'shared_cleanup_seconds':15,'total_bytes':96*1024**2,'file_bytes':5*1024**2}
RELEASE=HERE/'root_release'/'RELEASE.json'
def read_release(approved):
    require(type(approved) is str and len(approved)==64,'EXPLICIT_ROOT_RELEASE_HASH_REQUIRED')
    raw=RELEASE.read_bytes();require(sha(raw)==approved,'ROOT_RELEASE_BYTES');release=json.loads(raw)
    require(release['schema']=='native_oracle_family_panel_release.v1' and release['approved'] is True
        and release['attempt']==ATTEMPT and release['limits']==LIMITS
        and Path(release['output']).resolve()==output().resolve(),'EXACT_NATIVE_RELEASE')
    require_oracle(release.get('oracle_authority'))
    for name,key in (('SOURCE_FREEZE.json','source_freeze_sha256'),('inputs.json','inputs_sha256'),
                     ('CHECKPOINT.json','checkpoint_lock_sha256'),('OWNED_IDENTITY.json','owned_identity_sha256')):
        require(sha((HERE/name).read_bytes())==release[key],'RELEASE_BINDING_'+key)
    frozen=check_freeze()
    require(release['trace_sources']=={n:h for n,h in frozen['source_sha256'].items() if n.endswith('.py')},'EXACT_TRACE_SOURCE_BINDING')
    validate(json.loads((HERE/'inputs.json').read_bytes()))
    return release
def execution(release,approved):
    return {'scope':'ROOT_APPROVED_ORACLE_DEVELOPMENT_ONLY','attempt':ATTEMPT,'release_sha256':approved,
        'source_freeze_sha256':release['source_freeze_sha256'],'inputs_sha256':release['inputs_sha256'],
        'checkpoint_lock_sha256':release['checkpoint_lock_sha256'],'historical_parity_claimed':False,
        'oracle_authority':release['oracle_authority'],'learned_gate_success_claimed':False}
def authenticate():
    approved=os.environ.get('SP_NATIVE_RELEASE_SHA','');release=read_release(approved)
    raw=(output()/'ADMISSION.json').read_bytes();require(sha(raw)==os.environ.get('SP_NATIVE_ADMISSION_SHA',''),'ADMISSION_BYTES')
    admission=json.loads(raw);expected=execution(release,approved)
    require(admission['execution']==expected and admission['source_freeze_sha256']==release['source_freeze_sha256'],'ADMISSION_EXECUTION')
    require(time.monotonic()<=admission['absolute_end'],'ADMISSION_ABSOLUTE_END')
    return {'execution':expected,'admission':admission,'release':release}
def admit_once(approved):
    release=read_release(approved);output().mkdir(parents=True,exist_ok=False);started=time.monotonic()
    admission={'execution':execution(release,approved),'source_freeze_sha256':release['source_freeze_sha256'],
        'started_monotonic':started,'absolute_end':started+1005,'controller_pid':os.getpid()}
    receipt=write_new('ADMISSION.json',admission)
    os.environ['SP_NATIVE_RELEASE_SHA']=approved;os.environ['SP_NATIVE_ADMISSION_SHA']=receipt['sha256']
    return admission
