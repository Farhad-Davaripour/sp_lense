"""Exact one-shot TRAIN-only authority; no provider work without retained admission."""
import json,os,time
from pathlib import Path
from support import HERE,ATTEMPT,output,require,sha,check_freeze,write_new,GROUP_CAPS
LIMITS={'loads':1,'forwards':16,'derivatives':0,'worker_seconds':300,'audit_seconds':120,'shared_cleanup_seconds':15,'total_bytes':64*1024**2,'file_bytes':5*1024**2}
RELEASE=HERE/'root_release/RELEASE.json'
def read_release(approved):
    require(type(approved) is str and len(approved)==64,'EXPLICIT_ROOT_RELEASE_HASH_REQUIRED')
    raw=RELEASE.read_bytes();require(sha(raw)==approved,'ROOT_RELEASE_BYTES');release=json.loads(raw)
    require(release['schema']=='prechoice_diagnostic_capture_release.v1' and release['approved'] is True
        and release['attempt']==ATTEMPT and release['limits']==LIMITS and release['role']=='DIAGNOSTIC_PRECHOICE'
        and Path(release['output']).resolve()==output().resolve(),'EXACT_NATIVE_RELEASE')
    require(release['reserved_bytes']==sum(GROUP_CAPS.values())+65536==19628032
        and release['reserved_bytes']<=LIMITS['total_bytes'],'FULL_WORST_CASE_RESERVATION')
    for name,key in (('SOURCE_FREEZE.json','source_freeze_sha256'),('CHECKPOINT.json','checkpoint_lock_sha256'),
        ('OWNED_IDENTITY.json','owned_identity_sha256'),('DATA_LOCK.json','input_data_lock_sha256')):
        require(sha((HERE/name).read_bytes())==release[key],'RELEASE_BINDING_'+key)
    frozen=check_freeze()
    require(release['trace_sources']=={n:h for n,h in frozen['source_sha256'].items() if n.endswith('.py')},'EXACT_TRACE_SOURCE_BINDING')
    from input_reader import accepted_reader, read_bundle
    accepted=accepted_reader().verify_accepted_checkpoint()
    require(accepted['freeze_sha256']==release['fit_freeze_sha256'],'ACCEPTED_CHECKPOINT_RELEASE_JOIN')
    read_bundle(RELEASE.parent,release)
    return release
def execution(release,approved):
    return {'scope':'ROOT_APPROVED_NATIVE_CONSTRUCTION_CAPTURE_ONLY','attempt':ATTEMPT,'release_sha256':approved,
        'source_freeze_sha256':release['source_freeze_sha256'],'inputs_sha256':release['inputs_sha256'],
        'checkpoint_lock_sha256':release['checkpoint_lock_sha256'],'historical_parity_claimed':False,
        'input_data_lock_sha256':release['input_data_lock_sha256'],'certificate_sha256':release['certificate_sha256'],'role':'DIAGNOSTIC_PRECHOICE','fit_freeze_sha256':release['fit_freeze_sha256']}
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
        'started_monotonic':started,'absolute_end':started+435,'controller_pid':os.getpid()}
    receipt=write_new('ADMISSION.json',admission)
    os.environ['SP_NATIVE_RELEASE_SHA']=approved;os.environ['SP_NATIVE_ADMISSION_SHA']=receipt['sha256']
    return admission
