"""Freeze the specifically approved sampler source/failed references; never draw or submit."""
import argparse,json,shutil,socket,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.io_utils import read,dump,sha
from src.composite_source_archive import archive_source
from src.composite_bounded_sampling_profile import (MODE,PHASE,NOTEBOOK,OWNER_THREAD,PROFILE,MANIFEST,
 APPROVAL,PROPOSAL,APPROVAL_SHA,PROPOSAL_SHA,expected_profile,notebook_code)
from src.composite_bounded_sampling_parent import assemble_failed_parent,read_failed_parent
from src.composite_bounded_sampling_source import approval,validate_source
from src.composite_bounded_sampling_admission import PACKAGE,NOTEBOOK_FILE,validate_package
from src.composite_kaggle_v2_profile import HUMAN,HUMAN_SHA,identity
from src.composite_bounded_behavior_profile import validate_preparation


def prepare(repo,commit,failed_directory,alias_fit_archive,alias_qual_archive,human_approval,approved_proposal,kaggle_scope,output):
    if sha(human_approval)!=APPROVAL_SHA or sha(approved_proposal)!=PROPOSAL_SHA or sha(kaggle_scope)!=HUMAN_SHA:
        raise PermissionError('Exact verified specific sampler approval/proposal and existing Kaggle scope required')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    bundle=out/'bundle';source=bundle/'source';archive_source(repo,commit,source)
    cfg=validate_preparation(read(source/'config/composite_bounded_behavior_v4.json'))
    inputs=source/'inputs';inputs.mkdir(exist_ok=False)
    assemble_failed_parent(failed_directory,inputs)
    for path,name in [(human_approval,APPROVAL),(approved_proposal,PROPOSAL),(kaggle_scope,HUMAN)]:
        shutil.copyfile(path,inputs/name)
    approved=approval(source,cfg)
    # This is original-only raw failure/provenance replay, never an alternative
    # draw or a new scientific world. New sampling is confined to the CPU stage.
    parent=read_failed_parent(source,None,alias_fit_archive,alias_qual_archive)
    profile=expected_profile(commit,cfg,parent,approved)
    dump(inputs/PROFILE,profile);dump(inputs/'WORKSPACE.json',identity(NOTEBOOK))
    dump(inputs/'MANIFEST.json',{p.name:sha(p) for p in sorted(inputs.iterdir()) if p.name!='MANIFEST.json'})
    dump(bundle/MANIFEST,{'version':MODE,'source_commit':commit,'notebook':NOTEBOOK,
      'files':{p.relative_to(bundle).as_posix():sha(p) for p in sorted(source.rglob('*')) if p.is_file()}})
    validate_source(source,commit,parent)
    expanded=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file())
    if expanded>128*1024**2:raise PermissionError('Complete approved sampler source/input transport exceeds unchanged128MiB')
    archive=out/PACKAGE
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as stream:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():stream.write(p,p.relative_to(bundle).as_posix())
    validate_package(archive,bundle)
    code=notebook_code(source,commit,sha(bundle/MANIFEST))
    dump(out/NOTEBOOK_FILE,{'cells':[{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
      'source':code.splitlines(keepends=True)}],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
      'nbformat':4,'nbformat_minor':5})
    receipt={'status':'prepared_approved_sampling_amendment_not_submitted','mode':MODE,'phase':PHASE,
      'source_commit':commit,'notebook':NOTEBOOK,'owner_thread_id':OWNER_THREAD,'platform':'kaggle','account':'farhad89',
      'destination':profile['destination'],'bundle_sha256':sha(archive),'bundle_manifest_sha256':sha(bundle/MANIFEST),
      'profile_sha256':sha(inputs/PROFILE),'input_manifest_sha256':sha(inputs/'MANIFEST.json'),
      'notebook_file_sha256':sha(out/NOTEBOOK_FILE),'source_transport_expanded_bytes':expanded,
      'source_transport_maximum_bytes':128*1024**2,'approval_sha256':APPROVAL_SHA,'proposal_sha256':PROPOSAL_SHA,
      'alias_fit_archive_local_path':str(Path(alias_fit_archive).resolve()),
      'alias_qual_archive_local_path':str(Path(alias_qual_archive).resolve()),
      'sampling_plan_materialized':False,'training_launch_admitted':False,
      'scientific_allocation_admitted_by_preparation':False,'original_failure_preserved':True,
      'model_forwards':0,'optimizer_updates':0,'scientific_GPU_count':0,'heldout_content_read':False}
    dump(out/'PREPARATION.json',receipt);return receipt


def deny_network(*args,**kwargs):raise PermissionError('Local sampler source preparation is offline and original-reference-only')


if __name__=='__main__':
    socket.create_connection=deny_network;socket.socket.connect=deny_network;socket.socket.connect_ex=deny_network
    parser=argparse.ArgumentParser()
    for name in ['repo','commit','failed-directory','alias-fit-archive','alias-qual-archive','human-approval','approved-proposal','kaggle-scope','output']:
        parser.add_argument('--'+name,required=True)
    print(json.dumps(prepare(**vars(parser.parse_args()))))
