# Trusted no-mount native export. Root downloads/verifies and privately persists before release.
import tarfile
from datetime import datetime,timezone
assert TR_PROCESS.poll() is not None
TR_WORKER_WATCH.cancel()
assert _sha_file(TR_ADAPTER/'adapter_model.safetensors')=='8ae28fb51107a31419bcb744d9f47d80b681781a114f8bc9491751410b4b87aa'
assert _sha_file(TR_ADAPTER/'adapter_config.json')=='06a6ad4588f4f0d7b46cf9f7f67f9b7b5028d153134cdd0829a929e1caf9fcda'
_bundle=Path('/content/sp_lense_transport')/TR_ROOT.name;_bundle.mkdir(parents=True,exist_ok=True)
_archive=_bundle/'SENTENCE_NATIVE_UPLOAD_v1.tar.gz'
_files={};_excluded={'model','worker_home','__pycache__'}
for _path in sorted(TR_ROOT.rglob('*')):
 if not _path.is_file():continue
 _rel=_path.relative_to(TR_ROOT)
 if any(_part in _excluded for _part in _rel.parts):continue
 if _rel.parts[:2]==('inputs','A'):continue  # exact checkpoint already verified locally/private; referenced below
 assert not _path.is_symlink()
 _files[_rel.as_posix()]={'bytes':_path.stat().st_size,'sha256':_sha_file(_path)}
with tarfile.open(_archive,'w:gz',compresslevel=1) as _tar:
 for _name in _files:_tar.add(TR_ROOT/_name,arcname=_name,recursive=False)
_manifest={'format':'sentence_native_upload_v1','run':TR_ROOT.name,'native_files':_files,
           'archive_bytes':_archive.stat().st_size,'archive_sha256':_sha_file(_archive),'exclusions':sorted(_excluded)+['inputs/A'],
           'A_checkpoint_preserved_elsewhere_before_execution':True,
           'A_weights_sha256':'8ae28fb51107a31419bcb744d9f47d80b681781a114f8bc9491751410b4b87aa',
           'A_config_sha256':'06a6ad4588f4f0d7b46cf9f7f67f9b7b5028d153134cdd0829a929e1caf9fcda',
           'all_generated_native_output_files_preserved':True,'optimizer_steps':0,'main_scores_unchanged':True,
           'private_and_local_copy_not_yet_confirmed':True,'worker_exit':TR_PROCESS.poll()}
_manifest_path=_bundle/'SENTENCE_EXPORT_MANIFEST_UPLOAD_v1.json';_manifest_path.write_text(json.dumps(_manifest,indent=2)+'\n')
print('UPLOAD_EXPORT_READY',json.dumps({'run':TR_ROOT.name,'native_files':len(_files),'archive_path':str(_archive),
     'manifest_path':str(_manifest_path),'archive_bytes':_manifest['archive_bytes'],'archive_sha256':_manifest['archive_sha256'],
     'manifest_sha256':_sha_file(_manifest_path),'hard_phase_guard_still_armed':True},separators=(',',':')),flush=True)
