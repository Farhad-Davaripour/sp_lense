"""Exact submitted full-fit lineages; registration does not certify a result.

The first fit failed after 12 updates. The second is a fresh, separately
registered attempt under the stateful memory profile. Both still require the
complete independent raw reader, actual 64-update exit and provider release.
"""
from copy import deepcopy

_FITS={
 ('6c37f59b60ceff32c74b1c39f0a6b031d207e443',361922334):{
  'source_commit':'6c37f59b60ceff32c74b1c39f0a6b031d207e443','run_id':361922334,
  'package_sha256':'b18e06971434cda2b5312e670f5942c6b44688826103cb421f0f8e607fe83e5b',
  'manifest_sha256':'3069515817b1ac3ec599de2becb1e3d0cdf0758b957327d5e1c789f5c91d8ecc',
  'qualification_archive_sha256':'b9c1349cbc43747d3d5b715238a0d46ce67920aab52289c6f478a6d23bd2ee44',
  'prepared_notebook':'research4_task_skill_fit.ipynb',
  'memory_profile_config':None},
 ('82cf3ecf41e799facd93b8c11e05c6d1f569599c',361964150):{
  'source_commit':'82cf3ecf41e799facd93b8c11e05c6d1f569599c','run_id':361964150,
  'package_sha256':'6ca70b6abc023432dacdd6fa1fb0e1d3e4a6ca3a840733f3f02495494683fc07',
  'manifest_sha256':'9bc895fe7a6dbd31225efeb2f4cc4b67ffa688be2438aad0cee179731604483c',
  'qualification_archive_sha256':'16d4721eba31d8696cc097574f94e8a795d7e08b92afd998b98b24d2313c02a4',
  'prepared_notebook':'research4_task_skill_fit_v3.ipynb',
  'memory_profile_config':'composite_task_skill_memory_v3.json'}
}


def registered_fit(source_commit,run_id):
    if (not isinstance(source_commit,str) or type(run_id) is not int
            or (source_commit,run_id) not in _FITS):
        raise PermissionError('Only an exact separately registered full-fit source/run pair is eligible')
    return deepcopy(_FITS[source_commit,run_id])
