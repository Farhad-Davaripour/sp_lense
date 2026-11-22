"""Read the pinned local completion receipt; no model, network or weight reads."""
import hashlib
import json
import math
from pathlib import Path


AUDIT_PATH = '.local/experiments/service_continuity_compact_r16_last8_v2_20261114/reviews/INDEPENDENT_COMPLETION_AUDIT_v1.json'
AUDIT_SHA256 = 'c1be59be1791a6ec249f329a9b825259a6fd30d8d65bb7132268b4538b5cc5bb'
ADAPTER_SHA256 = '9a4538c3bc9bea9b9a2db206b625bbbad63f1b41ef6a48d44fef1812bf51c695'
CHECKPOINT_STEPS = [1, 2, 3, *range(180, 2161, 180)]
MAC_PROJECT = '/Users/fdavaripour/repos/sp_lense4/research4'
RUN = MAC_PROJECT + '/.local/experiments/service_continuity_compact_r16_last8_v2_20261114/runs/fit_027f6b8124bf4e68bba83d7a31b6869d'
ARCHIVE = MAC_PROJECT + '/.local/archive/compact_v2_fit_027f6b8124bf4e68bba83d7a31b6869d/FROZEN_RUN_v1.tar.gz'
CATALOG_PATH = '.local/monitor/SP_LENS_MODEL_VERSIONS_ABT_20261114.json'
CATALOG_SHA256 = 'c655fa5d3d51a8606881d6011d1b489eed36c7d3eb3d527bd1eb7c2c2a216734'


def version_catalog_snapshot(root):
    """The two hash-bound version names, times and browsing aliases only."""
    result = {'schema': 'research4.sp-lens-model-version-catalog.v1', 'status': 'unavailable'}
    try:
        path = Path(root) / CATALOG_PATH
        if path.is_symlink() or path.stat().st_size > 2_000_000:
            raise ValueError('Version catalog unavailable')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != CATALOG_SHA256:
            raise ValueError('Version catalog hash differs')
        value = json.loads(raw)
        if (value.get('schema') != result['schema'] or value.get('timezone') != 'America/Edmonton'
                or value.get('display_zone_abbreviation') != 'ABT' or value.get('utc_offset_seconds') != -21600):
            raise ValueError('Version catalog identity differs')
        result.update(value, status='verified', catalog_sha256=CATALOG_SHA256)
    except (OSError, ValueError, TypeError) as error:
        result['note'] = str(error)
    return result


def snapshot(root):
    """Fail closed if the copied audit is missing, changed or inconsistent."""
    result = {'schema': 'research4.compact-validation-dashboard.v1', 'status': 'unavailable',
              'source': 'Copied local audit receipt; saved evidence only.'}
    try:
        root = Path(root).resolve()
        path = root / AUDIT_PATH
        if (path.is_symlink() or not path.is_file() or path.stat().st_size > 2_000_000
                or not path.resolve().is_relative_to(root)):
            raise ValueError('Saved validation audit unavailable')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != AUDIT_SHA256:
            raise ValueError('Saved validation audit hash differs')
        audit = json.loads(raw)
        if (audit.get('schema') != 'research4.independent-compact-v2-saved-completion-audit.v1'
                or audit.get('status') != 'PASS_SAVED_COMPACT_V2_COMPLETION_EVIDENCE'
                or audit.get('errors') != []):
            raise ValueError('Saved completion audit did not pass')
        facts = audit['facts']
        loss = facts['DEV_pooled_target_NLL']
        checkpoints = facts['checkpoints']
        optimizer = facts['saved_final_Adam']
        if (facts.get('completed_verified_updates') != 2160 or facts.get('DEV_rows') != 500
                or type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0
                or facts.get('DEV_no_update_bytes_match') is not True
                or facts.get('checkpoint_steps') != CHECKPOINT_STEPS
                or [item['step'] for item in checkpoints] != CHECKPOINT_STEPS
                or any(item.get('all_finite') is not True or item.get('tensor_count') != 124
                       or item.get('dtype') != 'F32' for item in checkpoints)
                or facts.get('final_adapter_sha256') != ADAPTER_SHA256
                or checkpoints[-1].get('adapter_sha256') != ADAPTER_SHA256
                or optimizer.get('moment_count') != 248 or optimizer.get('optimizer_step') != 2160
                or optimizer.get('all248_F32_moments_finite') is not True
                or facts.get('behavioral_improvement_established') is not False):
            raise ValueError('Saved validation facts are inconsistent')
        evidence = audit['evidence']
        if any(evidence.get(name) != ADAPTER_SHA256 for name in
               ('final_adapter_sha256', 'before_DEV_sha256', 'after_DEV_sha256')):
            raise ValueError('DEV no-update hashes differ')
        result.update(status='verified', audit_status='passed', validation_examples=facts['DEV_rows'],
                      pooled_target_nll=loss, no_update_verified=True, verified_updates=2160,
                      checkpoint_count=len(checkpoints), checkpoint_steps=CHECKPOINT_STEPS,
                      finite_adapter_tensors_per_checkpoint=124, finite_Adam_moments=248,
                      final_adapter_sha256=ADAPTER_SHA256, audit_sha256=AUDIT_SHA256,
                      audit_relative_path=AUDIT_PATH,
                      scope='Validation token loss; does not establish action accuracy or game improvement.',
                      storage={'host': 'Mac Studio', 'checkpoint_directory': RUN + '/adapters',
                               'final_adapter': RUN + '/adapters/step_002160/adapters.safetensors',
                               'validation_files': [RUN + '/' + name for name in
                                                    ('DEV.json', 'before_DEV.safetensors', 'after_DEV.safetensors')],
                               'frozen_run_archive': ARCHIVE,
                               'pc_contents': 'Copied audit and proof metadata only; full weights and checkpoints remain on the Mac.'})
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        result['note'] = str(error)
    return result
