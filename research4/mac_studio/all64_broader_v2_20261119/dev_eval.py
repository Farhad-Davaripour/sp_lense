"""Frozen DEV likelihood scoring and nine-checkpoint selection.

This module does not load a model, create checkpoints or admit a job. The
caller owns the model and supplies the already authenticated frozen inputs.
"""
import importlib.util
import math
from pathlib import Path

import numpy as np

from frozen_data import MANIFEST, authenticate, sha
from loss import HEAD_WINDOW, make_batch, selected_suffix_vector


def score_rows(runtime, neural, model, prepared_dev, masks, parameter_digest,
               optimizer_step=None, *, checkpoint_step=None, restore_training=True,
               row_boundary=None):
    """Score fixed reference suffixes; verify weights and optimizer do not move."""
    if len(prepared_dev) != 36 or len(masks) != 36:
        raise ValueError('Exactly 36 frozen likelihood rows are required')
    if type(restore_training) is not bool:
        raise ValueError('Explicit training-mode restoration required')
    before = parameter_digest(model)
    if optimizer_step is None:
        if type(checkpoint_step) is not int or checkpoint_step < 1:
            raise ValueError('A separately evaluated checkpoint must have its saved step')
        step_before = None
    else:
        if not callable(optimizer_step) or checkpoint_step is not None:
            raise ValueError('Use either a live optimizer or a separate checkpoint evaluation')
        step_before = optimizer_step()
    if (not isinstance(before, str) or len(before) != 64
            or (optimizer_step is not None and type(step_before) is not int)):
        raise ValueError('Invalid pre-evaluation parameter or optimizer evidence')
    output = []
    model.eval()
    try:
        for prepared, reference in zip(prepared_dev, masks):
            row_id, family, ids, prefix, action_start, action_end, _ = prepared
            if row_boundary is not None:
                row_boundary(row_id)
            if (row_id != reference.get('id') or family != reference.get('family')
                    or reference.get('schema') != 'sp-lense.action-token-span.v1'
                    or reference.get('reference_reasoning_conditioned') is not True
                    or reference.get('final_eos_included') is not True
                    or reference.get('prompt_tokens') != prefix
                    or reference.get('full_sequence_tokens') != len(ids)
                    or reference.get('full_suffix_start_token') != prefix
                    or reference.get('full_suffix_end_token_exclusive') != len(ids)
                    or reference.get('action_start_token') != action_start
                    or reference.get('action_end_token_exclusive') != action_end
                    or not prefix <= action_start < action_end == len(ids)):
                raise ValueError('DEV row or fixed action mask differs')
            batch, lengths, geometry = make_batch(runtime, ids, prefix)
            nll, mask = selected_suffix_vector(runtime, neural, model, batch, lengths)
            runtime.eval(nll, mask)
            values = np.asarray(nll.tolist(), dtype=np.float64)
            valid = np.asarray(mask.tolist(), dtype=bool)
            suffix_count = geometry['target_tokens']
            if (values.shape != (1, HEAD_WINDOW) or valid.shape != values.shape
                    or not np.array_equal(valid[0], np.arange(HEAD_WINDOW) < suffix_count)
                    or not np.isfinite(values[0, :suffix_count]).all()
                    or np.any(values[0, :suffix_count] < 0)):
                raise ValueError('Invalid per-token DEV likelihood or suffix mask')
            start, end = action_start - prefix, action_end - prefix
            if (end != suffix_count or end - start != reference.get('action_tokens_including_eos')
                    or not 0 <= start < end <= HEAD_WINDOW):
                raise ValueError('Action span falls outside the scored suffix')
            action_sum = float(values[0, start:end].sum(dtype=np.float64))
            suffix_sum = float(values[0, :suffix_count].sum(dtype=np.float64))
            if not math.isfinite(action_sum) or not math.isfinite(suffix_sum):
                raise ValueError('DEV score is not finite')
            output.append({'id': row_id, 'family': family,
                           'action_nll_sum': action_sum, 'action_tokens': end - start,
                           'full_suffix_nll_sum': suffix_sum, 'full_suffix_tokens': suffix_count})
        after = parameter_digest(model)
        step_after = optimizer_step() if optimizer_step is not None else None
        if after != before or step_after != step_before:
            raise RuntimeError('DEV scoring changed trainable parameters or optimizer step')
        return {'development_rows': output, 'development_no_optimizer_updates': True,
                'trainable_digest_before': before, 'trainable_digest_after': after,
                'optimizer_absent': optimizer_step is None, 'checkpoint_step': checkpoint_step,
                'optimizer_step_before': step_before, 'optimizer_step_after': step_after}
    finally:
        if restore_training:
            model.train()


def choose_verified(candidates, masks, checkpoint_verifier):
    """Verify all candidate checkpoint bytes, then call the frozen selector.

    Candidates are ordered as the frozen selector's GRID. Each must contain
    rank, epoch, status, development_no_optimizer_updates, development_rows,
    checkpoint_path, checkpoint_sha256, scoring_checkpoint_sha256, the
    before/after trainable digests and optimizer steps. ``checkpoint_verifier``
    must raise on incorrect rank, layout, F32/nonzero-layer coverage or
    provenance, then return the parameter-state digest reconstructed from the
    checkpoint bytes. It must not update model weights or optimizer state.
    """
    manifest, _, _, frozen_masks = authenticate()
    if masks != frozen_masks:
        raise ValueError('Selection masks do not match the authenticated frozen masks')
    if not callable(checkpoint_verifier) or len(candidates) != 9:
        raise ValueError('All nine verified epoch checkpoints are required')
    source = MANIFEST.parent / 'hpo_selection.py'
    if sha(source) != manifest['source_files']['hpo_selection.py']:
        raise ValueError('Frozen selector changed')
    spec = importlib.util.spec_from_file_location('research4_frozen_hpo_selector', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selector_candidates = []
    for candidate, (rank, epoch) in zip(candidates, module.GRID):
        if (candidate.get('rank'), candidate.get('epoch'), candidate.get('status'),
                candidate.get('development_no_optimizer_updates')) != (rank, epoch, 'complete', True):
            raise ValueError('Candidate identity or development evidence differs')
        path = Path(candidate['checkpoint_path'])
        digest = candidate.get('checkpoint_sha256')
        if not isinstance(digest, str) or len(digest) != 64 or sha(path) != digest:
            raise ValueError('Epoch checkpoint bytes differ')
        live_optimizer = (candidate.get('optimizer_absent') is False
                          and candidate.get('checkpoint_step') is None
                          and candidate.get('optimizer_step_before') == 136 * epoch
                          and candidate.get('optimizer_step_after') == 136 * epoch)
        separate_eval = (candidate.get('optimizer_absent') is True
                         and candidate.get('checkpoint_step') == 136 * epoch
                         and candidate.get('optimizer_step_before') is None
                         and candidate.get('optimizer_step_after') is None)
        if candidate.get('scoring_checkpoint_sha256') != digest or not (live_optimizer or separate_eval):
            raise ValueError('DEV score is not bound to the expected checkpoint and optimizer step')
        state_digest = checkpoint_verifier(candidate, path)
        if (not isinstance(state_digest, str) or len(state_digest) != 64
                or candidate.get('trainable_digest_before') != state_digest
                or candidate.get('trainable_digest_after') != state_digest):
            raise ValueError('DEV parameters do not match the saved epoch checkpoint')
        selector_candidates.append({key: candidate[key] for key in (
            'rank', 'epoch', 'status', 'development_no_optimizer_updates', 'development_rows')})
    return module.choose(selector_candidates, frozen_masks)
