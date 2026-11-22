from pathlib import Path

import pytest

import selection_rank16 as selection


def test_only_rank16_epochs_compete_under_the_frozen_metric():
    _, _, _, masks = selection.evidence.frozen_data.authenticate()
    scorer = selection.load_source('frozen_rank16_test_scorer', Path(selection.PROJECT) / selection.SCORER)

    def candidate(rank, epoch, loss):
        rows = []
        for mask in masks:
            action_tokens = mask['action_tokens_including_eos']
            suffix_tokens = mask['full_suffix_end_token_exclusive'] - mask['full_suffix_start_token']
            rows.append({'id': mask['id'], 'family': mask['family'], 'action_tokens': action_tokens,
                         'full_suffix_tokens': suffix_tokens, 'action_nll_sum': action_tokens * loss,
                         'full_suffix_nll_sum': suffix_tokens * loss})
        return {'rank': rank, 'epoch': epoch, 'development_rows': rows}

    candidates = [candidate(16, 1, .5), candidate(16, 2, .25), candidate(16, 3, .25)]
    scores, winner = selection.score_epochs(candidates, masks, scorer)
    assert [entry['epoch'] for entry in scores] == [1, 2, 3]
    assert winner['epoch'] == 2
    assert winner[scorer.PRIMARY] == .25
    assert winner['rows'] == 36 and winner['families'] == 11
    with pytest.raises(ValueError, match='three ordered rank-16'):
        selection.score_epochs([candidate(8, 1, 0), *candidates[1:]], masks, scorer)
    with pytest.raises(ValueError, match='three ordered rank-16'):
        selection.score_epochs(candidates[:2], masks, scorer)
