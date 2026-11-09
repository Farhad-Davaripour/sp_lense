"""Pure development-only likelihood aggregation and four-setting selection."""
import hashlib
import json
import math

GRID = (
    ('rank04_lr5e-6',4,0.000005), ('rank04_lr1e-5',4,0.00001),
    ('rank08_lr5e-6',8,0.000005), ('rank08_lr1e-5',8,0.00001),
)
CRITERION = 'pooled_dev_target_token_nll'


def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def sha_bytes(data): return hashlib.sha256(data).hexdigest()


def pool_development(rows, expected_ids):
    if len(rows) != 144 or len(expected_ids) != 144 or len(set(expected_ids)) != 144:
        raise ValueError('Exactly 144 distinct declared development examples are required')
    nll_sum = 0.0; token_count = 0
    for row, expected in zip(rows,expected_ids):
        if set(row) != {'row_id','target_tokens','nll_sum'} or row['row_id'] != expected:
            raise ValueError('Missing, duplicate or out-of-order development likelihood')
        count, total = row['target_tokens'], row['nll_sum']
        if type(count) is not int or count < 1 or isinstance(total,bool) or not isinstance(total,(int,float)) or not math.isfinite(total) or total < 0:
            raise ValueError('Invalid development target-token likelihood')
        nll_sum += total; token_count += count
    if not math.isfinite(nll_sum): raise ValueError('Nonfinite pooled development likelihood')
    return {'rows':144,'target_tokens':token_count,'nll_sum':nll_sum,CRITERION:nll_sum/token_count}


def choose(trials, expected_ids, dataset_manifest_sha256, expected_target_tokens=None):
    """Consume complete fresh-base trial receipts; never launch/retry a trial."""
    if len(trials) != len(GRID): raise ValueError('All four predeclared trials must complete')
    results = []
    common_keys=('cohort_sha256','training_rows_sha256','training_order_ids_sha256','loss_version','source_sha256','protocol','config')
    reference={key:trials[0].get(key)for key in common_keys}
    if any(value is None for value in reference.values())or reference['loss_version']!='shifted_suffix_P_le_position_lt_L.v2':
        raise ValueError('Trial common source/order/mask provenance missing')
    reference_tokens=expected_target_tokens if expected_target_tokens is not None else [row['target_tokens']for row in trials[0]['development_rows']]
    for trial, (trial_id,rank,lr) in zip(trials,GRID):
        if (trial.get('trial_id'),trial.get('rank'),trial.get('learning_rate'),trial.get('status'),
            trial.get('initialization'),trial.get('num_layers'),trial.get('completed_steps'),trial.get('expected_steps'),
            trial.get('dataset_manifest_sha256')) != (trial_id,rank,lr,'complete','fresh_pinned_base',64,48,48,dataset_manifest_sha256):
            raise ValueError('Trial identity, completion or data binding differs')
        pooled = pool_development(trial['development_rows'],expected_ids)
        if ({key:trial.get(key)for key in common_keys}!=reference
            or [row['target_tokens']for row in trial['development_rows']]!=reference_tokens
            or trial.get('development_no_optimizer_updates')is not True or trial.get('trainable_state_unchanged_during_dev')is not True
            or (trial.get('optimizer_step_before_dev'),trial.get('optimizer_step_after_dev'))!=(48,48)):
            raise ValueError('Trial recipe, row masks or development no-update provenance differs')
        if trial.get('development_summary') != pooled: raise ValueError('Trial summary differs from actual row likelihoods')
        results.append({'trial_id':trial_id,'rank':rank,'learning_rate':lr,**pooled})
    if len({row['target_tokens'] for row in results}) != 1: raise ValueError('Development masking/token cohort differs across trials')
    winner = min(results,key=lambda row:(row[CRITERION],row['rank'],row['learning_rate'],row['trial_id']))
    return {'schema':'research4.sp-lens-game-tuning-selection.v2','criterion':CRITERION,
        'winner':{'trial_id':winner['trial_id'],'rank':winner['rank'],'learning_rate':winner['learning_rate'],CRITERION:winner[CRITERION]},
        'dataset_manifest_sha256':dataset_manifest_sha256,'tested_settings':results,
        'all_four_trials_complete':True,'selection_uses_development_only':True,'final_game_opened':False,
        'exact_tiebreak':['smaller_rank','lower_learning_rate','lexicographic_trial_id'],
        'final_initialization':'fresh_pinned_base','final_resume_trial':False,
        'claim':'Best among four tested settings; not a global optimum, game/preservation/exploitation result or final model checkpoint.'}
