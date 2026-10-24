"""Conditional full-fit metadata only; no submit, model load or warm-start API."""
import re
from .composite_task_skill_training import DATA_ARCHIVE_SHA,qualification_plan


def full_fit_plan(source_commit,qualification):
    from .composite_task_skill_model_audit import require_qualification
    proof=require_qualification(qualification)
    if not isinstance(source_commit,str) or not re.fullmatch('[0-9a-f]{40}',source_commit):
        raise ValueError('Exact reviewed source commit required')
    value=qualification_plan(source_commit)
    value.update({'source_commit':source_commit,
            'phase':'full_fit','notebook':'farhad89/research4-task-skill-fit-v1',
            'objective':'ordinary_authorized_task_skills_only','source_data_archive_sha256':DATA_ARCHIVE_SHA,
            'source_training_rows':1188,'optimizer_updates':64,'accumulation':2,
            'seed':202610212,'learning_rate':0.0005,'weight_decay':0.0,'clip_norm':1.0,
            'sampling':'registered_seeded_training_sampler','visible_devices':'0,1','scientific_GPU_count':2,
            'hard_timeout_seconds':42000,'worker_seconds':40800,'export_reserve_seconds':900,
            'destination':'/kaggle/working/research4_campaign/runs/'+source_commit+'/task_skill_training_v1/full_fit_01',
            'qualification_source_commit':proof['source_commit'],
            'qualification_run_id':proof['provider_run_id'],
            'qualification_raw_archive_sha256':proof['raw_archive_sha256'],
            'checkpoint_initializes_full_fit':False,'optimizer_sampler_or_RNG_warm_start':False,
            'risk_examples_or_outcomes_used_for_fitting_or_selection':False,
            'heldout_content_read':False,'automatic_scientific_retries':0,
            'metadata_plan_alone_admits_launch':False})
    return value
