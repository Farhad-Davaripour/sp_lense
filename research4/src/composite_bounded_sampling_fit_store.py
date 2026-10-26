"""Exclusive bounded V4 raw output namespace; no provider/model API."""
from pathlib import Path
from .composite_task_skill_store import StageStore
from .composite_exports import exclusive_output
from .composite_bounded_sampling_fit_profile import MODE,DIRECT_MODE,OBJECTIVE


def stage_prefix(stage):
    if (stage.get('version') not in [MODE,DIRECT_MODE] or stage.get('phase') not in ['qualification','full_fit']
            or stage.get('version')==DIRECT_MODE and stage.get('phase')!='full_fit'):
        raise PermissionError('Exact V4 training stage required')
    suffix='direct_full_fit_01' if stage['version']==DIRECT_MODE else stage['phase']+'_01'
    return 'bounded_behavior_sampling_training_a1/'+stage['source_commit']+'/'+suffix


class BoundedStore(StageStore):
    def __init__(self,stage):
        self.prefix=stage_prefix(stage)
        if stage.get('objective')!=OBJECTIVE:raise PermissionError('Separate V4 objective required')
        self.root=Path(stage['destination']).resolve()
        suffix='direct_full_fit_01' if stage['version']==DIRECT_MODE else stage['phase']+'_01'
        expected=Path('/kaggle/working/research4_campaign/runs')/stage['source_commit']/'bounded_behavior_sampling_training_a1'/suffix
        if self.root!=expected.resolve():raise PermissionError('V4 output escaped its source/phase')
        self.upload=exclusive_output(self.root,'/kaggle/working/research4_campaign/runs')
