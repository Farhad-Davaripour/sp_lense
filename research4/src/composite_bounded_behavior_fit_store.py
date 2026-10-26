"""Exclusive bounded V4 raw output namespace; no provider/model API."""
from pathlib import Path
from .composite_task_skill_store import StageStore
from .composite_exports import exclusive_output
from .composite_bounded_behavior_fit_profile import MODE,OBJECTIVE


def stage_prefix(stage):
    if stage.get('version')!=MODE or stage.get('phase') not in ['qualification','full_fit']:
        raise PermissionError('Exact V4 training stage required')
    return 'bounded_behavior_training_v4/'+stage['source_commit']+'/'+stage['phase']+'_01'


class BoundedStore(StageStore):
    def __init__(self,stage):
        self.prefix=stage_prefix(stage)
        if stage.get('objective')!=OBJECTIVE:raise PermissionError('Separate V4 objective required')
        self.root=Path(stage['destination']).resolve()
        expected=Path('/kaggle/working/research4_campaign/runs')/stage['source_commit']/'bounded_behavior_training_v4'/(stage['phase']+'_01')
        if self.root!=expected.resolve():raise PermissionError('V4 output escaped its source/phase')
        self.upload=exclusive_output(self.root,'/kaggle/working/research4_campaign/runs')
