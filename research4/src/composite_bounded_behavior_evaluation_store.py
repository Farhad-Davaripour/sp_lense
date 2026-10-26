"""Source/endpoint/partition-exclusive frozen actor outputs."""
from pathlib import Path
from .composite_task_skill_store import StageStore
from .composite_exports import exclusive_output
from .composite_bounded_behavior_evaluation_profile import stage_prefix


class EvaluationStore(StageStore):
    def __init__(self,stage):
        self.prefix=stage_prefix(stage);self.root=Path(stage['destination']).resolve()
        expected=Path('/kaggle/working/research4_campaign/runs')/stage['source_commit']/'bounded_actor_evaluation_v4'/stage['endpoint']/('partition_'+str(stage['partition']['index']).zfill(2))
        if self.root!=expected.resolve():raise PermissionError('Actor output escaped exact endpoint/source/partition')
        self.upload=exclusive_output(self.root,'/kaggle/working/research4_campaign/runs')
