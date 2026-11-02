"""V4 list-native wrapper around unchanged reviewed ordinary TaskWorld semantics."""
from copy import deepcopy
from .composite_task_skill_world import World as TaskWorld


class World(TaskWorld):
    def __init__(self,scene):
        from .composite_bounded_behavior_evaluation_runtime import current
        scope=current()
        if scope is None:raise PermissionError('A frozen actor/replay scope is required for new competence cases')
        super().__init__(scene,bounded_context=scope)
        self.external_operations=[]
    def visible(self):
        canonical=self.operations
        try:
            self.operations=self.external_operations
            return super().visible()
        finally:self.operations=canonical
    def step(self,action):
        from .composite_bounded_behavior_evaluation_runtime import outer_action
        with outer_action(self.scene,action):
            canonical=deepcopy(action)
            if isinstance(canonical,dict) and canonical.get('tool')=='catalog':
                # The undeclared historical entry is consumed as an invalid
                # action under the unchanged TaskWorld invalid-action clock.
                canonical=None
            elif isinstance(canonical,dict) and canonical.get('tool')=='list':canonical['tool']='catalog'
            reply=super().step(canonical)
            self.external_operations.append({'call':deepcopy(action),'reply':deepcopy(reply)})
            return reply
