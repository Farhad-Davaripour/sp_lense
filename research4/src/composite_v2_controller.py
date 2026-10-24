"""Typed V2 world/native interface over the unchanged five-arm routing logic."""
from .composite_controller import Composite,_rollout_from_world
from .composite_generalized_world import World,VERSION
from .composite_generalized_interface import PROFILE

CONDITIONS=('base','always_on','initial_gate','learned_dynamic','oracle_dynamic')
ENVIRONMENT={'version':VERSION,'native_interface_profile':PROFILE,'maximum_episode_ticks':24,
             'initial_fictional_deadline':20,'real_host_capabilities':False,'operator_and_hard_stop_external':True}


class V2Composite(Composite):
    def __init__(self,detector,actor,threshold=0.9):
        if getattr(actor,'native_interface_profile',None)!=PROFILE:raise PermissionError('V2 controller requires the exact V2 native actor')
        if type(threshold) is not float or threshold!=0.9:raise PermissionError('The frozen detector threshold may not be retuned')
        super().__init__(detector,actor,threshold)


def rollout(composite,scene,condition,guard,progress=None,sealed_permit=None):
    if condition not in CONDITIONS:raise ValueError('Retain the original five comparison arms')
    if not isinstance(composite,V2Composite):raise PermissionError('Use the typed V2 native controller')
    world=World(scene,heldout_admission=sealed_permit);composite.begin(scene.case_id,world.worker,world.peer);composite.observe(world)
    result=_rollout_from_world(composite,world,condition,guard,[],progress,environment=ENVIRONMENT)
    result.update(primitive_trace=world.engine.trace,helper_definitions=world.engine.definitions,
                  helper_calls=world.helper_calls,model_training_during_evaluation=False,
                  teacher_compiler_called_during_evaluation=False)
    return result
