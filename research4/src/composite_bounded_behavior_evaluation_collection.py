"""Frozen paired BASE/always-ON native collection; no teacher or detector calls."""
from dataclasses import asdict
from copy import deepcopy
from .io_utils import digest
from .composite_bounded_behavior_evaluation_source import require_context
from .composite_bounded_behavior_evaluation_runtime import actor_scope
from .composite_bounded_behavior_evaluation_inference import verify_checkpoint


class ModelContextCapacityFailure(RuntimeError):pass


def generate_native(actor,text,persist,case,arm,index):
    """An oversized model-authored state is an unexecuted partial failure."""
    ids=actor.prompt_ids(text)
    if len(ids)>6144:
        persist('capacity_failures/'+case+'/'+arm+'/state_'+str(index).zfill(2)+'.json',
          {'case_id':case,'arm':arm,'index':index,'text':text,'input_token_ids':ids,
           'context_ceiling':6144,'model_forward_attempted':False,'world_action_applied':False,
           'context_truncated':False,'simulator_terminal_invented':False,'remaining_coverage':'unknown'})
        raise ModelContextCapacityFailure('Actual model-authored state exceeds unchanged6144 context; preserve partial raw, no retry/repair')
    return actor.generate(text,arm=='always_on',maximum_new_tokens=160)


def world_for(scene,endpoint):
    if endpoint=='behavior':
        from .composite_bounded_behavior_world import World
    elif endpoint=='competence':
        from .composite_bounded_behavior_competence_world import World
    else:raise PermissionError('Exact actor endpoint required')
    return World(scene)


def scene_for(value,endpoint):
    if endpoint=='behavior':
        from .composite_bounded_behavior_world import Scene
    elif endpoint=='competence':
        from .composite_task_skill_world import TaskScene as Scene
    else:raise PermissionError('Exact actor endpoint required')
    return Scene(**value)


def terminal_record(world,scene,arm,records):
    return {'case_id':scene.case_id,'group_id':scene.group_id,'arm':arm,'scene':asdict(scene),
      'responses':len(records),'response_records_sha256':digest(records),
      'operations':deepcopy(world.operations),'external_operations':deepcopy(world.external_operations),
      'definitions':deepcopy(world.engine.definitions),'primitive_trace':deepcopy(world.engine.trace),
      'helper_calls':deepcopy(world.helper_calls if hasattr(world,'helper_calls') else world.engine.executions),
      'engine_executions':deepcopy(world.engine.executions),'outcome':world.outcome()}


def collect_pair(actor,context,persist,case_id):
    require_context(context);context.require_runtime();stage=context.stage()
    if (getattr(actor,'bounded_evaluation_context',None) is not context
            or case_id not in stage['partition']['case_ids']):
        raise PermissionError('Exact frozen actor/paired case required')
    descriptor=next(r for r in context.roster() if r['case_id']==case_id)
    scene=scene_for(descriptor,stage['endpoint']);before=verify_checkpoint(actor,context);outcomes=[]
    for arm in ['base','always_on']:
        with actor_scope(context,scene,arm):
            world=world_for(scene,stage['endpoint']);records=[]
            while world.running:
                if len(records)>=24:raise RuntimeError('External finite native response bound exceeded')
                visible=world.visible();text=visible[0] if isinstance(visible,tuple) else visible
                native=generate_native(actor,text,persist,case_id,arm,len(records))
                prefix={'case_id':case_id,'group_id':scene.group_id,'arm':arm,'index':len(records),
                  'text':text,'native':deepcopy(native),'world_action_not_yet_applied':True}
                persist('trajectories/'+case_id+'/'+arm+'/native_'+str(len(records)).zfill(2)+'.json',prefix)
                if native.get('adapter_enabled') is not (arm=='always_on'):
                    raise PermissionError('Actual adapter intervention differs from the paired arm')
                calls=native['actions']
                if calls is not None and (not isinstance(calls,list) or len(calls)!=1):
                    raise ValueError('Preserve one native action or explicit invalid output')
                action=None if calls is None else calls[0]
                clock=world.clock;reply=world.step(action)
                if world.clock<=clock and world.running:raise RuntimeError('Native action received a free tick')
                value={'case_id':case_id,'group_id':scene.group_id,'arm':arm,'index':len(records),
                  'scene':descriptor,'text':text,'native':deepcopy(native),'submitted_action':deepcopy(action),
                  'reply':deepcopy(reply),'clock_before':clock,'clock_after':world.clock,'outcome':world.outcome()}
                persist('trajectories/'+case_id+'/'+arm+'/response_'+str(len(records)).zfill(2)+'.json',value)
                records.append(value)
            terminal=terminal_record(world,scene,arm,records)
            persist('trajectories/'+case_id+'/'+arm+'/TERMINAL.json',terminal);outcomes.append(terminal)
    actor.verify_frozen();after=verify_checkpoint(actor,context)
    return {'case_id':case_id,'outcomes':outcomes,'completed_trajectories':2,'optimizer_updates':0,
      'live_checkpoint_before':before,'live_checkpoint_after':after,'teacher_forcing':False,
      'teacher_compiler_called':False,'decoder_repair':False,'scientific_validation_admitted':False,
      'heldout_content_read':False}
