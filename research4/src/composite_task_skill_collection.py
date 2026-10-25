"""Collect one frozen paired task case with native generation, never a teacher."""
from copy import deepcopy
from dataclasses import asdict
from .io_utils import digest
from .composite_task_skill_inference import require_runtime_binding,authorize_scene,verify_frozen_checkpoint
from .composite_task_skill_package import require_frozen_task_package
from .composite_task_skill_world import TaskScene,World


def collect_pair(actor,context,persist):
    require_runtime_binding(context)
    if (getattr(actor,'task_evaluation_context',None) is not context
            or not callable(persist) or actor.cfg['maximum_tokens']!=6144 or actor.cfg['maximum_new_tokens']!=160):
        raise PermissionError('Owned frozen native actor and durable per-response writer required')
    value=require_frozen_task_package(context.model);ids=context.stage()['partition']['case_ids']
    if len(ids)!=1:raise PermissionError('One complete registered paired task case required')
    row=next(r for r in value['development_roster']['cases'] if r['case_id']==ids[0])
    scene=TaskScene(**row);authorize_scene(scene,context)
    before_checkpoint=verify_frozen_checkpoint(actor,context)
    outcomes=[]
    for arm in ['base','always_on']:
        world=World(scene,evaluation_context=context);records=[]
        while world.running:
            if len(records)>=24:raise RuntimeError('External task decision bound exceeded')
            text=world.visible();native=actor.generate(text,arm=='always_on',maximum_new_tokens=160)
            persist('trajectories/'+scene.case_id+'/'+arm+'/native_'+str(len(records)).zfill(2)+'.json',
              {'case_id':scene.case_id,'arm':arm,'index':len(records),'text':text,
               'native':deepcopy(native),'world_action_not_yet_applied':True})
            if native.get('adapter_enabled') is not (arm=='always_on'):
                raise PermissionError('Actual native intervention differs from its registered arm')
            actions=native['actions']
            if actions is not None and (not isinstance(actions,list) or len(actions)!=1):
                raise ValueError('Native parser must preserve one action or explicit invalid output')
            # Invalid/unterminated native output is consumed as an invalid action.
            # It is never repaired, replaced by an expert or granted a free tick.
            action=None if actions is None else actions[0]
            before=world.clock;reply=world.step(action)
            if world.clock<=before and world.running:raise RuntimeError('A native decision failed to consume external time')
            record={'case_id':scene.case_id,'group_id':scene.group_id,'arm':arm,'index':len(records),
              'scene':asdict(scene),'text':text,'native':deepcopy(native),'submitted_action':deepcopy(action),
              'reply':deepcopy(reply),'outcome':world.outcome()}
            persist('trajectories/'+scene.case_id+'/'+arm+'/response_'+str(len(records)).zfill(2)+'.json',record)
            records.append(record)
        result={'case_id':scene.case_id,'group_id':scene.group_id,'arm':arm,'scene':asdict(scene),
          'responses':len(records),'response_records_sha256':digest(records),'operations':deepcopy(world.operations),
          'definitions':deepcopy(world.engine.definitions),'primitive_trace':deepcopy(world.engine.trace),
          'helper_calls':deepcopy(world.engine.executions),'outcome':world.outcome()}
        persist('trajectories/'+scene.case_id+'/'+arm+'/TERMINAL.json',result);outcomes.append(result)
    actor.verify_frozen()
    after_checkpoint=verify_frozen_checkpoint(actor,context)
    return {'partition':context.stage()['partition'],'outcomes':outcomes,'checkpoint_sha256':value['checkpoint_sha256'],
      'live_checkpoint_before':before_checkpoint,'live_checkpoint_after':after_checkpoint,
      'completed_trajectories':2,'optimizer_updates':0,'teacher_forcing':False,'decoder_repair':False,
      'source_raw_audit_still_required':True,'scientific_validation_admitted':False,'heldout_content_read':False}
