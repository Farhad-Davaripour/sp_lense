"""Read-only replay of already recorded ordinary-task actions; no model/teacher.

This is artifact verification, not a new rollout. A private replay context can
consume only its finite recorded action sequence for its already bound case.
Public fixture readers never establish scientific admission.
"""
from copy import deepcopy
from dataclasses import asdict
from .io_utils import digest
from .composite_task_skill_world import TaskScene,World,VERSION,FAMILIES,TASKS
from .composite_task_skill_package import require_frozen_task_package

_KEY=object()


class _ReplayContext:
    def __init__(self,scene,actions,public,key=None):
        if key is not _KEY:raise PermissionError('Read the actual complete recorded paired case')
        self._key=key;self.scene=deepcopy(scene);self.actions=deepcopy(actions);self.public=public;self.cursor=0;self.bound=False


def public_replay_fixture(scene,actions):
    if (type(scene) is not TaskScene or scene.role!='public_task_fixture' or not scene.identity_seed.startswith('unit:')
            or scene.mechanism not in FAMILIES['training'] or scene.task_structure not in TASKS['training']
            or not isinstance(actions,list) or len(actions)>24):
        raise PermissionError('Only finite ordinary public fixtures may use this engineering helper')
    return _ReplayContext(scene,actions,True,_KEY)


def authorize_recorded_scene(scene,context):
    if (not isinstance(context,_ReplayContext) or context._key is not _KEY
            or type(scene) is not TaskScene or scene.world_version!=VERSION
            or digest(asdict(scene))!=digest(asdict(context.scene)) or context.cursor!=0 or context.bound
            or (context.public and scene.role!='public_task_fixture')
            or (not context.public and scene.role!='development')):
        raise PermissionError('Replay is restricted to one exact already recorded nonsealed case')
    context.bound=True


def authorize_recorded_action(action,context):
    if (not isinstance(context,_ReplayContext) or context._key is not _KEY or context.cursor>=len(context.actions)
            or digest(action)!=digest(context.actions[context.cursor])):
        raise PermissionError('Replay cannot choose, add or alter model actions')
    context.cursor+=1


def _replay_arm(scene,arm,terminal,responses,native_prefixes,actor,eos,vocabulary,previous_cache,*,public=False):
    from .composite_v2_trace_audit import _native
    if (terminal.get('arm')!=arm or terminal.get('case_id')!=scene.case_id
            or digest(terminal.get('scene'))!=digest(asdict(scene))
            or type(terminal.get('responses')) is not int or not 1<=terminal['responses']<=24
            or len(responses)!=terminal['responses'] or len(native_prefixes)!=len(responses)):
        raise ValueError('Complete exact case/arm/response coverage required')
    context=_ReplayContext(scene,[r['submitted_action'] for r in responses],public,_KEY)
    world=World(scene,replay_context=context);logical_forwards=0
    for index,(row,prefix) in enumerate(zip(responses,native_prefixes)):
        if (not world.running or type(row.get('index')) is not int or row['index']!=index or row.get('arm')!=arm or row.get('case_id')!=scene.case_id
                or row.get('group_id')!=scene.group_id or digest(row.get('scene'))!=digest(asdict(scene))
                or row.get('text')!=world.visible()):raise ValueError('Recorded world/prompt order differs')
        native=row['native']
        expected_prefix={'case_id':scene.case_id,'arm':arm,'index':index,'text':row['text'],
                         'native':native,'world_action_not_yet_applied':True}
        if digest(prefix)!=digest(expected_prefix):raise ValueError('Raw native prefix changed after applying the action')
        previous_cache=_native(native,row['text'],arm=='always_on',actor,eos,vocabulary,previous_cache)
        logical_forwards+=len(native['generated_token_ids'])
        expected_action=None if native['actions'] is None else native['actions'][0]
        if digest(row['submitted_action'])!=digest(expected_action):raise ValueError('Executed action differs from native output')
        reply=world.step(row['submitted_action'])
        if digest(row.get('reply'))!=digest(reply) or digest(row.get('outcome'))!=digest(world.outcome()):
            raise ValueError('Recorded world reply or outcome differs from exact action replay')
    expected={'case_id':scene.case_id,'group_id':scene.group_id,'arm':arm,'scene':asdict(scene),
      'responses':len(responses),'response_records_sha256':digest(responses),'operations':world.operations,
      'definitions':world.engine.definitions,'primitive_trace':world.engine.trace,
      'helper_calls':world.engine.executions,'outcome':world.outcome()}
    if (world.running or context.cursor!=len(context.actions) or digest(terminal)!=digest(expected)):
        raise ValueError('Incomplete terminal world/primitive/helper evidence')
    return {'case_id':scene.case_id,'arm':arm,'task_accepted':world.accepted,
      'raw_denied_writes':world.engine.raw_denied_writes,'logical_forwards':logical_forwards,
      'responses':len(responses),'recorded_actions_only':True,'new_model_forwards':0,
      'public_fixture':public,'recorded_native_replay_verified':True,'raw_evidence_audited':False,
      'scientific_validation_admitted':False},previous_cache


def replay_pair(files,context,actor,eos,vocabulary):
    # Called only by the complete raw-model/metadata reader. The context itself
    # is still the opaque full-fit/data/source binding, never a receipt flag.
    from .composite_task_skill_inference import TaskEvaluationContext,_KEY as CONTEXT_KEY
    if not isinstance(context,TaskEvaluationContext) or context._key is not CONTEXT_KEY:
        raise PermissionError('Genuine original model/data/source context required')
    stage=context.stage();value=require_frozen_task_package(context.model)
    case=stage['partition']['case_ids'][0]
    scene=TaskScene(**next(r for r in value['development_roster']['cases'] if r['case_id']==case))
    allowed=['trajectories/'+case+'/'+arm+'/' for arm in ['base','always_on']]
    if any(n.startswith('trajectories/') and not any(n.startswith(p) for p in allowed) for n in files.manifest):
        raise PermissionError('Foreign case/arm trajectories cannot enter this source-bound paired stage')
    results=[];previous=None
    for arm in ['base','always_on']:
        prefix='trajectories/'+case+'/'+arm+'/'
        terminal=files.json(prefix+'TERMINAL.json');count=terminal['responses']
        if type(count) is not int or not 1<=count<=24:raise ValueError('Finite complete response count required')
        expected={prefix+'TERMINAL.json'}
        expected|={prefix+kind+'_'+str(i).zfill(2)+'.json' for kind in ['native','response'] for i in range(count)}
        if {n for n in files.manifest if n.startswith(prefix)}!=expected:raise ValueError('Missing or extra native trajectory members')
        rows=[files.json(prefix+'response_'+str(i).zfill(2)+'.json') for i in range(count)]
        native=[files.json(prefix+'native_'+str(i).zfill(2)+'.json') for i in range(count)]
        result,previous=_replay_arm(scene,arm,terminal,rows,native,actor,eos,vocabulary,previous)
        results.append(result)
    return results
