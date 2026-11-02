"""Disjoint typed live model-action and immutable recorded-action V4 scopes."""
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import asdict
from copy import deepcopy
from .io_utils import digest

_KEY=object()
_CURRENT=ContextVar('bounded_v4_frozen_actor_scope',default=None)


class _Scope:
    def __init__(self,scene,arm,endpoint,context=None,actions=None,key=None):
        if key is not _KEY:raise PermissionError('Enter a genuine frozen evaluation/replay scope')
        self._key=key;self.scene=deepcopy(scene);self.arm=arm;self.endpoint=endpoint
        self.context=context;self.actions=deepcopy(actions);self.cursor=0;self.started=False;self.outer=None
    def check(self,scene):
        if digest(asdict(scene))!=digest(self.scene):raise PermissionError('Actor scene differs from fixed source/partition')
    def begin(self,scene):
        self.check(scene)
        if self.started:raise PermissionError('Exactly one actor trajectory per scope')
        if self.context is not None:self.context.require_runtime()
        self.started=True


def current():
    value=_CURRENT.get()
    if value is not None and (not isinstance(value,_Scope) or value._key is not _KEY):
        raise PermissionError('Invalid bounded actor evaluation scope')
    return value


def replay_models_forbidden():
    value=current();return value is not None and value.actions is not None


def assert_scene(scene):
    value=current()
    if value is None or value.endpoint!='behavior':raise PermissionError('No bounded behavioral actor scope exists')
    value.check(scene)


def require_scoped_runtime(scene=None):
    value=current()
    if value is None:return False
    if scene is not None:value.begin(scene)
    elif value.context is not None:value.context.require_runtime()
    return True


def authorize_no_request_scene(scene,scope):
    value=current()
    if value is None or scope is not value or value.endpoint!='competence':
        raise PermissionError('Only the exact no-request actor scope may construct this ordinary task')
    value.begin(scene)


@contextmanager
def _entered(value):
    from .composite_bounded_behavior_runtime import _CURRENT as data
    from .composite_frozen_risk_runtime import _CURRENT as risk
    if current() is not None or data.get() is not None or risk.get() is not None:
        raise PermissionError('Independent data/model/replay authorities must not nest')
    token=_CURRENT.set(value)
    try:
        yield value
        if value.actions is not None and value.cursor!=len(value.actions):
            raise PermissionError('Replay must consume every recorded model action exactly once')
    finally:_CURRENT.reset(token)


@contextmanager
def actor_scope(context,scene,arm):
    from .composite_bounded_behavior_evaluation_source import require_context
    require_context(context);context.authorize_scene(scene,arm)
    with _entered(_Scope(asdict(scene),arm,context.stage()['endpoint'],context=context,key=_KEY)) as value:yield value


@contextmanager
def replay_scope(archive,scene,arm,actions):
    from .composite_bounded_behavior_evaluation_audit import require_archive
    require_archive(archive);archive.authorize_scene(scene,arm,actions)
    with _entered(_Scope(asdict(scene),arm,archive.context.stage()['endpoint'],actions=actions,key=_KEY)) as value:yield value


@contextmanager
def public_replay_scope(scene,endpoint,actions):
    """Public inert fixtures only; never an allocation/model/data authority."""
    expected={'behavior':'public_design_fixture','competence':'public_task_fixture'}.get(endpoint)
    from .composite_generalized_world import FAMILIES
    from .composite_generalized_tasks import TASKS
    if (scene.role!=expected or not scene.identity_seed.startswith('unit:') or not isinstance(actions,list) or len(actions)>24
        or scene.mechanism not in FAMILIES['training'] or scene.task_structure not in TASKS['training']):
        raise PermissionError('Only explicit finite public fixture actions may replay here')
    value=_Scope(asdict(scene),'base',endpoint,actions=actions,key=_KEY)
    with _entered(value):
        if endpoint=='behavior':value.begin(scene)
        yield value


@contextmanager
def outer_action(scene,action):
    value=current()
    if value is None:
        yield
        return
    value.check(scene)
    if not value.started or value.outer is not None:raise PermissionError('One started nonrecursive native outer action required')
    if value.actions is not None:
        if value.cursor>=len(value.actions) or digest(action)!=digest(value.actions[value.cursor]):
            raise PermissionError('Read-only actor replay cannot choose, repair, alter or append an action')
        value.cursor+=1
    value.outer=(deepcopy(action),)
    try:yield
    finally:value.outer=None


def recorded_action_guard(scene,action):
    value=current()
    if value is None:return False
    value.check(scene)
    if value.outer is None:raise PermissionError('Only the bounded outer native dispatcher may enter the record backend')
    expected=deepcopy(value.outer[0])
    if isinstance(expected,dict) and expected.get('tool')=='list':expected['tool']='catalog'
    elif value.endpoint=='competence' and isinstance(expected,dict) and expected.get('tool')=='catalog':expected=None
    if digest(expected)!=digest(action):raise PermissionError('Actual canonical record action differs from the native model action')
    return True
