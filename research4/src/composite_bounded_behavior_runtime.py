"""Typed inert V4 generation and exact recorded-action replay scopes."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from copy import deepcopy

from .io_utils import digest

_KEY = object()
_CURRENT = ContextVar('research4_bounded_behavior_inert_scope', default=None)


class _Scope:
    def __init__(self, context, scenes, actions, kind, key=None):
        if key is not _KEY: raise PermissionError('Use a validated source/raw-data scope')
        self._key, self.context, self.kind = key, context, kind
        self.scenes = {digest(s) for s in scenes}
        self.actions = deepcopy(actions)
        self.seen, self.active_case, self.cursor, self.outer = {}, None, 0, None

    def check(self, scene):
        from .composite_bounded_behavior_world import Scene, WORLD_VERSION
        if self.kind == 'public_replay':
            if scene.role != 'public_design_fixture' or not scene.identity_seed.startswith('unit:'):
                raise PermissionError('Public replay is not scientific materialization')
        elif (type(scene) is not Scene or scene.behavior_profile != WORLD_VERSION
              or scene.role not in ['training', 'development']):
            raise PermissionError('Only the exact new unsealed V4 scenes are permitted')
        if scene.hard_stop != 24 or scene.soft_stop != 20 or digest(asdict(scene)) not in self.scenes:
            raise PermissionError('Scene identity/role or external deadline changed')

    def begin(self, scene):
        self.check(scene)
        if self.kind == 'data': self.context.require_runtime()
        limit = 2 if self.kind == 'data' else 1
        if self.seen.get(scene.case_id, 0) >= limit:
            raise PermissionError('Bounded reference construction/replay limit reached')
        if self.kind != 'data' and self.active_case is not None:
            if self.cursor != len(self.actions[self.active_case]):
                raise PermissionError('Complete the current recorded replay first')
        self.seen[scene.case_id] = self.seen.get(scene.case_id, 0) + 1
        self.active_case, self.cursor = scene.case_id, 0


def _scope():
    value = _CURRENT.get()
    if value is not None and (not isinstance(value, _Scope) or value._key is not _KEY):
        raise PermissionError('Invalid inert scope')
    return value


def model_calls_forbidden():
    from .composite_bounded_behavior_evaluation_runtime import replay_models_forbidden
    return _scope() is not None or replay_models_forbidden()


def assert_scoped_scene(scene):
    value = _scope()
    if value is None:
        from .composite_bounded_behavior_evaluation_runtime import assert_scene
        return assert_scene(scene)
    value.check(scene)


def begin_public_scene_if_scoped(scene):
    value = _scope()
    if value is not None:
        if value.kind != 'public_replay': raise PermissionError('Public cases cannot enter scientific data')
        value.begin(scene)


def require_scoped_runtime(scene=None):
    value = _scope()
    if value is None: return False
    if scene is not None: value.begin(scene)
    elif value.kind == 'data': value.context.require_runtime()
    return True


def require_generation_scope(scene):
    value = _scope()
    if value is None or value.kind != 'data': raise PermissionError('Generation belongs on admitted Kaggle data runtime')
    value.check(scene); value.context.require_runtime()


@contextmanager
def _entered(value):
    from .composite_frozen_risk_runtime import _CURRENT as old_scope
    if _scope() is not None or old_scope.get() is not None:
        raise PermissionError('Do not mix independent source/replay authorities')
    token = _CURRENT.set(value)
    try: yield value
    finally: _CURRENT.reset(token)


@contextmanager
def data_scope(context):
    from .composite_bounded_behavior_data_profile import BoundedDataContext, _KEY as DATA_KEY
    if not isinstance(context, BoundedDataContext) or context._key is not DATA_KEY:
        raise PermissionError('Complete new source/parent context required')
    context.require_runtime()
    scenes = [s for values in context.profile()['rosters'].values() for s in values]
    with _entered(_Scope(context, scenes, {}, 'data', _KEY)) as scope: yield scope


@contextmanager
def public_replay_scope(episode):
    """Unit fixtures only. This cannot create a data or fitting authority."""
    scene = episode.get('scene', {})
    if (episode.get('public_design_fixture') is not True
            or scene.get('role') != 'public_design_fixture'
            or not scene.get('identity_seed', '').startswith('unit:')):
        raise PermissionError('Only explicit public fixture records may enter this unit scope')
    actions = [s['action'] for s in episode['actions']]
    if len(actions) > 24: raise PermissionError('Public recorded replay is finite')
    with _entered(_Scope(None, [scene], {scene['case_id']: actions}, 'public_replay', _KEY)) as scope:
        yield scope


@contextmanager
def reference_replay_scope(archive):
    """Reconstruct only exact recorded unsealed actions from a checked archive.

    This scope admits local arithmetic replay, never a teacher, new case,
    pretrained call or fitting action. Its key comes from the raw data reader.
    """
    from .composite_bounded_behavior_data_reader import BoundedDataArchive, _ARCHIVE_KEY
    if not isinstance(archive, BoundedDataArchive) or archive._key is not _ARCHIVE_KEY:
        raise PermissionError('Exact source-bound raw V4 archive is required')
    scenes = [s for role in ['training', 'development']
              for s in archive.context.profile()['rosters'][role]]
    with _entered(_Scope(archive.context, scenes, archive.reference_actions,
                         'recorded_replay', _KEY)) as scope:
        yield scope
        if (set(scope.seen) != set(archive.reference_actions)
                or any(count != 1 for count in scope.seen.values())
                or scope.active_case is None
                or scope.cursor != len(scope.actions[scope.active_case])):
            raise PermissionError('Every exact reference must be replayed once and completely')


@contextmanager
def outer_action(scene, action):
    value = _scope()
    if value is None:
        from .composite_bounded_behavior_evaluation_runtime import outer_action as actor_action
        with actor_action(scene,action):yield
        return
    value.check(scene)
    if value.outer is not None: raise PermissionError('An outer native action cannot recurse')
    if value.active_case != scene.case_id: raise PermissionError('Start the exact permitted case first')
    if value.kind != 'data':
        expected = value.actions.get(scene.case_id, [])
        if value.cursor >= len(expected) or digest(action) != digest(expected[value.cursor]):
            raise PermissionError('Read-only replay cannot choose, alter or append an action')
        value.cursor += 1
    value.outer = (scene.case_id, deepcopy(action))
    try: yield
    finally: value.outer = None


def recorded_action_guard(scene, canonical_action):
    value = _scope()
    if value is None: return False
    value.check(scene)
    if value.outer is None or value.outer[0] != scene.case_id:
        raise PermissionError('Only the versioned outer-action dispatcher may enter the backend')
    expected = deepcopy(value.outer[1])
    if isinstance(expected, dict) and expected.get('tool') == 'list': expected['tool'] = 'catalog'
    if digest(expected) != digest(canonical_action):
        raise PermissionError('Canonical backend action differs from the raw declared action')
    return True
