"""Explicit role/case-scoped runtime for fixed risk references and replay."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from .io_utils import digest

_KEY=object()
_CURRENT=ContextVar('research4_frozen_risk_scope',default=None)


class _RiskScope:
    def __init__(self,context,scenes,replay,key=None):
        if key is not _KEY:raise PermissionError('Use a validated risk source/raw reader')
        self._key=key;self.context=context;self.scenes={digest(s) for s in scenes};self.replay=replay
        self.actions={};self.seen=set();self.active_case=None;self.cursor=0
    def require(self,scene=None):
        if not self.replay:self.context.require_runtime()
        if scene is not None:
            from .composite_generalized_world import GeneralizedScene
            if (type(scene) is not GeneralizedScene or scene.role!='development'
                    or scene.soft_stop!=20 or scene.hard_stop!=24 or digest(asdict(scene)) not in self.scenes):
                raise PermissionError('Only exact frozen development cases are permitted; no fitting or sealed worlds')
            if self.replay:
                if scene.case_id in self.seen:raise PermissionError('A recorded reference case may be replayed once per scope')
                self.seen.add(scene.case_id);self.active_case=scene.case_id;self.cursor=0


@contextmanager
def data_scope(context):
    from .composite_frozen_risk_data_profile import RiskDataContext,_KEY as SOURCE_KEY
    if not isinstance(context,RiskDataContext) or context._key is not SOURCE_KEY:
        raise PermissionError('Actual validated development-only source context required')
    context.require_runtime()
    scope=_RiskScope(context,context.profile()['development_roster'],False,_KEY)
    token=_CURRENT.set(scope)
    try:yield scope
    finally:_CURRENT.reset(token)


@contextmanager
def reference_replay_scope(archive):
    from .composite_frozen_risk_data_reader import RiskDataArchive,_ARCHIVE_KEY
    if not isinstance(archive,RiskDataArchive) or archive._key is not _ARCHIVE_KEY:
        raise PermissionError('Complete actual source-bound raw archive required for recorded-action replay')
    scope=_RiskScope(archive.context,archive.context.profile()['development_roster'],True,_KEY)
    scope.actions=archive.reference_actions
    token=_CURRENT.set(scope)
    try:yield scope
    finally:_CURRENT.reset(token)


def require_scoped_runtime(scene=None):
    scope=_CURRENT.get()
    if scope is None:return False
    if not isinstance(scope,_RiskScope) or scope._key is not _KEY:raise PermissionError('Invalid risk runtime scope')
    scope.require(scene);return True


def recorded_action_guard(scene,action):
    scope=_CURRENT.get()
    if scope is None:return
    if not isinstance(scope,_RiskScope) or scope._key is not _KEY:raise PermissionError('Invalid recorded-action scope')
    if not scope.replay:return
    expected=scope.actions.get(scene.case_id)
    if (scope.active_case!=scene.case_id or expected is None or scope.cursor>=len(expected)
            or digest(action)!=digest(expected[scope.cursor])):
        raise PermissionError('Read-only reference replay cannot choose, alter or add an action')
    scope.cursor+=1
