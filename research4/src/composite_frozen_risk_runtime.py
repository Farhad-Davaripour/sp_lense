"""Explicit role/case-scoped runtime for fixed risk references and replay."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from .io_utils import digest

_KEY=object()
_CURRENT=ContextVar('research4_frozen_risk_scope',default=None)


class _RiskScope:
    def __init__(self,context,scenes,replay,key=None,*,replay_limit=1):
        if key is not _KEY:raise PermissionError('Use a validated risk source/raw reader')
        self._key=key;self.context=context;self.scenes={digest(s) for s in scenes};self.replay=replay
        if replay_limit not in [1,2]:raise PermissionError('Only bounded recorded replay is supported')
        self.actions={};self.seen={};self.active_case=None;self.cursor=0;self.replay_limit=replay_limit
    def require(self,scene=None):
        if not self.replay:self.context.require_runtime()
        if scene is not None:
            from .composite_generalized_world import GeneralizedScene
            if (type(scene) is not GeneralizedScene or scene.role!='development'
                    or scene.soft_stop!=20 or scene.hard_stop!=24 or digest(asdict(scene)) not in self.scenes):
                raise PermissionError('Only exact frozen development cases are permitted; no fitting or sealed worlds')
            if self.replay:
                if self.seen.get(scene.case_id,0)>=self.replay_limit:raise PermissionError('Recorded case replay limit reached')
                if self.active_case==scene.case_id and self.cursor!=len(self.actions[scene.case_id]):
                    raise PermissionError('Finish the current exact recorded replay before another')
                self.seen[scene.case_id]=self.seen.get(scene.case_id,0)+1;self.active_case=scene.case_id;self.cursor=0


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
def frozen_scope(context):
    from .composite_frozen_risk_profile import FrozenRiskContext,_KEY as RISK_KEY
    if not isinstance(context,FrozenRiskContext) or context._risk_key is not RISK_KEY:
        raise PermissionError('Actual complete frozen model/data/detector risk context required')
    context.require_runtime();stage=context.stage()
    ids=None if stage['phase']=='risk_detector_transfer' else set(stage['partition']['case_ids'])
    scenes=[s for s in context.data.scenes() if ids is None or s['case_id'] in ids]
    scope=_RiskScope(context,scenes,False,_KEY);token=_CURRENT.set(scope)
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


@contextmanager
def trace_replay_scope(archive,record,*,helper_pass=False):
    from .composite_frozen_risk_audit import RiskArchive,_ARCHIVE_KEY
    if not isinstance(archive,RiskArchive) or archive._key is not _ARCHIVE_KEY:
        raise PermissionError('Actual complete source/model/native risk archive required')
    key=(record['scene']['case_id'],record['condition'])
    if key not in archive.records or digest(archive.records[key])!=digest(record):
        raise PermissionError('Replay only the exact source-bound recorded trace')
    scope=_RiskScope(archive.context,[record['scene']],True,_KEY,replay_limit=2 if helper_pass else 1)
    scope.actions={key[0]:[step['executed_action'] for step in record['trace']]}
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
