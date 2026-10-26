"""Frozen native generation and inert backend semantics across the evaluation port."""
from pathlib import Path
import ast
from .io_utils import digest
from .composite_task_skill_semantics import semantic_fingerprint,_stable_ast


def fingerprint(root):
    root=Path(root)
    world=ast.parse((root/'src/composite_bounded_behavior_world.py').read_text(encoding='utf-8-sig'))
    cls=next(n for n in world.body if isinstance(n,ast.ClassDef) and n.name=='World')
    methods={n.name:digest(_stable_ast(n)) for n in cls.body if isinstance(n,ast.FunctionDef)
      and n.name in ['__init__','apply_record_effects','request_visible','visible','_record_step','outcome']}
    declarations=[n for n in world.body if
      (isinstance(n,ast.ClassDef) and n.name=='Scene') or
      (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['PROFILE','WORLD_VERSION','RETURN_TICK'] for t in n.targets))]
    return {'historical_native_task_semantics':semantic_fingerprint(root),
      'V4_native_interface':digest((root/'src/composite_bounded_behavior_interface.py').read_text(encoding='utf-8-sig')),
      'V4_request_and_outcome_methods':methods,'V4_scene_clock_profile_declarations':digest(_stable_ast(declarations))}


def require_frozen(expected,root):
    if digest(fingerprint(root))!=digest(expected):
        raise PermissionError('Fitted native generation, request/scoring or primitive/task semantics changed')
