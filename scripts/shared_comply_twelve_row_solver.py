"""Bound-only extension of the immutable exhaustive eight-row solver to twelve.

The selected-solution Decimal audit is deliberately outside this runner module.
No infeasibility certificate is manufactured when enumeration is unresolved.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from scripts import shared_preserve_eight_row_solver as previous

MAXIMUM_CONSTRAINTS = 12
MAXIMUM_MASKS = 4096


def extend_bound(source):
    """Change exactly the upper bound in the original finite-system guard."""
    tree = ast.parse(textwrap.dedent(source))
    constants = [
        node for node in ast.walk(tree) if isinstance(node, ast.Constant) and node.value == 8
    ]
    comparisons = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and len(node.ops) == 3
        and isinstance(node.ops[-1], ast.LtE)
        and node.comparators[-1] in constants
        and ast.unparse(node) == "0 < len(A) == len(b) <= 8"
    ]
    previous.io.require(
        len(tree.body) == 1
        and isinstance(tree.body[0], ast.FunctionDef)
        and tree.body[0].name == "solve"
        and len(constants) == len(comparisons) == 1,
        "single original eight-row bound; no algorithm adaptation",
    )
    constants[0].value = MAXIMUM_CONSTRAINTS
    namespace = dict(vars(previous))
    namespace["__name__"] = __name__
    exec(compile(tree, __file__ + "::<bound-only>", "exec"), namespace)  # noqa: S102 - trusted immutable definition; exact bound-only adaptation.
    return namespace["solve"]


solve = extend_bound(inspect.getsource(previous.solve))
dot, norm = previous.dot, previous.norm
