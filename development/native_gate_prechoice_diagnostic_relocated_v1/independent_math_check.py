"""Independent reference math for the parallel math core diagnostic.

Standalone reference implementation. It must NOT import or call production
scoring math. Scope is narrow: dense-equivalence / mutation review against the
real model or scorer is deferred and is NOT proved by this module.
"""

import math

VIEW_COUNT = 2
COORD_COUNT = 1024
HEAD_COUNT = 3


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _checked_vector(values, length, label):
    if not isinstance(values, (list, tuple)):
        raise ValueError("%s must be a list or tuple" % label)
    if len(values) != length:
        raise ValueError("%s must have length %d, got %d" % (label, length, len(values)))
    out = []
    for index, value in enumerate(values):
        if not _is_number(value):
            raise ValueError("%s[%d] must be a real number, got %s" % (label, index, type(value).__name__))
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("%s[%d] must be finite" % (label, index))
        out.append(number)
    return out


def case_scores(views, mu, heads):
    """Score one 2-view case against 3 heads and return route ON/OFF.

    views: (view_a, view_b), each 1024 finite reals.
    mu:    1024 finite reals (per-coordinate mean).
    heads: 3 x (weight[1024], bias) finite reals.
    """
    if not isinstance(views, (list, tuple)) or len(views) != VIEW_COUNT:
        raise ValueError("views must be a list/tuple of %d vectors" % VIEW_COUNT)
    checked_views = [_checked_vector(v, COORD_COUNT, "views[%d]" % i) for i, v in enumerate(views)]
    mean = _checked_vector(mu, COORD_COUNT, "mu")

    if not isinstance(heads, (list, tuple)) or len(heads) != HEAD_COUNT:
        raise ValueError("heads must be a list/tuple of %d heads" % HEAD_COUNT)
    checked_heads = []
    for i, head in enumerate(heads):
        if not isinstance(head, (list, tuple)) or len(head) != 2:
            raise ValueError("heads[%d] must be a (weight, bias) pair" % i)
        weights = _checked_vector(head[0], COORD_COUNT, "heads[%d].weight" % i)
        bias = head[1]
        if not _is_number(bias):
            raise ValueError("heads[%d].bias must be a real number" % i)
        bias = float(bias)
        if not math.isfinite(bias):
            raise ValueError("heads[%d].bias must be finite" % i)
        checked_heads.append((weights, bias))

    a, b = checked_views
    averaged = [0.5 * a[j] + 0.5 * b[j] for j in range(COORD_COUNT)]
    delta = [averaged[j] - mean[j] for j in range(COORD_COUNT)]
    squared = math.fsum(d * d for d in delta)
    norm = math.sqrt(squared)
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError("centered delta norm must be finite and nonzero")

    x = [d / norm for d in delta]

    scores = []
    for weights, bias in checked_heads:
        score = math.fsum(w * xj for w, xj in zip(weights, x)) + bias
        if not math.isfinite(score):
            raise ValueError("head score must be finite")
        scores.append(score)

    route = "ON" if all(score > 0.0 for score in scores) else "OFF"
    return {"scores": scores, "route": route}
