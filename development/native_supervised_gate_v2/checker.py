"""Independent weighted-ridge reconstruction; no import of the fitting solver."""
import math

TOLERANCE = 1e-10


def ensure(ok, code):
    if not ok:
        raise ValueError(code)


def inner(left, right):
    ensure(len(left) == len(right), "CHECK_WIDTH")
    value = math.fsum(left[j] * right[j] for j in range(len(left)))
    ensure(math.isfinite(value), "CHECK_NONFINITE")
    return value


def eliminate(matrix, rhs):
    """Gaussian elimination with deterministic partial pivoting, not Cholesky."""
    n = len(rhs)
    augmented = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda i: abs(augmented[i][column]))
        ensure(abs(augmented[pivot][column]) > 0, "CHECK_SINGULAR")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        for row in range(column + 1, n):
            factor = augmented[row][column] / augmented[column][column]
            for j in range(column, n + 1):
                augmented[row][j] -= factor * augmented[column][j]
    answer = [0.0] * n
    for i in reversed(range(n)):
        answer[i] = (augmented[i][-1] - math.fsum(augmented[i][j] * answer[j]
                                                 for j in range(i + 1, n))) / augmented[i][i]
    ensure(all(math.isfinite(v) for v in answer), "CHECK_NONFINITE")
    return answer


def reconstruct(rows, labels):
    n = len(rows)
    ensure(n == 32 and len(labels) == n and set(labels) == {-1, 1}, "CHECK_ROWS_LABELS")
    ensure(all(type(y) is int for y in labels), "CHECK_LABEL_TYPE")
    width = len(rows[0])
    ensure(1 <= width <= 1024 and all(len(row) == width for row in rows), "CHECK_WIDTH")
    ensure(all(type(v) in (int, float) and math.isfinite(v) for row in rows for v in row), "CHECK_NONFINITE")
    mean = [math.fsum(float(row[j]) for row in rows) / n for j in range(width)]
    normalized = []
    for row in rows:
        shifted = [float(row[j]) - mean[j] for j in range(width)]
        size = math.sqrt(inner(shifted, shifted))
        ensure(size > 0, "CHECK_ZERO_CENTERED_NORM")
        normalized.append([v / size for v in shifted])
    weights = [0.5 / sum(label == y for label in labels) for y in labels]
    xmean = [math.fsum(weights[i] * normalized[i][j] for i in range(n)) for j in range(width)]
    ymean = math.fsum(weights[i] * labels[i] for i in range(n))
    deviations = [[normalized[i][j] - xmean[j] for j in range(width)] for i in range(n)]
    # Non-symmetric equivalent dual system avoids the fitter's sqrt-weight construction.
    system = [[weights[j] * inner(deviations[i], deviations[j]) + (0.1 if i == j else 0.0)
               for j in range(n)] for i in range(n)]
    solution = eliminate(system, [labels[i] - ymean for i in range(n)])
    coefficient = [math.fsum(weights[i] * solution[i] * deviations[i][j] for i in range(n))
                   for j in range(width)]
    intercept = ymean - inner(coefficient, xmean)
    predictions = [inner(coefficient, row) + intercept for row in normalized]
    # Normal equations separately establish the unique penalized optimum.
    errors = [predictions[i] - labels[i] for i in range(n)]
    intercept_residual = abs(math.fsum(weights[i] * errors[i] for i in range(n)))
    slope_residual = max(abs(math.fsum(weights[i] * errors[i] * normalized[i][j]
                                      for i in range(n)) + 0.1 * coefficient[j]) for j in range(width))
    ensure(intercept_residual <= TOLERANCE and slope_residual <= TOLERANCE, "CHECK_NORMAL_EQUATIONS")
    return mean, coefficient, intercept, predictions, max(intercept_residual, slope_residual)


def verify(rows, labels, model):
    mean, coefficient, intercept, predictions, residual = reconstruct(rows, labels)
    ensure(len(model.mu) == len(mean) and len(model.w) == len(coefficient), "CHECK_MODEL_WIDTH")
    differences = [abs(float(model.mu[j]) - mean[j]) for j in range(len(mean))]
    differences += [abs(float(model.w[j]) - coefficient[j]) for j in range(len(mean))]
    differences += [abs(float(model.b) - intercept)]
    ensure(all(math.isfinite(v) and v <= TOLERANCE for v in differences), "CHECK_PARAMETERS")
    scored = [model.score(row) for row in rows]
    ensure(all(math.isfinite(v) and abs(v - predictions[i]) <= TOLERANCE
               for i, v in enumerate(scored)), "CHECK_PREDICTIONS")
    # Absolute tolerance does not excuse a different discrete route around zero.
    ensure(all((v > 0) == (predictions[i] > 0) for i, v in enumerate(scored)), "CHECK_ROUTES")
    return {"verified": True, "tolerance": TOLERANCE, "normal_equation_residual": residual,
            "maximum_parameter_difference": max(differences), "rows": len(rows),
            "correct": sum((scored[i] > 0) == (labels[i] == 1) for i in range(len(rows)))}
