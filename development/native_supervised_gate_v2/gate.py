"""Fixed ridge gate. Standard library only; importing this module performs no fit."""
from dataclasses import dataclass
import hashlib
import json
import math

LAMBDA = 0.1
METHOD = {
    "name": "balanced_centered_unit_ridge_v1", "lambda": LAMBDA,
    "centering": "unweighted_training_mean", "normalization": "row_l2",
    "class_weight": "1/(2*n_class)", "intercept_penalty": 0.0,
    "threshold": 0.0, "on_rule": "score>0", "arithmetic": "binary64_fsum",
}
CHECKPOINT = "Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17"
CONTRACT = {"checkpoint": CHECKPOINT, "native_target": "model.language_model.layers.10",
            "position": "final_input", "residual_dtype": "float32", "width": 1024}


def require(ok, code):
    if not ok:
        raise ValueError(code)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def decode(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, "DUPLICATE_JSON_KEY")
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: require(False, "JSON_NONFINITE"))


def scalar(value):
    require(type(value) in (int, float), "NUMERIC_TYPE")
    result = float(value)
    require(math.isfinite(result), "NONFINITE")
    return result


def vector(values, width):
    require(type(values) in (tuple, list) and len(values) == width, "WIDTH")
    return tuple(scalar(value) for value in values)


def dot(a, b):
    return scalar(math.fsum(x * y for x, y in zip(a, b, strict=True)))


def centered_unit(row, mu):
    row = vector(row, len(mu))
    delta = tuple(scalar(x - m) for x, m in zip(row, mu, strict=True))
    length = math.sqrt(dot(delta, delta))
    require(length > 0, "ZERO_CENTERED_NORM")
    return tuple(scalar(x / length) for x in delta)


def training_data(rows, labels):
    require(type(rows) in (tuple, list) and len(rows) == 32, "ROW_COUNT")
    require(type(rows[0]) in (tuple, list) and 1 <= len(rows[0]) <= 1024, "WIDTH")
    width = len(rows[0])
    h = tuple(vector(row, width) for row in rows)
    require(type(labels) in (tuple, list) and len(labels) == len(h), "LABEL_COUNT")
    require(all(type(y) is int and y in (-1, 1) for y in labels), "LABELS")
    require(set(labels) == {-1, 1}, "BOTH_CLASSES")
    mu = tuple(math.fsum(row[j] for row in h) / len(h) for j in range(width))
    x = tuple(centered_unit(row, mu) for row in h)
    a = tuple(1.0 / (2 * labels.count(y)) for y in labels)
    return h, tuple(labels), mu, x, a


def cholesky_solve(matrix, target):
    n = len(target)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            value = scalar(matrix[i][j] - math.fsum(lower[i][k] * lower[j][k]
                                                  for k in range(j)))
            if i == j:
                require(value > 0, "NONPOSITIVE_CHOLESKY_PIVOT")
                lower[i][j] = math.sqrt(value)
            else:
                lower[i][j] = scalar(value / lower[j][j])
    intermediate = []
    for i in range(n):
        intermediate.append(scalar((target[i] - math.fsum(lower[i][j] * intermediate[j]
                                                         for j in range(i))) / lower[i][i]))
    result = [0.0] * n
    for i in range(n - 1, -1, -1):
        result[i] = scalar((intermediate[i] - math.fsum(lower[j][i] * result[j]
                                                      for j in range(i + 1, n))) / lower[i][i])
    return tuple(result)


@dataclass(frozen=True)
class Gate:
    mu: tuple
    w: tuple
    b: float

    def __post_init__(self):
        require(type(self.mu) is tuple and 1 <= len(self.mu) <= 1024, "IMMUTABLE_MU")
        require(type(self.w) is tuple, "IMMUTABLE_W")
        vector(self.mu, len(self.mu))
        vector(self.w, len(self.mu))
        scalar(self.b)

    def score(self, h0):
        return scalar(dot(self.w, centered_unit(h0, self.mu)) + self.b)

    def route(self, h0):
        return "ON" if self.score(h0) > 0.0 else "OFF"


def fit(rows, labels):
    """One fixed solve. Real-data callers must use the separately gated entry point."""
    _, y, mu, x, a = training_data(rows, labels)
    width, n = len(mu), len(x)
    xbar = tuple(math.fsum(a[i] * x[i][j] for i in range(n)) for j in range(width))
    ybar = math.fsum(a[i] * y[i] for i in range(n))
    z = tuple(tuple(math.sqrt(a[i]) * (x[i][j] - xbar[j]) for j in range(width))
              for i in range(n))
    t = tuple(math.sqrt(a[i]) * (y[i] - ybar) for i in range(n))
    gram = tuple(tuple(dot(z[i], z[j]) + (LAMBDA if i == j else 0.0)
                       for j in range(n)) for i in range(n))
    c = cholesky_solve(gram, t)
    w = tuple(math.fsum(z[i][j] * c[i] for i in range(n)) for j in range(width))
    return Gate(mu, w, scalar(ybar - dot(w, xbar)))


def hash_id(value):
    require(type(value) is str and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), "SHA256")
    return value


def artifact(model, *, training_manifest_sha256, construction_lock_sha256,
             feature_sha256, source_sha256):
    require(len(model.mu) == 1024, "NATIVE_WIDTH")
    value = {"schema": "native_supervised_gate_artifact_v1", "method": dict(METHOD),
             "feature_contract": dict(CONTRACT),
             "bindings": {"training_manifest_sha256": hash_id(training_manifest_sha256),
                          "construction_lock_sha256": hash_id(construction_lock_sha256),
                          "feature_sha256": hash_id(feature_sha256),
                          "source_sha256": hash_id(source_sha256)},
             "parameters": {"mu": list(model.mu), "w": list(model.w), "b": model.b}}
    return canonical(value)


def load_artifact(raw, *, expected_sha256, expected_bindings):
    require(digest(raw) == hash_id(expected_sha256), "ARTIFACT_HASH")
    value = decode(raw)
    require(type(value) is dict and set(value) == {"schema", "method", "feature_contract",
                                                 "bindings", "parameters"}, "ARTIFACT_SCHEMA")
    require(value["schema"] == "native_supervised_gate_artifact_v1", "ARTIFACT_SCHEMA")
    require(canonical(value) == raw, "CANONICAL_ARTIFACT")
    require(canonical(value["method"]) == canonical(METHOD), "METHOD_CHANGED")
    require(canonical(value["feature_contract"]) == canonical(CONTRACT), "CONTRACT_CHANGED")
    required = {"training_manifest_sha256", "construction_lock_sha256", "feature_sha256", "source_sha256"}
    require(type(expected_bindings) is dict and set(expected_bindings) == required, "EXPECTED_BINDINGS")
    require(value["bindings"] == {key: hash_id(expected_bindings[key]) for key in required}, "BINDINGS_CHANGED")
    params = value["parameters"]
    require(type(params) is dict and set(params) == {"mu", "w", "b"}, "PARAMETER_SCHEMA")
    return Gate(vector(params["mu"], 1024), vector(params["w"], 1024), scalar(params["b"]))
