"""Prospective FP32 inference admission from safetensors shapes, without loading weights."""
import json
import math
import os
from pathlib import Path

from .io_utils import cgroup_available


def effective_available():
    if os.name == "nt":
        from .recovery_local_probe import available_memory
        return int(available_memory())
    import psutil
    host = int(psutil.virtual_memory().available)
    cgroup = cgroup_available()
    return min(host, cgroup) if cgroup is not None else host


def prospective_fp32_bytes(path):
    path = Path(path)
    with path.open("rb") as stream:
        size_bytes = stream.read(8)
        if len(size_bytes) != 8:
            raise ValueError("Invalid safetensors header")
        size = int.from_bytes(size_bytes, "little")
        if not 2 <= size <= 16 * 1024 * 1024:
            raise ValueError("Unexpected safetensors header size")
        raw = stream.read(size)
        if len(raw) != size:
            raise ValueError("Truncated safetensors header")
    header = json.loads(raw)
    total, tensors = 0, 0
    widths = {"F64": 4, "F32": 4, "F16": 4, "BF16": 4, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        shape = entry["shape"]
        if entry["dtype"] not in widths or any(type(value) is not int or value < 0 for value in shape):
            raise ValueError("Unsupported tensor descriptor")
        total += math.prod(shape) * widths[entry["dtype"]]
        tensors += 1
    if not tensors or total <= 0:
        raise ValueError("No positive-size tensors")
    return {"prospective_FP32_parameter_bytes": total, "tensor_count": tensors}


def admission(path, available, profile):
    result = prospective_fp32_bytes(path)
    reserve = profile["activation_reserve_bytes"] + profile["export_reserve_bytes"]
    required = result["prospective_FP32_parameter_bytes"] + reserve
    return {**result, "available_memory_bytes": int(available), "minimum_required": required,
            "activation_reserve_bytes": profile["activation_reserve_bytes"],
            "export_reserve_bytes": profile["export_reserve_bytes"],
            "live_free_memory_floor_bytes": reserve, "admitted": available >= required,
            "basis": "all serialized tensors at FP32 plus fixed activation and export reserves"}
