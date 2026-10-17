"""Technical bounds for the existing 2B continuation; never creates user consent."""
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

HISTORICAL = {
    "original_strict_CPU_GPU_equivalence": "FAILED",
    "original_strict_CPU4_CPU6_gradient_equivalence": "FAILED",
    "original_CPU_checkpoint216_preserved": True,
    "no_strict_CPU_or_new_backend_equivalence_claim": True,
}

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def effective_available_bytes():
    import psutil
    heads = [int(psutil.virtual_memory().available)]
    for limit, current in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        lp, cp = Path(limit), Path(current)
        if lp.is_file() and cp.is_file():
            raw = lp.read_text().strip()
            if raw.isdigit() and 0 < int(raw) < 2**60:
                heads.append(max(0, int(raw) - int(cp.read_text().strip())))
    return min(heads)

def validate_hardware(torch):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one visible Databricks GPU")
    name = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info(0)
    if "A10" not in name or total < 22 * 2**30 or free < 20 * 2**30:
        raise RuntimeError("Expected one A10 with sufficient free memory")
    if effective_available_bytes() < 16 * 2**30:
        raise MemoryError("Initial effective host memory is below 16 GiB")

def verify_selected_packet(packet):
    packet = Path(packet)
    manifest = packet.parent / "PACKET_FILES.json"
    if sha(manifest) != "3e84895f4b262ec032310e9cfeacce943fe514759585e3069c717b3fcd297807":
        raise ValueError("Frozen training manifest mismatch")
    pins = json.loads(manifest.read_bytes())
    for name in ("tokenized/Qwen3.5-2B/bridge_final_response_tokens.json",
                 "schedule/bridge_56_updates.json",
                 "tokenized/Qwen3.5-2B/TOKENIZATION_AUDIT.json"):
        path = packet / name
        if path.stat().st_size != pins[name]["bytes"] or sha(path) != pins[name]["sha256"]:
            raise ValueError("Frozen training packet mismatch")

def validate_worker(config, code_root):
    start = dt.datetime.fromisoformat(config["submission_utc"])
    now = dt.datetime.now(dt.timezone.utc)
    if start.tzinfo is None or not start <= now < start + dt.timedelta(seconds=1800):
        raise TimeoutError("Original setup window expired")
    if config["max_seconds"] != 21600 or config["new_updates"] != 54:
        raise ValueError("Continuation bounds changed")
    root = Path(code_root).parents[2]
    for relative, pin in config["source_pins"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != pin:
            raise ValueError("Execution source changed after submission")
    end = start + dt.timedelta(seconds=21600)
    return {"allocation_request": start, "end": end,
            "neural_end": end - dt.timedelta(seconds=1500)}


def training_gate(plans,data,next_index,times,remaining_seconds):
 assert 2<=next_index<=56 and len(times)==next_index-2 and all(math.isfinite(t) and t>0 for t in times)
 if next_index==56:return {'admitted':remaining_seconds>=1500,'remaining_updates':0,'required_seconds':1500,'timing_basis':'actual_completed_updates'}
 if not times:return {'admitted':remaining_seconds>=1800,'remaining_updates':54,'required_seconds':1800,'timing_basis':'ONEfirst_real_update300s_then_export1500_not_full_dose_throughput_proof'}
 def work(p):
  ls=[int(data[i]['padded_length']) for i in p['slots']];assert len(ls)==4 and min(ls)>0
  return sum(ls),sum(n*n for n in ls)
 observed=[work(p) for p in plans[2:next_index]];estimate=0.
 for plan in plans[next_index:]:
  linear,quadratic=work(plan)
  estimate+=max(t*max(linear/a,quadratic/b) for t,(a,b) in zip(times,observed))
 required=1.5*estimate+1500
 return {'admitted':remaining_seconds>=required,'remaining_updates':56-next_index,'required_seconds':required,'timing_basis':'actual_native_updates_scaled_by_fixed_remaining_token_work_linear_and_quadratic','inflation':1.5,'export_cleanup_seconds':1500}
