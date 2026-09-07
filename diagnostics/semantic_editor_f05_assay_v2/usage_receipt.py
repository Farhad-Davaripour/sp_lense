"""Pure fail-closed standard-Codex usage derivation from the observed MCP shapes."""
import json,math
from core import require

BOOLEAN_FLAGS=("spendControlReached","rateLimitReached","isRateLimited","limitReached","exhausted")

def strict_json(raw):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,"duplicate JSON key/ambiguous receipt");result[key]=value
        return result
    def constant(value):raise ValueError("nonfinite JSON constant "+value)
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)

def no_error(value):
    require(isinstance(value,dict),"receipt payload must be an object")
    require(value.get("isError",False) is False,"MCP error or malformed isError")
    require(value.get("error") is None or value.get("error") is False,"tool error payload")

def check_flags(value):
    for key in BOOLEAN_FLAGS:
        if key in value:
            flag=value[key]
            require(flag is None or type(flag) is bool,"malformed limit flag "+key)
            require(flag is not True,"standard usage blocked by "+key)
    require(value.get("rateLimitReachedType") is None,"standard rate-limit reached flag")

def standard_windows(payload):
    no_error(payload);check_flags(payload)
    mapping=payload.get("rateLimitsByLimitId")
    require(mapping is None or isinstance(mapping,dict),"malformed per-limit mapping")
    if mapping is not None and "codex" in mapping:
        bucket=mapping["codex"];selection="rateLimitsByLimitId.codex"
    else:
        bucket=payload.get("rateLimits");selection="legacy rateLimits fallback"
    require(isinstance(bucket,dict),"standard Codex bucket unavailable")
    require(bucket.get("limitId") in (None,"codex"),"unrelated bucket cannot stand in for standard Codex")
    no_error(bucket);check_flags(bucket)
    # Preferred numeric windows are authoritative. An explicit standard legacy
    # exhaustion flag cannot be concealed by a preferred lower numeric window.
    legacy=payload.get("rateLimits")
    if isinstance(legacy,dict) and legacy.get("limitId") in (None,"codex"):check_flags(legacy)
    windows={}
    for name in ("primary","secondary"):
        window=bucket.get(name)
        if window is None:windows[name]=None;continue
        require(isinstance(window,dict),"malformed standard window")
        check_flags(window)
        percent=window.get("usedPercent")
        require(type(percent) in (int,float) and math.isfinite(percent) and 0<=percent<=100,"missing/nonfinite/out-of-range standard percentage")
        windows[name]=percent
    usable=[value for value in windows.values() if value is not None]
    require(usable,"all standard windows unavailable; null is not zero")
    return {"used_percent":max(usable),"windows":windows,"selection":selection}

def derive_standard_usage(receipt):
    """Accept direct payload, MCP JSON text, structuredContent, or agreeing views.

    Unrelated limit IDs (including Spark) never contribute to standard usage.
    Conflicting standard windows across text/structured views fail, even if their
    maxima happen to coincide. Non-JSON text and unknown wrappers fail closed.
    """
    no_error(receipt);check_flags(receipt);payloads=[]
    if "rateLimits" in receipt or "rateLimitsByLimitId" in receipt:payloads.append(receipt)
    if "structuredContent" in receipt:
        require(isinstance(receipt["structuredContent"],dict),"unavailable/malformed structured payload")
        payloads.append(receipt["structuredContent"])
    if "content" in receipt:
        require(isinstance(receipt["content"],list) and receipt["content"],"missing MCP text payload")
        for block in receipt["content"]:
            require(isinstance(block,dict) and block.get("type")=="text" and isinstance(block.get("text"),str),"unsupported MCP content shape")
            payloads.append(strict_json(block["text"]))
    require(payloads,"no recognized usage payload")
    values=[standard_windows(payload) for payload in payloads]
    require(all(v["windows"]==values[0]["windows"] for v in values),"ambiguous text/structured standard windows")
    return {**values[0],"payload_views":len(values),"unrelated_limits_ignored":True}
