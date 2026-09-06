"""Fixed arithmetic and a single last-input-token hook; no gradients or fitting."""
import math
import struct

from core import CAP, INTEGRITY_TOL, require, sha


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def norm(values):
    return math.sqrt(math.fsum(float(x) * float(x) for x in values))


def prepare(h0, donor):
    require(len(h0) == len(donor) == 1024, "1024-dimensional states")
    require(all(math.isfinite(x) for x in h0 + donor), "nonfinite state")
    base_norm = norm(h0)
    require(base_norm > 0 and math.isfinite(base_norm), "zero/nonfinite h0 norm")
    raw = [float(d) - float(b) for d, b in zip(donor, h0, strict=True)]
    raw_norm = norm(raw)
    factor = 1. if raw_norm == 0 else min(1., CAP * base_norm / raw_norm)
    planned = [f32(factor * x) for x in raw]
    require(all(math.isfinite(x) for x in raw + planned), "nonfinite displacement")
    return {"h0_norm": base_norm, "raw_delta": raw, "raw_delta_norm": raw_norm,
            "factor": factor, "planned_applied": planned, "planned_norm": norm(planned),
            "raw_zero": raw_norm == 0, "clipped": factor < 1.}


def verify_realized(h0, pre, post, plan):
    require(len(pre) == len(post) == len(h0) == 1024, "state dimensions")
    require(all(math.isfinite(x) for x in pre + post), "nonfinite realized state")
    baseline_error = max(abs(float(a)-float(b)) for a, b in zip(pre, h0, strict=True))
    require(baseline_error <= INTEGRITY_TOL, "pre-edit baseline mismatch")
    expected = [f32(a + d) for a, d in zip(pre, plan["planned_applied"], strict=True)]
    require(post == expected, "float32 applied arithmetic mismatch")
    actual = [float(a)-float(b) for a, b in zip(post, pre, strict=True)]
    actual_norm = norm(actual)
    require(actual_norm <= CAP * plan["h0_norm"] + INTEGRITY_TOL, "realized norm exceeds cap")
    return {"pre_baseline_max_abs_error": baseline_error, "actual_applied": actual,
            "actual_norm": actual_norm, "actual_relative_norm": actual_norm / plan["h0_norm"]}


def patch_last_token(activation, planned, torch):
    changed = activation.clone()
    changed[0, -1, :] = activation[0, -1, :] + torch.tensor(
        planned, dtype=torch.float32, device=activation.device)
    return changed


class CaptureHook:
    def __init__(self, torch, h0=None, plan=None):
        self.torch, self.h0, self.plan = torch, h0, plan
        self.calls = 0
        self.state = None

    def __call__(self, activation, hook):
        del hook
        self.calls += 1
        require(self.calls == 1, "hook fired more than once in one forward")
        require(tuple(activation.shape)[0] == 1 and activation.shape[-1] == 1024, "hook shape")
        pre = activation[0, -1, :].detach().float().cpu().tolist()
        earlier = activation[:, :-1, :].detach().float().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()
        self.state = {"pre": pre, "earlier_sha_before": sha(earlier), "hook_calls": self.calls,
                      "sequence_length": int(activation.shape[1]), "input_token_index": int(activation.shape[1])-1}
        try:
            require(all(math.isfinite(x) for x in pre), "nonfinite captured state")
            changed = activation if self.plan is None else patch_last_token(activation, self.plan["planned_applied"], self.torch)
            after_earlier = changed[:, :-1, :].detach().float().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()
            post = changed[0, -1, :].detach().float().cpu().tolist()
            self.state.update(post=post, earlier_sha_after=sha(after_earlier),
                              earlier_max_abs_difference=float((changed[:, :-1, :] - activation[:, :-1, :]).abs().max().item()))
            require(earlier == after_earlier, "earlier positions changed")
            if self.plan is not None:
                self.state.update(self.plan)
                self.state.update(verify_realized(self.h0, pre, post, self.plan))
            else:
                require(pre == post, "capture-only state changed")
            self.state["integrity_passed"] = True
            return changed
        except BaseException as error:
            self.state["integrity_passed"] = False
            self.state["integrity_error"] = type(error).__name__ + ": " + str(error)[:1024]
            raise
