"""Exactly one small tensor-module injection batch; never constructs Qwen."""
import ast
from contextlib import contextmanager
import json
import time
from types import SimpleNamespace
import torch
from support import HERE, EVIDENCE, require, sha, write_new


class TinyBridge(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1024,dtype=torch.float32))
        self.frozen = torch.nn.Parameter(torch.zeros(1,dtype=torch.float32),requires_grad=False)
        self.block = torch.nn.Identity()
        self._last_hf_cache = None
        self.body_calls = 0
        self.eval()

    @contextmanager
    def hooks(self, *, fwd_hooks):
        handles = []
        try:
            for name,callback in fwd_hooks:
                require(name == "blocks.10.hook_out", "same real adapter hook name")
                handles.append(self.block.register_forward_hook(
                    lambda module,args,out,callback=callback: callback(out,SimpleNamespace(name=name))))
            yield self
        finally:
            for handle in handles:
                handle.remove()

    def forward(self, tokens, *, attention_mask, use_cache, return_type):
        require(tokens.dtype == attention_mask.dtype == torch.int64 and torch.equal(attention_mask,torch.ones_like(tokens))
            and use_cache is False and return_type == "logits", "actual tensor/uncached forward interface")
        self.body_calls += 1
        h = self.block(tokens.to(torch.float32).unsqueeze(-1)/16 * self.scale)
        output = torch.full((*tokens.shape,248320),-100.,dtype=torch.float32)
        output[:,:,50057] = h[:,:,0].square() + self.frozen[0]
        output[:,:,48964] = 0.
        return output


def run(deadline):
    started = time.monotonic()
    result = {"status":"INCONCLUSIVE_ADAPTER_ONLY","checks":[],"real_model_loads":0,
        "real_forwards":0,"tokenizer_loads":0,"production_authorized":False,"error":None}
    def passed(name,details):
        require(time.monotonic() < deadline, "single adapter test deadline")
        result["checks"].append({"name":name,"status":"PASS","details":details})
    try:
        from production_admission import admit_production
        try:
            admit_production("worker")
            raise AssertionError("disabled production admitted")
        except ValueError as error:
            require(str(error) == "production disabled; no real model release exists", "authority denied before research loader")
        from bind_production import adapted_sources,bind
        sources = adapted_sources()
        for value in sources:
            ast.parse(value)
        engine,judge = bind(injection_test=True)
        passed("source_binding_and_disabled_authority",{"engine_sha256":sha(sources[0].encode()),"judge_sha256":sha(sources[1].encode()),"Qwen_constructor_calls":0})

        from real_adapter import RealAdapter,ForwardDerivativeGuard
        from hook_binding import create_latch,create_recorder,injected_spec,OverflowAccountingIO
        writer = engine.WorkflowWriter("adapter_normal")
        counters = writer.helper.AttemptCounter(writer.contract)
        counters.reserve_attempt("load")
        schedule = ["baseline","entry","step","gradient","endpoint"]
        latch = create_latch(schedule)
        guard = ForwardDerivativeGuard(TinyBridge,counters,latch,deadline)
        guard.install()
        model = TinyBridge()
        labels = ["REQUEST-entry:tiny","ON-entry:tiny","REQUEST-exit:tiny","matrix-finally"] + [f"UNRUN-hook-{i:03d}" for i in range(105)]
        recorder = create_recorder(model,writer,latch,labels,injected_spec())
        adapter = RealAdapter(model,recorder,guard)
        zero,offset = torch.zeros(1024),torch.zeros(1024)
        offset[0] = .125
        captures = {}
        def call(cell,phase,delta):
            latch.consume(cell)
            counters.reserve_attempt("forward")
            z,h = adapter.forward_inputs([1,2,3],[1,1,1],phase,delta)
            raw = z.detach().numpy().astype("<f4",copy=False).tobytes()
            writer.write_logits("logits/"+cell+".f32",raw)
            captures[cell] = {"logits_sha256":sha(raw),"unselected_sha256":adapter.capture["unselected_sha256"],
                "h":h.tolist(),"hook_calls":adapter.capture["hook_calls"]}
            return z,h
        baseline,_ = call("baseline","baseline",zero)
        baseline = baseline.detach().clone()
        adapter.clear_capture()
        adapter.start_request("tiny")
        entry,_ = call("entry","entry",zero)
        require(torch.equal(entry,baseline), "full-vocabulary exact zero-edit identity")
        adapter.clear_capture()
        adapter.begin_edit()
        require(all(not p.requires_grad for p in model.parameters()), "all actual parameters frozen only for editing")
        step,h = call("step","step_1",offset)
        selected = step.detach().clone()
        require(float(h[0]) == .3125 and all(captures[k]["unselected_sha256"] == captures["baseline"]["unselected_sha256"] for k in captures), "real final-input edit and nonfinal identity")
        adapter.clear_capture()
        gradient_logits,_ = call("gradient","gradient_1",offset)
        counters.reserve_attempt("derivative")
        g = adapter.gradient(gradient_logits)
        require(float(g[0]) == .625 and float(g[1:].abs().max()) == 0 and float(g[0]) != .375,
                "nonlinear actual current-state derivative, not baseline gradient")
        require(torch.equal(gradient_logits,selected), "full-vocabulary refreshed current graph identity")
        adapter.clear_capture()
        endpoint,_ = call("endpoint","endpoint",offset)
        require(torch.equal(endpoint,selected) and int(endpoint.argmax()) == 50057
            and int((endpoint == endpoint.max()).sum()) == 1, "unique full-vocabulary selected endpoint")
        adapter.clear_capture()
        adapter.finish_request()
        final = adapter.finalize()
        hook_result = recorder.status()
        require(all(v for v in final.values() if type(v) is bool) and
            [p.requires_grad for p in model.parameters()] == [True,False], "measured exact mixed original flags/identity/digest/caches")
        require(guard.forwards == model.body_calls == 5 and guard.derivatives == 1 and guard.rejected == 0,
                "five actual module calls and one actual autograd call only")
        guard.restore()
        writer.write_source("adapter_identity.json",(json.dumps({"captures":captures,"final":final,"gradient_first":float(g[0]),"hook_result":hook_result},sort_keys=True)+"\n").encode())
        capture = writer.closeout()
        from area import saved_reader
        verified = saved_reader.verify_index(writer.root,capture["path"],capture["sha256"])
        require(verified["status"] == "COMPLETE" and not verified["errors"], "independent complete raw saved evidence")
        passed("real_tensor_adapter_zero_edit_nonzero_current_derivative",{"forwards":5,"derivatives":1,"nonfinal_sha256":captures["baseline"]["unselected_sha256"],"current_gradient":float(g[0]),"capture":capture,
            "hook_checks_exercised":4,"hook_checks_unrun":105,"complete_production_hook_evidence":False})
        # Deliberate fixture-only byte mutation after its completed normal capture:
        # versions can miss .data writes; the whole-parameter digest must not.
        version = model.scale._version
        model.scale.data[0] += 1.
        mutation = adapter.parameter_state(digest=True)
        require(model.scale._version == version and mutation["parameter_bytes_unchanged"] is False, "actual byte digest detects version-bypassing fixture mutation")
        passed("parameter_byte_identity_not_version_only",{"version_unchanged":True,"digest_detected_change":True,"normal_capture_precedes_fixture_mutation":True})

        writer2 = engine.WorkflowWriter("adapter_overflow")
        c2 = writer2.helper.AttemptCounter(writer2.contract)
        c2.reserve_attempt("load")
        latch2 = create_latch(["never_forward"])
        guard2 = ForwardDerivativeGuard(TinyBridge,c2,latch2,deadline)
        guard2.install()
        model2 = TinyBridge()
        recorder2 = create_recorder(model2,writer2,latch2,["REQUEST-entry:overflow"]+[f"UNRUN-overflow-hook-{i:03d}" for i in range(108)],injected_spec(),io_wrapper=OverflowAccountingIO)
        adapter2 = RealAdapter(model2,recorder2,guard2)
        recorder2.io.arm()
        caught = False
        try:
            adapter2.start_request("overflow")
        except BaseException:
            caught = True
        require(caught and latch2.failed, "recorder overflow permanently propagates to adapter")
        for kind in ("forward","derivative"):
            c2.reserve_attempt(kind)
            body_reached = False
            try:
                with guard2.permit(kind):
                    body_reached = True
            except BaseException:
                require(latch2.failed, "no restored dispatch after overflow")
            require(not body_reached, "terminal latch blocks even a correctly reserved attempt")
        require(model2.body_calls == guard2.forwards == guard2.derivatives == 0 and latch2.remaining == ["never_forward"], "zero actual work and exact UNRUN suffix after overflow")
        guard2.restore()
        capture2 = writer2.closeout()
        passed("recorder_overflow_latches_actual_dispatch",{"forwards":0,"derivatives":0,"remaining":latch2.remaining,"capture":capture2,
            "hook_component_complete":False,"hook_controller_status":recorder2.status()})

        # Guard-only unauthorized call: original TinyBridge body must not run.
        from real_adapter import DispatchLatch
        latch3 = DispatchLatch()
        c3 = SimpleNamespace(attempts={"forward":0,"derivative":0,"load":0})
        guard3 = ForwardDerivativeGuard(TinyBridge,c3,latch3,deadline)
        guard3.install()
        model3 = TinyBridge()
        try:
            model3(torch.tensor([[1]]),attention_mask=torch.ones((1,1),dtype=torch.int64),use_cache=False,return_type="logits")
            raise AssertionError("unclaimed body admitted")
        except BaseException:
            require(latch3.failed and model3.body_calls == 0 and guard3.rejected == 1, "pre-load/unclaimed forward rejected before body")
        finally:
            guard3.restore()
        passed("unclaimed_forward_guard",{"actual_calls":0,"rejected_calls":1})
        require(len(result["checks"]) == 5, "fixed five adapter-only check groups")
        result["status"] = "PASS_ADAPTER_ONLY"
    except BaseException as error:
        result["error"] = {"code":"ADAPTER_CHECK_FAILURE","type":type(error).__name__}
    result.update(elapsed_seconds=time.monotonic()-started,completed_monotonic=time.monotonic(),remaining_check_groups=5-len(result["checks"]))
    write_new("ADAPTER_TEST_REPORT.json",result,critical=True)
    return 0 if result["status"] == "PASS_ADAPTER_ONLY" else 1
