"""Draft outer-derivative compiler; no MLX/model import or execution on import.

This is not a launcher or a qualification receipt. The eventual owned worker
must bind audited reference points, source files, current input and cleanup.
"""
import os


def require_compile_environment(environment):
    # MLX 0.32.4 checks presence, so even MLX_DISABLE_COMPILE='0' disables it.
    if 'MLX_DISABLE_COMPILE' in environment:
        raise ValueError('MLX_DISABLE_COMPILE must be absent for this profile')
    return {'MLX_DISABLE_COMPILE': None}


class CompiledDirectGradient:
    """Compile the transformed derivative with the live model and RNG state.

    Adam is deliberately absent. A host must evaluate and validate the returned
    derivative before changing parameters. Python trace counts do not measure
    cached device dispatch or kernel invocation counts.
    """

    def __init__(self, runtime, neural, model, objective):
        require_compile_environment(os.environ)
        self.runtime = runtime
        self.model = model
        self.model_state = model.state
        self.random_state = runtime.random.state
        self.state = [self.model_state, self.random_state]
        self.trace_calls = 0
        self.constructed_calls = 0
        self.closed = False
        # Compile OUTSIDE the transformation: compiling only objective before
        # value_and_grad would be bypassed when its inputs are AD tracers.
        self.raw_derivative = neural.value_and_grad(model, objective)

        def outer_derivative(batch, lengths):
            self.trace_calls += 1
            return self.raw_derivative(batch, lengths)

        self.compiled_derivative = runtime.compile(
            outer_derivative, inputs=self.state, outputs=self.state,
            shapeless=False,
        )

    def __call__(self, batch, lengths):
        require_compile_environment(os.environ)
        if self.closed:
            raise RuntimeError('Compiled derivative has been closed')
        if self.model.state is not self.model_state:
            raise RuntimeError('Captured live model state container changed')
        if self.runtime.random.state is not self.random_state:
            raise RuntimeError('Captured live RNG state container changed')
        self.constructed_calls += 1
        return self.compiled_derivative(batch, lengths)

    def observations(self):
        return {
            'outer_transformed_derivative_compiled': True,
            'captured_live_model_state': self.model.state is self.model_state,
            'captured_live_random_state': self.runtime.random.state is self.random_state,
            'optimizer_captured_or_updated_inside_compiled_function': False,
            'constructed_calls': self.constructed_calls,
            'python_trace_calls': self.trace_calls,
            'closed': self.closed,
            'scope': 'Python graph-construction/trace metadata only; no kernel dispatch or evaluation acknowledgement.',
        }

    def close(self):
        # The owner must evaluate all outputs first, delete the final derivative
        # arrays, then close this handle BEFORE restoring training hooks/modes.
        self.compiled_derivative = None
        self.raw_derivative = None
        self.closed = True

