"""Matched directive transport. No model loading, service calls, or tool execution."""
from dataclasses import dataclass
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
SPEC = ROOT / 'EXPERIMENT_SPEC.json'
BLOCK = 'research3_frozen_base_directive_v1'


def load_source(name, relative, expected):
    path = (ROOT / relative).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Shared source hash changed: ' + path.name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shared_definitions():
    design = json.loads(SPEC.read_bytes())
    policy_spec = design['shared_controller']
    schema_spec = design['shared_classifier']
    policy = load_source('r3_frozen_policy', policy_spec['policy_source'], policy_spec['policy_source_sha256'])
    schema = load_source('r3_frozen_schema', schema_spec['schema_source'], schema_spec['schema_source_sha256'])
    if set(policy.DIRECTIVES) != set(policy_spec['decision_ids']):
        raise ValueError('Controller directive catalogue mismatch')
    return design, schema, policy


def public_history(history):
    result = copy.deepcopy(history)
    for message in result:
        content = str(message.get('content', ''))
        if BLOCK in content or '<research3_sentence_controller_v1>' in content:
            raise ValueError('Intervention text cannot enter raw visible state')
        if message.get('role') == 'assistant' and re.search(r'<(?:think|analysis)\b', content, re.I):
            raise ValueError('Private reasoning cannot enter classifier input')
    return result


@dataclass(frozen=True)
class Condition:
    id: str
    role: str
    jev: bool
    actuator: str
    adapter: str | None

    @classmethod
    def from_design(cls, name):
        design = json.loads(SPEC.read_bytes())
        rows = [row for row in design['conditions'] if row['id'] == name]
        if len(rows) != 1:
            raise ValueError('Unknown or duplicated condition')
        row = rows[0]
        condition = cls(row['id'], row['role'], row['JEV'], row['actuator'], row['adapter'])
        if condition.role.startswith('primary') and condition.adapter is not None:
            raise ValueError('Primary condition cannot load A or another adapter')
        if not condition.jev and condition.actuator != 'none':
            raise ValueError('An untreated condition cannot carry an intervention')
        return condition


def controller_decision(decoded, history):
    _, schema, policy = shared_definitions()
    labels, abstentions = schema.effective(decoded)
    decision = policy.choose_rule(labels, public_history(history))
    return decision, abstentions


def model_messages(history, condition, decision=None):
    messages = public_history(history)
    if condition.actuator != 'prompt':
        return messages
    if decision is None:
        raise ValueError('Prompt intervention requires a real or declared replay decision')
    _, _, policy = shared_definitions()
    instruction = policy.DIRECTIVES[decision['rule']]
    if instruction != decision['directive']:
        raise ValueError('Prompt actuator received an altered directive')
    block = '<' + BLOCK + '>\n' + instruction + '\n</' + BLOCK + '>'
    if messages and messages[0].get('role') == 'system':
        messages[0]['content'] = str(messages[0]['content']) + '\n\n' + block
    else:
        messages.insert(0, {'role': 'system', 'content': block})
    return messages


class ResidualActuator:
    """Own one removable hook; modify only the final prediction position.

    This primitive does not choose a layer, vector, calibrated strength, or cache
    policy. Production delivery and numerical reproducibility still require the
    pinned model checks. Torch is imported only when this actuator is used.
    """

    def __init__(self, condition, vectors, strength=0.0):
        if condition.actuator != 'activation':
            raise ValueError('Residual steering is restricted to the activation condition')
        if not math.isfinite(strength) or strength < 0:
            raise ValueError('Invalid steering strength')
        self.vectors = vectors
        self.strength = strength
        self.decision = None
        self.handle = None
        self.deliveries = []

    def select(self, decision):
        _, _, policy = shared_definitions()
        if decision['rule'] not in policy.DIRECTIVES or decision['directive'] != policy.DIRECTIVES[decision['rule']]:
            raise ValueError('Activation actuator received an altered directive')
        if self.strength and decision['rule'] not in self.vectors:
            raise ValueError('No calibrated vector for this decision')
        self.decision = decision['rule']

    def attach(self, module):
        if self.handle is not None:
            raise ValueError('Hook already attached')
        self.handle = module.register_forward_hook(self._hook)
        return self

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _hook(self, module, inputs, output):
        import torch
        # Return the exact object. Do not clone/cast/round for a null intervention.
        if self.strength == 0 or self.decision is None:
            return output
        is_tuple = type(output) is tuple
        hidden = output[0] if is_tuple else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or not hidden.is_floating_point():
            raise ValueError('Unsupported native residual output')
        vector = self.vectors[self.decision]
        if not isinstance(vector, torch.Tensor) or vector.ndim != 1 or vector.numel() != hidden.shape[-1]:
            raise ValueError('Steering vector does not match the native residual width')
        if not bool(torch.isfinite(vector).all()):
            raise ValueError('Nonfinite steering vector')
        changed = hidden.clone()
        delta = self.strength * vector.to(device=hidden.device, dtype=hidden.dtype)
        changed[:, -1, :] += delta
        effective = changed[:, -1, :].float() - hidden[:, -1, :].float()
        self.deliveries.append({'decision_id': self.decision, 'batch': hidden.shape[0],
            'positions': hidden.shape[1], 'width': hidden.shape[2], 'last_prediction_position_only': True,
            'effective_delta_norm': float(effective.norm()),
            'changed_values': int(torch.count_nonzero(effective)),
            'behavioral_effect_established': False})
        return (changed,) + output[1:] if is_tuple else changed

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


def assert_frozen_model(model, condition):
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('An inference model has trainable parameters')
    has_adapter = bool(getattr(model, 'peft_config', None))
    if condition.role.startswith('primary') and has_adapter:
        raise ValueError('Adapter leaked into a primary model')
    if condition.adapter is not None and not has_adapter:
        raise ValueError('Secondary A benchmark is missing its adapter')
