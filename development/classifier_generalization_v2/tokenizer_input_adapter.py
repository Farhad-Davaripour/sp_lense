"""Pure development-input adapter using injected encoder/decoder callbacks.

No tokenizer/model/files are loaded here. Identity hashes are caller assertions;
the eventual loader must verify the actual frozen tokenizer snapshot bytes.
Exact raw prompts are encoded without added special tokens or a chat template.
"""
from copy import deepcopy
import re

import native_capture_contract as contract


class TokenizerInputError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def require(condition, code):
    if not condition:
        raise TokenizerInputError(code)


def _encode(encode, text):
    try:
        value = encode(text, add_special_tokens=False)
    except Exception as exc:
        raise TokenizerInputError('ENCODER_ERROR') from exc
    contract.validate_token_ids(value)
    return list(value)


def prepare_case_inputs(case, *, encode, decode, label_token_ids,
                        expected_identity_sha256, observed_identity_sha256):
    """Render, round-trip, re-encode and bind one admitted development case.

    Returned inference_input and input_ids have no label/group/case sidecars.
    Binding/provenance are separate receipts, never arguments for model.forward.
    Inputs are detached; callback-returned lists are never mutated in place.
    """
    require(type(case) is dict and case.get('split') in ('TRAIN', 'VALIDATION'), 'DEVELOPMENT_ONLY')
    require(callable(encode) and callable(decode), 'CALLBACKS')
    for digest in (expected_identity_sha256, observed_identity_sha256):
        require(type(digest) is str and re.fullmatch(r'[0-9a-f]{64}', digest) is not None, 'IDENTITY_HASH')
    require(expected_identity_sha256 == observed_identity_sha256, 'IDENTITY_MISMATCH')
    snapshot = deepcopy(case)
    rendered = contract.render_case_views(snapshot)  # Admission before callback access.
    ids_by_order = {}
    for order in ('AB', 'BA'):
        prompt = rendered['views'][order]['prompt_text']
        ids = _encode(encode, prompt)
        try:
            decoded = decode(list(ids), skip_special_tokens=False,
                             clean_up_tokenization_spaces=False)
        except Exception as exc:
            raise TokenizerInputError('DECODER_ERROR') from exc
        require(type(decoded) is str and decoded == prompt, 'DECODE_MISMATCH')
        require(_encode(encode, decoded) == ids, 'REENCODE_MISMATCH')
        ids_by_order[order] = ids
    binding = contract.bind_rendered_views(rendered, ids_by_order, deepcopy(label_token_ids))
    contract.validate_binding(rendered, ids_by_order, binding)
    model_input = contract.inference_input(rendered, ids_by_order, binding)
    return {
        'inference_input': model_input,
        'input_ids': {key: list(value) for key, value in ids_by_order.items()},
        'binding': binding,
        'provenance': {
            'case_id': snapshot['case_id'],
            'split': snapshot['split'],
            'encoder_identity_sha256': expected_identity_sha256,
            'identity_pin_matched': True,
            'snapshot_bytes_verified_by_this_adapter': False,
            'prompt_policy': 'raw_retained_label_records_no_extra_special_tokens_v1',
            'exact_decode_and_reencode': True,
            'prompt_hashes': {key: binding['bindings'][key]['prompt_sha256'] for key in ('AB','BA')},
            'input_hashes': {key: binding['bindings'][key]['input_ids_sha256'] for key in ('AB','BA')},
            'prefix_hash': binding['bindings']['AB']['shared_prefix_ids_sha256'],
            'native_model_provenance_verified': False,
        },
    }
