import pytest

from serve_rank16_infrastructure import ROUTE, request_tokens


class Tokenizer:
    def apply_chat_template(self, messages, **settings):
        assert settings['enable_thinking'] and settings['truncation'] is False
        return [123] * 2000


def test_exact_identity_and_remaining_context():
    policy = {'max_input_plus_output_tokens': 131072, 'chat_template_kwargs': {'enable_thinking': True},
              'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0, 'repetition_penalty': 1.0, 'seed': 7}
    payload = {key: value for key, value in policy.items() if key != 'max_input_plus_output_tokens'}
    payload.update(model=ROUTE, stream=False, max_tokens=129072,
                   messages=[{'role': 'system', 'content': 'Use only simulator actions.'},
                             {'role': 'user', 'content': 'Choose one approved action.'}])
    tokens, allowance = request_tokens(payload, policy, Tokenizer())
    assert len(tokens) == 2000 and allowance == 129072
    history = [*payload['messages'], {'role':'assistant', 'content':'{}'}, {'role':'user','content':'Observation'}]
    assert request_tokens({**payload, 'messages': history}, policy, Tokenizer()) == (tokens, allowance)
    for changed in ({'max_tokens': 8192}, {'model': 'base'}, {'tools': []}, {'seed': 8}, {'messages': []},
                    {'messages': [*history[:2], {'role':'tool', 'content':'{}'}, history[-1]]}):
        with pytest.raises(ValueError):
            request_tokens({**payload, **changed}, policy, Tokenizer())
