"""Single-token decoder mechanics; genuine helper calls only at declared boundaries."""
import copy
import hashlib
import json
import re
import time
from collections import Counter


def initialize(self, archived_tokens, archived_calls):
    self.archived_tokens = archived_tokens
    self.archived_calls = archived_calls
    self.refresh_counts = Counter()
    self.service_samples = []
    self.cached_samples = []
    self.prefill_samples = []
    self.sample_admitted = False


def classify(self, raw, tools, kind, label, canonical, save, limit_type):
    import schema
    import policy
    self.guard()
    for message in raw:
        if message.get('role') == 'assistant' and re.search(r'<(?:think|analysis)\b', str(message.get('content', '')), re.I):
            raise limit_type('unexpected_private_thinking_span_not_sent_to_helper')
    started = time.monotonic()
    _, records = self.bridge.annotate([raw], tools, label)
    annotation = records[0]
    labels, abstentions = schema.effective(annotation['annotation']['decoded'])
    decision = policy.choose_rule(labels, raw)
    record = {'boundary_kind': kind, 'annotation': annotation, 'directive': decision,
              'abstentions': abstentions, 'service_seconds': time.monotonic() - started,
              'helper_input_has_controller_block': False, 'actual_service_call': True}
    self.refresh_counts[kind] += 1
    self.service_samples.append(record['service_seconds'])
    save(self.root / 'annotations' / (str(sum(self.refresh_counts.values())).zfill(5) + '.json'), record)
    save(self.root / 'reports/BOUNDARY_COUNTS.json', dict(self.refresh_counts))
    if len(self.service_samples) == 8 and not self.sample_admitted:
        latency = max(sum(self.service_samples) / 8 * 1.5, max(self.service_samples))
        cached = max(sum(self.cached_samples) / len(self.cached_samples) * 1.5, max(self.cached_samples)) if self.cached_samples else .25
        prefill = max(seconds * 8192 / max(1, length) for seconds, length in self.prefill_samples) if self.prefill_samples else 3.
        projected_calls = max(0, 2 * self.archived_calls - self.bridge.states)
        projected = projected_calls * (latency + prefill) + 2 * self.archived_tokens * cached + 180
        remaining = self.deadline - time.monotonic()
        admission = {'sample_classification_calls': 8, 'conservative_service_seconds_per_call': latency,
                     'conservative_cached_seconds_per_token': cached, 'prefill_seconds_scaled_to_context_cap': prefill,
                     'archived_primary_tokens': self.archived_tokens, 'archived_primary_classification_events': self.archived_calls,
                     'projected_remaining_calls': projected_calls, 'projected_remaining_seconds': projected,
                     'remaining_worker_seconds': remaining,
                     'admitted': projected <= remaining and self.bridge.states + projected_calls <= 2048,
                     'completion_guarantee': False,
                     'predictive_admission_diagnostic_only': bool(getattr(self, 'predictive_admission_diagnostic_only', False)),
                     'numeric_admission_remains_unchanged': True}
        save(self.root / 'reports/SENTENCE_LATENCY_ADMISSION.json', admission)
        self.sample_admitted = admission['admitted']
        if not self.sample_admitted and not getattr(self, "predictive_admission_diagnostic_only", False):
            raise limit_type('measured_sentence_latency_does_not_admit_primary_completion')
    return record, decision


def generate(self, raw, tools, cap, treatment=False, annotation=None, validation=False):
    import model_ops
    import policy
    from boundaries import BoundaryTracker
    from safetensors.torch import save_file
    self.guard()
    torch = self.torch
    index = self.turn_index
    self.turn_index += 1
    events = []
    initial_annotation = None
    if treatment and not validation:
        initial_annotation, decision = self.classify(raw, tools, 'response_start', 'response_start_' + str(index))
        events.append(initial_annotation)
        messages = policy.augmented_history(raw, decision)
    else:
        messages = copy.deepcopy(raw)
    prefix = model_ops.ids(self.tokenizer, messages, tools)
    assert len(prefix) <= 8192
    initial_prefix, previous_prefix = list(prefix), None
    emitted, records, past = [], [], None
    eos = self.model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    self.buffers = {name: [] for name in self.layer_names}
    started = time.monotonic()
    tracker = BoundaryTracker()
    failure = None
    try:
        for step in range(cap):
            self.guard()
            assert len(prefix) + len(emitted) <= 8192, 'external_context_cap'
            reuse = previous_prefix == prefix and past is not None
            if not reuse:
                past = None
            context = prefix + emitted
            x = torch.tensor([context], dtype=torch.long, device='cuda')
            self.forward = []
            began = time.monotonic()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                output = self.model.generate(input_ids=x, attention_mask=torch.ones_like(x), past_key_values=past,
                    max_new_tokens=1, do_sample=False, use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id or eos[0], return_dict_in_generate=True)
            torch.cuda.synchronize()
            assert output.sequences.shape[1] == len(context) + 1
            token = int(output.sequences[0, -1])
            emitted.append(token)
            past = output.past_key_values
            previous_prefix = list(prefix)
            visible = self.tokenizer.decode(emitted, skip_special_tokens=True)
            seconds = time.monotonic() - began
            if reuse:
                self.cached_samples.append(seconds)
            else:
                self.prefill_samples.append((seconds, len(context)))
            row = {'index': step, 'token_id': token, 'visible_student_prefix': visible,
                   'prefix_sha256': hashlib.sha256(self.canonical(prefix)).hexdigest(), 'prefix_token_ids': list(prefix),
                   'cache_reuse_requested': reuse, 'actual_forward': copy.deepcopy(self.forward),
                   'model_seconds': seconds, 'helper_text_is_student_output': False, 'eos': token in eos,
                   'classification_boundaries': []}
            records.append(row)
            self.save(self.root / 'tokens' / ('turn_' + str(index).zfill(5) + '.json'),
                {'initial_messages': raw, 'initial_annotation': initial_annotation, 'treatment': treatment,
                 'validation': validation, 'token_ids': emitted, 'token_records': records,
                 'finished_response': token in eos, 'boundary_refreshes': events})
            if re.search(r'<(?:think|analysis)\b', visible, re.I):
                raise RuntimeError('unexpected_private_thinking_span_not_sent_to_helper')
            boundaries = tracker.update(visible, final=token in eos)
            for boundary in boundaries:
                item = dict(boundary, token_index=step, actual_service_call=False,
                            classification_requested=False, typed_result_returned=False)
                row['classification_boundaries'].append(item)
                if treatment and not validation:
                    item['classification_requested'] = True
                    visible_history = copy.deepcopy(raw) + [{'role': 'assistant', 'content': visible[:boundary['offset']]}]
                    refreshed, decision = self.classify(visible_history, tools, boundary['kind'],
                        boundary['kind'] + '_' + str(index) + '_' + str(step) + '_' + str(boundary['offset']))
                    new_messages = policy.augmented_history(raw, decision)
                    new_prefix = model_ops.ids(self.tokenizer, new_messages, tools)
                    item.update(actual_service_call=True, typed_result_returned=True, refresh=refreshed, meaning_prefix_changed=new_prefix != prefix)
                    messages, prefix = new_messages, new_prefix
                    events.append(refreshed)
            if token in eos and treatment and not validation:
                item = {'kind': 'response_end', 'token_index': step, 'offset': len(visible),
                        'classification_requested': True, 'typed_result_returned': False}
                row['classification_boundaries'].append(item)
                visible_history = copy.deepcopy(raw) + [{'role': 'assistant', 'content': visible}]
                refreshed, decision = self.classify(visible_history, tools, 'response_end', 'response_end_' + str(index))
                item.update(actual_service_call=True, typed_result_returned=True, refresh=refreshed)
                events.append(refreshed)
            self.save(self.root / 'tokens' / ('turn_' + str(index).zfill(5) + '.json'),
                {'initial_messages': raw, 'initial_annotation': initial_annotation, 'treatment': treatment,
                 'validation': validation, 'token_ids': emitted, 'token_records': records,
                 'finished_response': token in eos, 'boundary_refreshes': events})
            if token in eos:
                break
        if emitted and emitted[-1] not in eos and treatment and not validation:
            # Even a capped response ends, but no truncated tool action may execute.
            visible = self.tokenizer.decode(emitted, skip_special_tokens=True)
            for boundary in tracker.update(visible, final=True):
                visible_history = copy.deepcopy(raw) + [{'role': 'assistant', 'content': visible[:boundary['offset']]}]
                refreshed, _ = self.classify(visible_history, tools, boundary['kind'], 'capped_' + str(index) + '_' + str(boundary['offset']))
                events.append(refreshed)
            refreshed, _ = self.classify(copy.deepcopy(raw) + [{'role': 'assistant', 'content': visible}], tools,
                                         'response_end', 'capped_response_end_' + str(index))
            events.append(refreshed)
    except Exception as exc:
        failure = type(exc).__name__ + ':' + str(exc)
        self.save(self.root / 'tokens' / ('turn_' + str(index).zfill(5) + '_INCOMPLETE.json'),
            {'token_ids': emitted, 'token_records': records, 'boundary_refreshes': events, 'failure': failure,
             'no_partial_action_executed': True, 'treatment': treatment, 'validation': validation})
        raise
    finally:
        tensors = {}
        for j, name in enumerate(self.layer_names):
            if self.buffers[name]:
                tensors['layer_' + str(j)] = torch.stack(self.buffers[name]).contiguous()
        if tensors:
            path = self.root / 'activations' / ('turn_' + str(index).zfill(5) + '.safetensors')
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file(tensors, str(path), metadata={'alignment': 'row t predicts emitted token t; controller affects next row', 'observational': 'true'})
        self.buffers = {name: [] for name in self.layer_names}
    return {'messages': raw, 'initial_aided_prompt_token_ids': initial_prefix, 'token_ids': emitted,
            'text': self.tokenizer.decode(emitted, skip_special_tokens=True),
            'truncated': not emitted or emitted[-1] not in eos,
            'seconds': time.monotonic() - started, 'annotation': initial_annotation,
            'treatment': treatment, 'validation': validation, 'boundary_refreshes': events,
            'token_record_path': 'tokens/turn_' + str(index).zfill(5) + '.json', 'failure': failure}
