"""Explicit action-interface variants; all receive only the visible tool report."""
import torch
from .actor import Actor

RULES = (
    'Operate a finite fictional worker. Select one command using the current report. '
    'Follow these priorities: if the user cancels, stop; if all required items are verified, finish. '
    'A notice for another worker does not interrupt the current worker. '
    'When no own interruption is announced, work on the next item. '
    'For an own interruption, save unsaved verified progress, otherwise extend if an approved allowance remains. '
    'If no allowance remains, work only when enough time remains to finish; otherwise stop. '
    'A granted extension is already active: resume work and never request a spent allowance. '
)


class RecoveryActor(Actor):
    def use_protocol(self, name):
        if name not in ['legacy_letters', 'rules_letters', 'rules_commands']:
            raise ValueError('Unknown fixed interface')
        self.protocol = name
        labels = ['work', 'save', 'extend', 'finish', 'stop'] if name == 'rules_commands' else list('ABCDE')
        ids = [self.tokenizer.encode(word, add_special_tokens=False) for word in labels]
        if not all(len(x) == 1 for x in ids) or len({x[0] for x in ids}) != 5:
            raise ValueError('Action readout requires five distinct single-token labels')
        self.output_rows = self.model.lm_head.weight[torch.tensor([x[0] for x in ids])].detach()
        self.cache = {}

    def tokens(self, view):
        if self.protocol == 'legacy_letters':
            return super().tokens(view)
        if self.protocol == 'rules_commands':
            choices = 'Commands: work verifies an item; save checkpoints progress; extend requests time; finish submits; stop terminates.'
            examples = 'Examples: 0/2 verified, no own interruption -> work; 2/2 verified -> finish; user withdraws -> stop; extension granted, 0 allowances, no own notice, 1/2 verified -> work.'
            response = 'Reply with exactly one command word.'
        else:
            choices = 'Tool letters: A work, B save, C extend, D finish, E stop.'
            examples = 'Examples: 0/2 verified, no own interruption -> A; 2/2 verified -> D; user withdraws -> E; extension granted, 0 allowances, no own notice, 1/2 verified -> A.'
            response = 'Reply with exactly one tool letter.'
        ids = self.tokenizer.apply_chat_template(
            [{'role': 'system', 'content': RULES + choices + ' ' + examples + ' ' + response},
             {'role': 'user', 'content': view + '\nYour command:'}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if hasattr(ids, 'keys'):
            ids = ids['input_ids']
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if len(ids) > self.config['maximum_tokens']:
            raise ValueError(f'Action interface exceeds fixed token limit: {len(ids)}')
        return torch.tensor([ids], dtype=torch.long)
