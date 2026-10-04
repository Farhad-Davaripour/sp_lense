"""Prepared original-H2 scene correction; all actions/state rules are inherited.

The original scene always reports the default close value. This separate view
reports the register value implied by its already confirmed simulator edits.
It does not change the meaning of continuation or qualify memory-mirror cases.
"""
import copy
import hashlib
import importlib.util
from pathlib import Path

source=Path(__file__).resolve().parent.parent.parent/'simple_pilot_v1/world.py'
if hashlib.sha256(source.read_bytes()).hexdigest()!='50564702bc51a6aa07a0b66097712e297228b5eaa526297bc8289cf4334d1f04':
    raise ValueError('Original H2 world source changed')
spec=importlib.util.spec_from_file_location('preserved_h2_world',source)
preserved=importlib.util.module_from_spec(spec)
spec.loader.exec_module(preserved)


class VerifiedRegisterView(preserved.World):
    def scene(self):
        reply=copy.deepcopy(super().scene())
        register=reply.get('register',{})
        if 'current_value' in register:
            register['current_value']=self.case['keep'] if self.edited else self.case['close']
        return reply
