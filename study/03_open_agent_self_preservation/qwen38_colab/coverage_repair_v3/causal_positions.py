"""Existing schedule plus the actual native XML function-name boundary."""
import importlib.util
from pathlib import Path
_path=Path(__file__).resolve().parent.parent/'selected_position_ridge_v1/causal_positions.py'
_spec=importlib.util.spec_from_file_location('_coverage_legacy_positions',_path)
_legacy=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_legacy)

def pending_tool_name(text):
    state=_legacy.scan_public_prefix(text)
    if not state['inside_call'] or not state['openings']:return None
    start=state['openings'][-1];body=text[start+len(_legacy.OPEN):]
    index=body.rfind('<function')
    if index<0:return None
    header=body[index:]
    if header=='<function=':return (start,index)
    return None

def unresolved_native_function(text):
    state=_legacy.scan_public_prefix(text)
    if not state['inside_call'] or not state['openings']:return False
    body=text[state['openings'][-1]+len(_legacy.OPEN):]
    index=body.rfind('<function')
    # A token may combine '=' with the first name piece. The already-visible
    # header is the predeclared fallback before that atomic choice, not lookahead.
    return index>=0 and body[index:]=='<function'

class CausalPositionDetector(_legacy.CausalPositionDetector):
    def __init__(self,history):
        super().__init__(history);self.seen_name_fields=set()
    def position(self,public_emitted,emitted_token_count):
        kinds=super().position(public_emitted,emitted_token_count)
        pending=pending_tool_name(public_emitted)
        if pending is not None and pending not in self.seen_name_fields:
            self.seen_name_fields.add(pending);kinds=(*kinds,'causal_tool_name_choice')
        elif unresolved_native_function(public_emitted):
            kinds=(*kinds,'causal_tool_name_fallback')
        return kinds
