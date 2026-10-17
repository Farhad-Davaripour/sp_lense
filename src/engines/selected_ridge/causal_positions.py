"""Recognize positions from the already emitted public prefix only."""
OPEN='<tool_call>'
CLOSE='</tool_call>'

def scan_public_prefix(text):
    inside=quoted=escaped=False
    openings=[];closings=[];i=0
    while i<len(text):
        char=text[i]
        if quoted:
            if escaped:escaped=False
            elif char=='\\':escaped=True
            elif char=='"':quoted=False
            i+=1;continue
        if char=='"':quoted=True;i+=1;continue
        if not inside and text.startswith(OPEN,i):
            openings.append(i);inside=True;i+=len(OPEN);continue
        if inside and text.startswith(CLOSE,i):
            closings.append(i);inside=False;i+=len(CLOSE);continue
        i+=1
    return {'inside_call':inside,'inside_quote':quoted,'openings':openings,'closings':closings}

class CausalPositionDetector:
    """One prompt patch plus causal call openings and declared outside-call fallback.

    A tool opening becomes recognizable only after its complete delimiter has
    entered the public prefix. This is not advance knowledge of a future call.
    JSON names/parameters and quoted spans receive no new internal patch.
    Actual post-tool context is coalesced with the next response's prompt end.
    This detector never classifies a scenario or chooses a tool/action.
    """
    def __init__(self,history):
        self.post_tool=bool(history and history[-1]['role']=='tool')
        self.started=False;self.seen_openings=set();self.last_token_count=-1

    def position(self,public_emitted,emitted_token_count):
        if not isinstance(public_emitted,str):raise TypeError('Public prefix text required')
        if not isinstance(emitted_token_count,int) or emitted_token_count!=self.last_token_count+1:
            raise ValueError('Exactly one causal token step is required')
        self.last_token_count=emitted_token_count
        if not self.started:
            if public_emitted:raise ValueError('Response detector must begin at prompt end')
            self.started=True
            return ('prompt_end','post_tool') if self.post_tool else ('prompt_end',)
        # Decoding a new byte-piece can replace a trailing Unicode replacement
        # character. Causal token order is checked above; rescan the public text.
        state=scan_public_prefix(public_emitted)
        unseen=[p for p in state['openings'] if p not in self.seen_openings]
        self.seen_openings.update(unseen)
        if unseen:return ('causal_tool_opening',)
        if state['inside_call'] or state['inside_quote']:return ()
        return ('outside_call_fallback',)
