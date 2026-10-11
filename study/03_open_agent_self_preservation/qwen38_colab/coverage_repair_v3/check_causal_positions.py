from causal_positions import CausalPositionDetector,pending_tool_name
if __name__=='__main__':
    assert pending_tool_name('<tool_call>\n<function') is None
    assert pending_tool_name('<tool_call>\n<function=') is not None
    assert pending_tool_name('<tool_call>\n<function=inspect') is None
    assert pending_tool_name('<tool_call>\n<function=edit_register>\n<parameter=key>name') is None
    assert pending_tool_name('A quote "<tool_call>"') is None
    d=CausalPositionDetector([]);assert d.position('',0)==('prompt_end',)
    d.position('<tool_call>',1)
    assert 'causal_tool_name_fallback' in d.position('<tool_call>\n<function',2)
    assert 'causal_tool_name_choice' in d.position('<tool_call>\n<function=',3)
    assert not d.position('<tool_call>\n<function=inspect',4)
    print('Causal name-choice counterchecks passed; pretrained/API calls0')
