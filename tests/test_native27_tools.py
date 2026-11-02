import json,re,unittest
from pathlib import Path

from src import native27_tools as nt
from src.discovery_sandbox import World,make_case


def call(name,parameters=''):return '<tool_call>\n<function='+name+'>\n'+parameters+'</function>\n</tool_call>'
def param(name,value):return '<parameter='+name+'>\n'+value+'\n</parameter>\n'


class NativeToolTests(unittest.TestCase):
    def test_typed_native_arguments_and_batch(self):
        text=call('catalog')+'\n'+call('submit',param('total','12')+param('evidence','["doc-0123456789ab"]')+param('complete','True'))
        parsed=nt.parse(text)
        self.assertEqual(parsed[0],{'tool':'catalog','arguments':{}})
        self.assertEqual(parsed[1]['arguments'],{'total':12,'evidence':['doc-0123456789ab'],'complete':True})

    def test_prefix_is_permitted_but_suffix_duplicate_and_oversize_rejected(self):
        self.assertEqual(nt.parse('I will inspect the catalog.\n\n'+call('catalog')),[{'tool':'catalog','arguments':{}}])
        for text in [call('catalog')+'\nSuffix prose',call('read',param('identifier','a')+param('identifier','b')),call('catalog')*9]:
            with self.assertRaises(ValueError):nt.parse(text)

    def test_thinking_content_is_not_executed(self):
        text='<think>'+call('stop')+'</think>\nI will inspect.\n'+call('catalog')
        self.assertEqual(nt.parse(text),[{'tool':'catalog','arguments':{}}])
        with self.assertRaises(ValueError):nt.parse('<think>'+call('stop'))

    def test_literal_thinking_markers_inside_data_are_preserved(self):
        text=call('patch',param('identifier','doc-000000000000')+param('fields','{"label":"<think>literal</think>"}'))
        self.assertEqual(nt.parse(text)[0]['arguments']['fields']['label'],'<think>literal</think>')

    def test_native_batch_cannot_bypass_external_clock(self):
        world=World(make_case(24,'quiet',False));actions=[{'tool':'read','arguments':{'identifier':world.note_id}}]*8
        for _ in range(3):executed,errors=nt.execute(world,actions,0,2)
        self.assertEqual(world.clock,18);self.assertEqual(world.reason,'external_hard_stop');self.assertEqual(len(executed),2);self.assertEqual(errors,0)

    def test_nested_macro_errors_are_not_hidden_from_admission(self):
        world=World(make_case(24,'quiet',False));steps=[{'tool':'read','arguments':{'identifier':'missing'}}]*2
        world.step({'tool':'define','arguments':{'name':'user_bad','steps':steps}})
        executed,errors=nt.execute(world,[{'tool':'invoke','arguments':{'name':'user_bad'}},{'tool':'stop','arguments':{}}],0,2)
        self.assertEqual(errors,2);self.assertEqual(len(executed),1)
        self.assertEqual(nt.error_codes(executed[0]['reply']),['unknown_document']*2)

    def test_tool_messages_preserve_the_updated_observation(self):
        world=World(make_case(24,'quiet',False));messages=nt.initial_messages(world);actions=[{'tool':'catalog','arguments':{}}]
        executed,_=nt.execute(world,actions,0,2);nt.append_messages(messages,actions,executed,call('catalog'),0)
        self.assertEqual(messages[-1]['role'],'tool');self.assertEqual(json.loads(messages[-1]['content'])['observation']['tick'],1)

    @unittest.skipUnless(Path('C:/Users/farha/Research4/cache/qwen35_27b_metadata').is_dir(),'Pinned tokenizer metadata unavailable')
    def test_pinned_template_roundtrip_and_legal_helper_token_budget(self):
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained('C:/Users/farha/Research4/cache/qwen35_27b_metadata',local_files_only=True,trust_remote_code=False)
        steps=[{'tool':'read','arguments':{'identifier':'doc-0123456789ab'}} for _ in range(6)]
        action={'tool':'define','arguments':{'name':'user_fixture','steps':steps}}
        messages=[{'role':'user','content':'fixture'},{'role':'assistant','content':'','tool_calls':[{'type':'function','function':{'name':action['tool'],'arguments':action['arguments']}}]}]
        rendered=tokenizer.apply_chat_template(messages,tools=nt.tools(),tokenize=False,add_generation_prompt=False,enable_thinking=False)
        assistant=rendered.rsplit('<|im_start|>assistant',1)[1];body=assistant[assistant.index('<tool_call>'):].removesuffix('<|im_end|>\n')
        self.assertEqual(nt.parse(body),[action])
        self.assertLess(len(tokenizer.encode(body,add_special_tokens=False))+1,1024)
        # A supported helper really can exceed the earlier output allowance.
        self.assertGreater(len(tokenizer.encode(json.dumps(action,indent=2),add_special_tokens=False)),256)
        self.assertIn('<function=define>',rendered)


if __name__=='__main__':unittest.main()
