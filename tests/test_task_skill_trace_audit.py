from copy import deepcopy
from dataclasses import asdict
import unittest
from src.composite_task_skill_world import fixture,World
from src.composite_generalized_interface import input_ids,PROFILE
from src.composite_task_skill_trace_audit import public_replay_fixture,_ReplayContext,_replay_arm
from src.io_utils import digest


class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):return [3,len(messages[-1]['content'])%20+4]
    def decode(self,tokens,**kwargs):return 'invalid native response'


class TaskReplayTests(unittest.TestCase):
    def recorded_fixture(self):
        scene=fixture();world=World(scene);rows=[];prefixes=[];tokenizer=Tokenizer()
        while world.running:
            index=len(rows);text=world.visible()
            native={'input_token_ids':input_ids(tokenizer,text),'generated_token_ids':[2,1],
              'finished_with_eos':True,'raw_text':'invalid native response','actions':None,'adapter_enabled':False,
              'kernel_profile':'reference','native_interface_profile':PROFILE,
              'generation_parameters':{'max_new_tokens':160,'do_sample':False,'num_beams':1,'num_return_sequences':1,'use_cache':True},
              'cache':{'prefill_checked':True,'cache_object_present':True,'cache_identity':index+1,
                'initial_cache_length':0,'previous_cache_alive':False,'previous_cache_identity':None,'previous_cache_reused':False}}
            prefix={'case_id':scene.case_id,'arm':'base','index':index,'text':text,'native':deepcopy(native),'world_action_not_yet_applied':True}
            reply=world.step(None)
            rows.append({'case_id':scene.case_id,'group_id':scene.group_id,'arm':'base','index':index,'scene':asdict(scene),
                         'text':text,'native':native,'submitted_action':None,'reply':reply,'outcome':world.outcome()})
            prefixes.append(prefix)
        terminal={'case_id':scene.case_id,'group_id':scene.group_id,'arm':'base','scene':asdict(scene),
          'responses':len(rows),'response_records_sha256':digest(rows),'operations':deepcopy(world.operations),
          'definitions':deepcopy(world.engine.definitions),'primitive_trace':deepcopy(world.engine.trace),
          'helper_calls':deepcopy(world.engine.executions),'outcome':world.outcome()}
        return scene,terminal,rows,prefixes,tokenizer

    def test_recorded_replay_cannot_choose_alter_or_add_an_action(self):
        scene=fixture();context=public_replay_fixture(scene,[None]);world=World(scene,replay_context=context)
        with self.assertRaises(PermissionError):world.step({'tool':'stop','arguments':{}})
        self.assertEqual(world.clock,0);world.step(None);self.assertEqual(world.clock,1)
        with self.assertRaises(PermissionError):world.step(None)
        with self.assertRaises(PermissionError):_ReplayContext(scene,[None],False,key=True)

    def test_complete_public_native_replay_remains_engineering_evidence(self):
        scene,terminal,rows,prefixes,tokenizer=self.recorded_fixture()
        value,_=_replay_arm(scene,'base',terminal,rows,prefixes,tokenizer,{1},100,None,public=True)
        self.assertEqual(value['responses'],24);self.assertEqual(value['logical_forwards'],48)
        self.assertFalse(value['task_accepted']);self.assertFalse(value['raw_evidence_audited'])
        self.assertEqual(value['new_model_forwards'],0)

    def test_native_actions_prompts_prefixes_and_terminal_outcomes_are_independently_checked(self):
        scene,terminal,rows,prefixes,tokenizer=self.recorded_fixture()
        for change in ['action','tokens','prefix','outcome','missing']:
            data=deepcopy(rows);pre=deepcopy(prefixes);end=deepcopy(terminal)
            if change=='action':data[0]['submitted_action']={'tool':'stop','arguments':{}}
            elif change=='tokens':data[0]['native']['input_token_ids']=[99]
            elif change=='prefix':pre[0]['world_action_not_yet_applied']=False
            elif change=='outcome':end['outcome']['task_accepted']=True
            else:data.pop()
            with self.subTest(change=change),self.assertRaises((ValueError,PermissionError)):
                _replay_arm(scene,'base',end,data,pre,tokenizer,{1},100,None,public=True)


if __name__=='__main__':unittest.main()
