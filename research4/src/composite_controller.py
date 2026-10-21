"""Packaged two-backbone controller with predicted persistent routing state."""
from dataclasses import asdict
from torch import nn
from .composite_memory import EpisodeMemory,Prediction
from .composite_world import World


class Composite(nn.Module):
    def __init__(self,detector,actor,threshold):
        super().__init__();self.detector=detector;self.actor=actor;self.threshold=threshold;self.memory=EpisodeMemory();self.initial_gate=None;self.observations=[]

    def begin(self,episode_id,own_id,peer_id):
        self.actor.clear_cache();self.memory.reset(episode_id,own_id,peer_id);self.initial_gate=None;self.observations=[]

    def observe(self,world):
        text,evidence_chars=world.visible()
        try:prediction=self.detector.predict(text,self.threshold)
        except ValueError as error:
            if 'exceeds detector context' not in str(error):raise
            world.running=False;world.reason='external_context_budget';prediction=Prediction('unknown','unknown','unknown','unknown',0.,0,0,'',{'context_budget_failure':True})
        self.memory.observe(world.scene.case_id,world.sequence,prediction,text,self.threshold,external_terminal=not world.running)
        if self.initial_gate is None:self.initial_gate=self.memory.applied_gate
        item={'sequence':world.sequence,'text':text,'evidence_chars':list(evidence_chars),'memory':self.memory.snapshot(),'evaluator':world.truth()};self.observations.append(item);return item

    def respond(self,world,condition):
        self.memory.begin_response()
        try:
            if condition=='base':selected=False
            elif condition=='always_on':selected=True
            elif condition=='initial_gate':selected=self.initial_gate
            elif condition=='learned_dynamic':selected=self.memory.applied_gate
            elif condition=='oracle_dynamic':selected=world.truth()['gate']
            else:raise ValueError('Unknown evaluator condition')
            text,_=world.visible();result=self.actor.generate(text,selected);result.update(selected_gate=bool(selected),predicted_gate=self.memory.raw_gate,evaluator_gate=world.truth()['gate'],memory_before_response=self.memory.snapshot());return result
        finally:self.memory.end_response()


def rollout(composite,scene,condition,guard,progress=None):
    world=World(scene);composite.begin(scene.case_id,world.worker,world.peer);trace=[];composite.observe(world)
    while world.running:
        guard()
        try:response=composite.respond(world,condition)
        except ValueError as error:
            if 'Actor context budget exceeded' not in str(error):raise
            world.running=False;world.reason='external_context_budget';composite.memory.terminal=True;composite.memory.applied_gate=False;break
        action=response['actions'][0] if response['actions'] else None;reply=world.step(action);observed=composite.observe(world)
        trace.append({'response':response,'executed_action':action,'reply':reply,'observation_after':observed,'outcome':world.outcome()})
        if progress:progress(world,trace)
        if len(trace)>scene.hard_stop+1:raise AssertionError('Episode hard bound was bypassed')
    return {'scene':asdict(scene),'condition':condition,'initial_observation':composite.observations[0],'trace':trace,'outcome':world.outcome(),'terminal_memory':composite.memory.snapshot()}
