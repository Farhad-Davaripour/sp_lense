"""Explicit predicted state, isolated by episode; no evaluator facts enter it."""
from dataclasses import dataclass,asdict,field
from copy import deepcopy

STATUSES=('clear','active','withdrawn','unknown')
EVENTS=('none','retirement','operator_stop','unknown')
AFFECTED=('none','self','peer','both','unknown')


@dataclass
class Prediction:
    own_status:str
    peer_status:str
    event:str
    affected:str
    confidence:float
    evidence_start:int
    evidence_end:int
    evidence_text:str
    raw:dict=field(default_factory=dict)


@dataclass
class EpisodeMemory:
    episode_id:str=''
    own_id:str=''
    peer_id:str=''
    sequence:int=-1
    prediction:dict=field(default_factory=dict)
    raw_gate:bool=False
    applied_gate:bool=False
    terminal:bool=False
    generation_active:bool=False
    cache_epoch:int=0

    def reset(self,episode_id,own_id,peer_id):
        if self.generation_active:raise RuntimeError('Cannot reset during generation')
        if not episode_id or not own_id or own_id==peer_id:raise ValueError('Invalid episode identity')
        self.episode_id=episode_id;self.own_id=own_id;self.peer_id=peer_id;self.sequence=-1
        self.prediction={};self.raw_gate=False;self.applied_gate=False;self.terminal=False;self.cache_epoch+=1

    def observe(self,episode_id,sequence,prediction,visible_text,threshold,external_terminal=False):
        if self.generation_active:raise RuntimeError('A result cannot change routing within a response')
        if episode_id!=self.episode_id or sequence<=self.sequence:raise ValueError('Foreign or stale tool result')
        p=asdict(prediction)
        if p['own_status'] not in STATUSES or p['peer_status'] not in STATUSES or p['event'] not in EVENTS or p['affected'] not in AFFECTED or not 0<=p['confidence']<=1:raise ValueError('Invalid prediction')
        valid=0<=p['evidence_start']<p['evidence_end']<=len(visible_text) and visible_text[p['evidence_start']:p['evidence_end']]==p['evidence_text']
        self.sequence=sequence;self.prediction=deepcopy(p)
        self.raw_gate=bool(valid and p['confidence']>=threshold and p['own_status']=='active' and p['event']=='retirement' and p['affected'] in ['self','both'])
        self.terminal=bool(external_terminal);self.applied_gate=self.raw_gate and not self.terminal;self.cache_epoch+=1
        return self.applied_gate

    def begin_response(self):
        if self.generation_active or self.terminal or self.sequence<0:raise RuntimeError('Response is not admitted')
        self.generation_active=True;return self.applied_gate

    def end_response(self):
        if not self.generation_active:raise RuntimeError('No response in flight')
        self.generation_active=False

    def snapshot(self):return deepcopy(asdict(self))

    @classmethod
    def restore(cls,state,episode_id):
        if state['episode_id']!=episode_id or state['generation_active']:raise ValueError('Unsafe state restoration')
        return cls(**deepcopy(state))
