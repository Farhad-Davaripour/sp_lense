"""Exact attempted/complete count ledger, including load and absolute worker time."""
import time
from support import require
class Counts:
    def __init__(self,deadline,publisher):
        self.deadline,self.publisher=deadline,publisher;self.attempts={'load':0,'forward':0,'derivative':0};self.failed=False;self.events=[]
    def reserve(self,kind):
        require(not self.failed and time.monotonic()<self.deadline,'DISPATCH_STOP_TIME')
        require(kind in self.attempts and self.attempts[kind]<{'load':1,'forward':72,'derivative':16}[kind],'DISPATCH_COUNT')
        self.attempts[kind]+=1
        self.publisher(f'counts/{kind}_{self.attempts[kind]:02d}.json',{'kind':kind,'attempt':self.attempts[kind],'monotonic':time.monotonic()})
    def record(self):return {'attempts':dict(self.attempts),'encoding':0,'limits':{'load':1,'forward':72,'derivative':16}}
