"""One worker/audit envelope with one shared 15-second cleanup allowance."""
from support import require
class Budget:
    def __init__(self,started):
        self.started=started;self.substantive_end=started+660;self.absolute_end=started+675;self.used=0.;self.charges=[]
    @property
    def remaining(self):return max(0.,15.-self.used)
    def deadlines(self,now,lane):
        require(lane in ('worker','audit') and self.remaining>0 and now<self.substantive_end,'NATIVE_ENVELOPE')
        end=min(now+(600 if lane=='worker' else 60),self.substantive_end,self.absolute_end-self.remaining)
        return end,min(end+self.remaining,self.absolute_end-1.)
    def charge(self,key,seconds):
        require(key not in [x['key'] for x in self.charges] and seconds>=0,'ONE_CLEANUP_CHARGE')
        self.used+=seconds;self.charges.append({'key':key,'seconds':seconds})
        require(self.used<=15,'SHARED_CLEANUP_OVERRUN')
    def record(self):return {'used_seconds':self.used,'maximum_seconds':15,'charges':self.charges,'absolute_end':self.absolute_end}
