"""One shared cleanup allowance; stricter setup subbudget never expands contract."""
from support import require

class Budget:
    def __init__(self,started,*,testing=False):
        self.started=started;self.substantive_end=started+(75 if testing else 240)
        self.absolute_end=started+(90 if testing else 255);self.used=0.;self.charges=[]
    @property
    def remaining(self): return max(0.,15.-self.used)
    def deadlines(self,now,lane):
        require(lane in ("worker","audit") and self.remaining>0 and now<self.substantive_end,"setup deadline or cleanup exhausted")
        end=min(now+(180 if lane=="worker" else 60),self.substantive_end,self.absolute_end-self.remaining)
        return end,min(end+self.remaining,self.absolute_end-1.)  # Final native parent receipt headroom.
    def charge(self,key,seconds):
        require(key not in [x["key"] for x in self.charges] and seconds>=0,"one cleanup charge")
        self.used+=seconds;self.charges.append({"key":key,"seconds":seconds})
        require(self.used<=15,"shared cleanup overrun")
    def record(self):
        return {"used_seconds":self.used,"maximum_seconds":15,"remaining_seconds":self.remaining,
            "charges":list(self.charges),"absolute_end":self.absolute_end,"substantive_end":self.substantive_end}
