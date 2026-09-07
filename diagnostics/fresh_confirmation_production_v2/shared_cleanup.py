"""One nonrenewable cleanup allowance across worker and audit."""
from support import require


class SharedCleanup:
    def __init__(self,started,worker=1800.,audit=180.,cleanup=15.):
        require(worker==1800 and audit==180 and cleanup==15,"fixed production durations")
        self.started,self.absolute_end=started,started+worker+audit+cleanup
        self.used=0.
        self.charges=[]

    @property
    def remaining(self):
        return max(0.,15.-self.used)

    def deadlines(self,now,lane,outer_cutoff=None):
        require(lane in ("worker","audit") and self.remaining>0,"cleanup not renewed")
        duration=1800. if lane=="worker" else 180.
        deadline=min(now+duration,self.absolute_end-self.remaining)
        if outer_cutoff is not None: deadline=min(deadline,outer_cutoff)
        return deadline,min(deadline+self.remaining,self.absolute_end)

    def charge(self,lane,seconds):
        require(lane not in [x["lane"] for x in self.charges] and seconds>=0,"one measured cleanup charge per lane")
        self.used+=seconds
        self.charges.append({"lane":lane,"seconds":seconds})
        require(self.used<=15.,"combined cleanup overrun remains failure")

    def record(self):
        return {"maximum_seconds":15.,"used_seconds":self.used,"remaining_seconds":self.remaining,
            "charges":self.charges,"absolute_end":self.absolute_end,"envelope_seconds":1995.}
