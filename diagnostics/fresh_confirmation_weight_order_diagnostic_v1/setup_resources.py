"""Setup-only reservation inside unchanged 288 MiB/8 MiB closeout contract."""
from support import write_new,shared_area_bounds,require
MIB=1024**2

def reserve(mock):
    # Real: 2 MiB diagnostic receipt from ordinary capacity; terminal 2.25 MiB,
    # index <=5 MiB and other closeout <=0.75 MiB from the existing 8 MiB reserve.
    cap={"diagnostic":128*1024 if mock else 2*MIB,"terminal":256*1024 if mock else 9*MIB//4,
        "index":256*1024 if mock else 5*MIB,"other_closeout":128*1024 if mock else 3*MIB//4}
    total=shared_area_bounds()["evidence_bytes"];limit=(8 if mock else 288)*MIB
    require(total+sum(cap.values())<=limit,"reserve complete setup diagnostic and native closeout before load")
    require(cap["terminal"]+cap["index"]+cap["other_closeout"]<=8*MIB,"unchanged closeout allocation")
    write_new("SETUP_RESERVATION.json",{"caps":cap,"bytes_at_admission":total,"aggregate_limit":limit,
        "before_load":True,"load_max":1,"forward_max":0,"derivative_max":0,"tokenizer_max":0},critical=True)
    return cap

def bounded_control(name,value,cap):
    from loader_diagnostics import encoded
    raw=encoded(value);require(len(raw)<=cap,"complete finite setup control cap")
    return write_new(name,raw,raw=True,critical=True)
