"""Saved constructor metadata AND-join; never grants scientific acceptance."""
import json
from support import HERE,sha,require
from constructor_operands import FILE_CAPS,source_identity
def verify_operands(control,execution,status,closed_status):
    result={"binding_verified":False,"scientific_pass":False,"interpretation":"OPERANDS_INCOMPLETE"}
    if status is None or status!=closed_status:return result
    require(status["execution"]==execution and status["source"]==source_identity(),"constructor closed source/execution")
    packets={}
    for name,pointer in status["pointers"].items():
        require(name in FILE_CAPS,"constructor fixed files")
        path=control/name;raw=path.read_bytes()
        require(len(raw)==pointer["bytes"]<=FILE_CAPS[name] and sha(raw)==pointer["sha256"]
            and pointer["path"]==path.relative_to(HERE).as_posix(),"constructor raw native acknowledgement")
        value=json.loads(raw);require(value["execution"]==execution and value["source"]==status["source"]
            and value["schema"]=="constructor_operands.v1","constructor packet joins")
        packets[name]=value
    if not status["complete"] or status["io_failed"] or not status["finished"] or set(packets)!=set(FILE_CAPS):return result
    reserve=packets["CONSTRUCTOR_RESERVATION.json"]
    require(reserve["before_load"] is True and reserve["reserved_bytes"]==65536 and reserve["file_caps"]==FILE_CAPS,"constructor original reserve")
    terminal=packets["CONSTRUCTOR_TERMINAL.json"]
    require(terminal["phase"]=="TERMINAL" and terminal["primary"]==status["primary"] and terminal["io_failed"] is False
        and terminal["prior_pointers"]=={k:v for k,v in status["pointers"].items() if k!="CONSTRUCTOR_TERMINAL.json"},"constructor first cause/phase join")
    for phase in ("BEFORE","AFTER"):
        value=packets["CONSTRUCTOR_"+phase+".json"]
        require(value["phase"]==phase and value["operands"]["observational_only"] is True,"constructor exact phase")
    result.update(binding_verified=True,interpretation="COMPLETE_OBSERVATIONAL_OPERANDS",primary=status["primary"],
        before=packets["CONSTRUCTOR_BEFORE.json"]["operands"],after=packets["CONSTRUCTOR_AFTER.json"]["operands"])
    return result
