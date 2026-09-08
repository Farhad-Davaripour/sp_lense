"""Saved helper AND join; original diagnostic reader remains independently necessary."""
import json
from support import HERE,require,sha
from helper_binding import ensure_package,source_lock,reservation_value
from helper_limits import FILE_CAPS

def verify_helper(control,execution,terminal_status,worker_status):
    result={"status":"HELPER_BINDING_INCOMPLETE","binding_verified":False,"scientific_pass":False}
    try:
        package=ensure_package()
        require(type(terminal_status) is dict and terminal_status==worker_status,"closed worker helper status join")
        live=dict(worker_status);outer=live.pop("outer_pointer")
        require(type(outer) is dict and live["first_code"] is None and live["outer_write_failed"] is False,
                "finite complete helper outer status")
        def native(name,cap):
            path=control/name
            require(path.stat().st_size<=cap,"saved helper cap")
            return path.read_bytes()
        def acknowledge(name,raw,pointer):
            require(pointer=={"path":(control/name).relative_to(HERE).as_posix(),"bytes":len(raw),"sha256":sha(raw)},
                    "helper native exact acknowledged pointer")
        raw=native("HELPER_OUTER_STATUS.json",FILE_CAPS["HELPER_OUTER_STATUS.json"])
        acknowledge("HELPER_OUTER_STATUS.json",raw,outer)
        require(json.loads(raw)==live and live["execution"]==execution and live["source_lock"]==source_lock(),
                "helper outer source execution exact bytes")
        reserve=native("HELPER_RESERVATION.json",16384)
        acknowledge("HELPER_RESERVATION.json",reserve,live["reservation_pointer"])
        require(json.loads(reserve)==reservation_value(execution),"helper reserved within existing limits before load")
        admitted=native("HELPER_ADMISSION.json",FILE_CAPS["HELPER_ADMISSION.json"])
        acknowledge("HELPER_ADMISSION.json",admitted,live["admission_pointer"])
        expected=json.loads(admitted)
        result=package["saved_reader"].judge(lambda name:native(name,FILE_CAPS[name]),
            execution=execution,source_lock=source_lock(),controller_status=live["session"],expected_setup=expected)
        require(result["scientific_pass"] is False,"helper never grants scientific pass")
        return result
    except BaseException:
        return {"status":"HELPER_BINDING_INCOMPLETE","binding_verified":False,"scientific_pass":False}
