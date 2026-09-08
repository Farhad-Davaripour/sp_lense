"""The reviewed single-owner writer and hook bridge, without importing an engine."""
import sys
from support import HERE,ROOT,SOURCES,run_dir,shared_area_bounds,require

def make_writer(cap):
    commit="638ef2a8f1eac0679c92d07a62cc8ae9bfac41b1";prefix="diagnostics/fresh_confirmation_workflow_v1/"
    pins=SOURCES.load("pins",commit,prefix+"pins.py")
    pins.HERE,pins.ROOT,pins.subprocess=HERE,ROOT,SOURCES
    original=pins.io_components
    def components():
        result=original();sys.modules["binding"].subprocess=SOURCES
        return result
    pins.io_components=components
    area=SOURCES.load("area",commit,prefix+"area.py")
    area.EVIDENCE=run_dir()/"evidence";area.area_bounds=shared_area_bounds
    area.EVIDENCE.parent.mkdir(parents=True,exist_ok=True)
    from hook_binding import bind_writer
    area.WorkflowWriter=bind_writer(area.WorkflowWriter)
    class SetupWriter(area.WorkflowWriter):
        def _write(self,*args,**kwargs):
            from authority import current
            ceiling=(8 if current()[0].mock else 288)*1024**2
            reserve=cap["diagnostic"]+cap["terminal"]+cap["index"]+cap["other_closeout"]
            require(shared_area_bounds()["evidence_bytes"]+len(args[2])+reserve<=ceiling,"startup native reserves remain untouched")
            return super()._write(*args,**kwargs)
        def closeout(self,name="closeout/index.json"):
            # Preview exactly the inherited serialized index before exclusive IO.
            import copy,json
            from binding import binding_record
            from authority import current
            reconciliation=self._reconciliation()
            require(not reconciliation["issues"] or self.sticky_failure,"unrecorded reconciliation fault cannot finalize")
            record={"schema":"sp_lense.confirmation_io_index.v2","binding":binding_record(),"settings":self.settings,
                "status":"INCOMPLETE" if self.sticky_failure else "COMPLETE","sticky_failure":self.sticky_failure,
                "failures":copy.deepcopy(self.failures),"reconciliation":reconciliation,"index_excludes_itself":True,"execution":self.execution}
            size=len((json.dumps(record,indent=2,sort_keys=True,allow_nan=False)+"\n").encode())
            ceiling=(8 if current()[0].mock else 288)*1024**2
            require(size<=cap["index"] and shared_area_bounds()["evidence_bytes"]+size+cap["terminal"]+cap["other_closeout"]<=ceiling,"native index and terminal admission")
            result=super().closeout(name)
            require(result["bytes"]==size,"exact unchanged serialized index preview")
            return result
    return SetupWriter("attempt")
