"""Minimal namespace/64 MiB/source-provider binding; scientific functions unchanged."""
import sys
from support import HERE, ROOT, EVIDENCE, MIB, SOURCES, bounds, require

WORKFLOW_COMMIT = "638ef2a8f1eac0679c92d07a62cc8ae9bfac41b1"
WORKFLOW = "diagnostics/fresh_confirmation_workflow_v1/"


def bind():
    pins = SOURCES.load("pins",WORKFLOW_COMMIT,WORKFLOW+"pins.py")
    pins.HERE = HERE  # Sole output namespace adaptation, never an old-file write.
    pins.ROOT = ROOT
    pins.subprocess = SOURCES
    original_io = pins.io_components

    def local_io():
        writer,reader = original_io()
        sys.modules["binding"].subprocess = SOURCES
        reader.subprocess = SOURCES
        return writer,reader

    pins.io_components = local_io
    area = SOURCES.load("area",WORKFLOW_COMMIT,WORKFLOW+"area.py")
    area.area_bounds = bounds
    require(area.EVIDENCE == EVIDENCE, "new evidence root")
    original_writer = area.WorkflowWriter

    class DeltaWriter(original_writer):
        def _write(self, category, name, data, reserve, **kwargs):
            with self.lock:
                self._owner()
                try:
                    require(bounds()["evidence_bytes"]+len(data) <= 56*MIB,
                            "64 MiB delta area retains shared 8 MiB closeout reserve")
                except BaseException as error:
                    self._fail("delta_area",error)
                    raise
                return super()._write(category,name,data,reserve,**kwargs)

        def closeout(self, name="closeout/index.json"):
            require(bounds()["evidence_bytes"]+5*MIB <= 64*MIB, "delta native closeout headroom")
            return super().closeout(name)

    area.WorkflowWriter = DeltaWriter
    SOURCES.load("schedule",WORKFLOW_COMMIT,WORKFLOW+"schedule.py")
    workflow = SOURCES.load("workflow",WORKFLOW_COMMIT,WORKFLOW+"workflow.py")
    judge = SOURCES.load("judge",WORKFLOW_COMMIT,WORKFLOW+"judge.py")
    return workflow,judge
