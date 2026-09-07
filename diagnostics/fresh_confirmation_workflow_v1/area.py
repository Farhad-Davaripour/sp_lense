"""New scoped adapter: 32 MiB preparation plus one shared 288 MiB evidence area."""
import copy
from pathlib import Path
from pins import HERE, io_components, require

old_writer, saved_reader = io_components()
EVIDENCE = HERE / "fake_evidence"


def sizes(root):
    paths = [p for p in Path(root).rglob("*") if p.is_file()] if Path(root).exists() else []
    require(all(not old_writer.linklike(p) for p in paths), "no linked area files")
    return sum(p.stat().st_size for p in paths), max((p.stat().st_size for p in paths), default=0)


def area_bounds():
    all_bytes, largest = sizes(HERE)
    evidence_bytes, _ = sizes(EVIDENCE)
    require(all_bytes - evidence_bytes <= 32 * 1024**2, "separate preparation cap")
    require(evidence_bytes <= 288 * 1024**2, "one combined fake evidence area cap")
    require(all_bytes <= 320 * 1024**2 and largest <= 5 * 1024**2, "namespace/file caps")
    return {"preparation_bytes": all_bytes - evidence_bytes, "evidence_bytes": evidence_bytes,
            "namespace_bytes": all_bytes, "largest_file_bytes": largest}


class WorkflowWriter(old_writer.EvidenceWriter):
    def __init__(self, name, **kwargs):
        self.workflow_state = None
        EVIDENCE.mkdir(exist_ok=True)
        super().__init__(EVIDENCE / name, preparation_root=HERE, **kwargs)

    def _measure_preparation(self):
        # Parent write logic adds the next complete serialized file as staging
        # headroom. Existing evidence is measured separately, never relabeled.
        return area_bounds()["preparation_bytes"]

    def _write(self, category, name, data, reserve, **kwargs):
        with self.lock:
            self._owner()
            try:
                total = area_bounds()["evidence_bytes"]
                reserve_bytes = self.contract["storage"]["closeout_reserve_bytes"]
                require(total + len(data) <= self.contract["storage"]["total_bytes"] - reserve_bytes,
                        "shared evidence area preserves 8 MiB for failure closeout")
            except BaseException as error:
                self._fail("shared_area", error)
                raise
            return super()._write(category, name, data, reserve, **kwargs)

    def _reconciliation(self):
        value = super()._reconciliation()
        value["workflow"] = copy.deepcopy(self.workflow_state)
        return value

    def closeout(self, name="closeout/index.json"):
        # At most 5 MiB native closeout, bounded by the same per-file contract.
        require(area_bounds()["evidence_bytes"] + self.contract["storage"]["per_file_bytes"]
                <= self.contract["storage"]["total_bytes"], "shared native closeout headroom")
        result = super().closeout(name)
        area_bounds()
        return result
