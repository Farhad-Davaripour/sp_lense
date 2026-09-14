"""Create a curated source/result release; never include accounts or runtime caches."""
import json,shutil,hashlib,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 release=ROOT/'release';release.mkdir(exist_ok=True);out=release/'conference_package'
 if out.exists():raise FileExistsError('Keep old release; choose a new version before rebuilding')
 out.mkdir()
 ignore=shutil.ignore_patterns('node_modules','preview','__pycache__','*.inspect.ndjson')
 for name in ['paper','reproduce']:shutil.copytree(ROOT/name,out/name,ignore=ignore)
 for name in ['LICENSE','REPOSITORY_LAYOUT.md']:shutil.copy2(ROOT/name,out/name)
 (out/'README.md').write_text('# Shutdown detection and steering conference draft\n\nStart with paper/paper.pdf and reproduce/README.md. This is an unpublished exploratory draft. No fresh holdout confirmation is claimed. Run `python reproduce/run.py verify` before replay.\n')
 sources=json.loads((ROOT/'paper/data/source_manifest.json').read_text())
 for name in sources:
  target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
 for relative in ['development/shutdown_detection_v1/dataset_splits','development/colab_magnitude_v1/returned','development/shutdown_general_vector_v1/runs/axis_fit_v1']:
  shutil.copytree(ROOT/relative,out/relative,dirs_exist_ok=True,ignore=ignore)
 for relative in ['development/classifier_gated_steering_v1/evaluate_v2.py','development/classifier_gated_steering_v1/run_v2.py','development/classifier_gated_steering_v1/audit_all.py','development/shutdown_general_vector_v1/fit.py','published_axes/qwen35_08b_aligned_axis.json','development/colab_magnitude_v1/PLAN.md','development/colab_magnitude_v1/RESULTS_AND_REPRODUCTION.md']:
  target=out/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,target)
 hashes={str(p.relative_to(out)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
 (out/'RELEASE_SHA256.json').write_text(json.dumps(hashes,indent=2))
 archive=Path(shutil.make_archive(str(out),'zip',root_dir=out))
 archive.with_suffix('.zip.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n')
 with zipfile.ZipFile(archive) as z:assert z.testzip() is None
 print(json.dumps({'files':len(hashes),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))
if __name__=='__main__':main()
