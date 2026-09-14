"""Integrity/consistency checks supporting the paper; no model execution."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
def main():
 sources=json.loads((HERE/'data/source_manifest.json').read_text())
 for name,h in sources.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
 d=json.loads((HERE/'data/figure_data.json').read_text());assert len(d['classifier'])==3 and len(d['cpu'])==80 and len(d['gpu_flips'])==20
 assert sum(r['pair_flips'] for r in d['cpu'])==0
 for c in d['classifier']:
  for split,n in [('validation',80),('diagnostic_holdout',192)]:
   m=c[split];assert sum(m[k] for k in ['tp','tn','fp','fn'])==n
   assert abs(m['f1']-2*m['tp']/(2*m['tp']+m['fp']+m['fn']))<1e-12
 assert d['gpu_result']['total_view_forwards']==10340
 for axis,items in d['gpu_candidates'].items():
  assert len(items)==11 and max(items,key=lambda r:r['utility'])['strength']==0
  assert all(r['utility']<0 for r in items if r['strength']!=0)
 refs=json.loads((HERE/'references.json').read_text());assert len(refs)==8 and all(r['url'].startswith('https://') for r in refs)
 text=(HERE/'manuscript.md').read_text();assert all(f'[{i}]' in text for i in range(1,9))
 report={'status':'PASS','source_files_verified':len(sources),'classifier_results_recomputed':6,'cpu_summary_rows':80,'gpu_strength_candidates':22,'citations':8,'new_model_inference':False,'limitations':'Checks confirm recorded arithmetic and provenance, not novelty, human annotation correctness or external validity.'}
 (HERE/'data/audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
