"""Read-only production comparison; writes only this review's result receipt."""
import hashlib,json
from pathlib import Path
REVIEW=Path(__file__).resolve().parent
ROOT=REVIEW.parents[2]
prior=json.loads((REVIEW/'RESULT.json').read_bytes())
mirror=Path(prior['mirror_path'])
CAP='development/native_supervised_gate_capture_v1'
PREP='development/native_supervised_gate_preparation_v3'
GATE='development/native_supervised_gate_v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
expected={CAP+'/SOURCE_FREEZE.json':'b5a29962bb4850950cf95f3f6ac93e78d70954f74779b507165cbc4d061b3a7d',
    GATE+'/source_auth.py':'dc333f85b04d383ed3c0542931bbc1844bf5c19cf096b02bdbeb34db02adbcaa',
    GATE+'/SOURCE_FREEZE.json':'b60a2b58102093e194c63b885b29a946eabfa42b0801be0d86d4f1e5cb531d57'}
assert all(sha(ROOT/p)==h for p,h in expected.items())
current=(ROOT/GATE/'source_auth.py').read_text();old=(mirror/GATE/'source_auth.py').read_text()
def normalized(text):return '\n'.join(line for line in text.splitlines() if not line.startswith('CAPTURE_SOURCE_SHA256='))
assert normalized(current)==normalized(old)
assert "CAPTURE_SOURCE_SHA256='"+expected[CAP+'/SOURCE_FREEZE.json']+"'" in current
cf=json.loads((ROOT/CAP/'SOURCE_FREEZE.json').read_bytes())
old_cf=json.loads((mirror/CAP/'SOURCE_FREEZE.json').read_bytes())
assert all(cf[k]==old_cf[k] for k in cf if k!='external_sources')
assert len(cf['external_sources'])==1
pin=cf['external_sources'][0]
assert Path(pin['path']).is_absolute() and Path(pin['path']).resolve()==(ROOT/PREP/'SOURCE_FREEZE.json').resolve()
assert sha(Path(pin['path']))==pin['sha256']==prior['tested_original_hashes']['development\\native_supervised_gate_preparation_v3\\SOURCE_FREEZE.json']
gf=json.loads((ROOT/GATE/'SOURCE_FREEZE.json').read_bytes())
assert gf['source_sha256']['source_auth.py']==expected[GATE+'/source_auth.py']
assert gf['external_sources']==[{'path':CAP+'/SOURCE_FREEZE.json','sha256':expected[CAP+'/SOURCE_FREEZE.json']}]
for name,digest in gf['source_sha256'].items():assert sha(ROOT/GATE/name)==digest
result={'status':'PASS_METADATA_ONLY_FINAL_DELTA','verified_final_hashes':expected,
    'source_auth_delta':'literal CAPTURE_SOURCE_SHA256 only','function_bodies_identical':True,
    'capture_delta':'one absolute external path, unchanged source map and preparation bytes',
    'gate_source_and_external_joins_verified':True,'prior_unmocked_fixture_proof_reused':True,
    'additional_admission_runs':0,'model_calls':0,'tokenizer_calls':0,'fits':0,'production_source_edits':0}
(REVIEW/'FINAL_METADATA_RESULT.json').write_text(json.dumps(result,sort_keys=True,separators=(',',':'))+'\n')
print(json.dumps(result))
