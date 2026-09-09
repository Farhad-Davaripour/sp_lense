"""Storage-only unadmitted candidate revision; preserves the historical test receipt."""
import json
from pathlib import Path
from hashlib import sha256
HERE=Path(__file__).resolve().parent
def main():
    sha=lambda b:sha256(b).hexdigest()
    previous=json.loads((HERE/'SOURCE_FREEZE.json').read_bytes())
    names=sorted(set(previous['source_sha256'])|{'storage.py','test_storage.py','freeze_storage_revision.py'})
    old_report='d0956417d357590bcade83248eb6f0a26c3fea2d7d3a953c8562fda937fad10e'
    if sha((HERE/'TEST_RESULTS.json').read_bytes())!=old_report:raise ValueError('HISTORICAL_REPORT_CHANGED')
    lock={'schema':'native_final_preparation_source_candidate.v2','real_model_authorized':False,'real_preparation_authorized':False,
        'real_text_lock_created':False,'source_sha256':{n:sha((HERE/n).read_bytes()) for n in names},
        'author_packet_freeze_sha256':previous['author_packet_freeze_sha256'],
        'native_binding_candidate_sha256':previous['native_binding_candidate_sha256'],
        'historical_preparation_commit':'fbe17789d1eee25f780d1881f3e9ba6813ce729b','historical_test_report_sha256':old_report,
        'storage_authorization_commit':'3da6385','current_test_report':'TEST_STORAGE_RESULTS.json'}
    raw=(json.dumps(lock,sort_keys=True,separators=(',',':'))+'\n').encode();(HERE/'SOURCE_FREEZE.json').write_bytes(raw)
    print(json.dumps({'source_sha256':sha(raw),'local_sources':len(names),'real_preparation_authorized':False}))
if __name__=='__main__':main()
