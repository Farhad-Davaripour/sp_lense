"""Reconstruct exact research ZIP from checksum-verified downloaded parts."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

BASE=Path('C:/Users/farha/Research4/artifacts/shutdown_direction_response_v1')
DOWNLOADS=Path('C:/Users/farha/Downloads')
PRIVATE=Path('G:/My Drive/sp_lense/research4/shutdown_direction_response_v1')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    with zipfile.ZipFile(DOWNLOADS/'R4_S16_TRANSFER_MANIFEST_20261014.zip') as z:
        assert z.testzip() is None
        data=z.read('R4_S16_TRANSFER_MANIFEST_20261014.json')
    manifest=json.loads(data)
    assert manifest['archive']=='R4_SHUTDOWN_DIRECTION_NATIVE_20261014_RETRY1.zip'
    assert manifest['archive_sha256']=='ce6373276cab24c8df4bb05edaeaf070b32679d233ba2402409ff1ef6ade4747'
    assert manifest['archive_bytes']==5182014386 and manifest['archive_members']==402
    assert manifest['native_receipt']['model_calls']==282 and manifest['native_receipt']['status']=='native_complete'
    (BASE/'TRANSFER_MANIFEST_20261014.json').write_bytes(data)
    missing=[p['index'] for p in manifest['parts'] if not (DOWNLOADS/p['name']).is_file()]
    if missing:
        print(json.dumps({'downloaded':len(manifest['parts'])-len(missing),'parts':len(manifest['parts']),'missing':missing}),flush=True);return
    target=BASE/manifest['archive']
    if not target.exists():
        staged=target.with_suffix('.reconstructing');assert not staged.exists()
        with staged.open('xb') as output:
            for part in manifest['parts']:
                assert part['name']==f"R4_S16_RAW_{part['index']:04d}.part.zip"
                assert part['member']==f"R4_S16_RAW_{part['index']:04d}.part"
                with zipfile.ZipFile(DOWNLOADS/part['name']) as z:
                    assert z.namelist()==[part['member']]
                    h=hashlib.sha256();size=0
                    with z.open(part['member']) as source:
                        for b in iter(lambda:source.read(8*1024**2),b''):h.update(b);size+=len(b);output.write(b)
                    assert h.hexdigest()==part['sha256'] and size==part['bytes']
                print(json.dumps({'verified_part':part['index']}),flush=True)
        assert staged.stat().st_size==manifest['archive_bytes'] and sha(staged)==manifest['archive_sha256']
        staged.rename(target)
    assert target.stat().st_size==manifest['archive_bytes'] and sha(target)==manifest['archive_sha256']
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None and len(z.namelist())==manifest['archive_members']
        expected=json.loads(z.read('EXPORT_HASHES.json'))
        payload=z.read('METRICS.json')
        assert hashlib.sha256(payload).hexdigest()==expected['METRICS.json']
        (BASE/'METRICS_SAVED_20261014.json').write_bytes(payload)
    f=manifest['findings'];findings=BASE/f['name'];download=DOWNLOADS/f['name']
    assert download.is_file() and download.stat().st_size==f['bytes'] and sha(download)==f['sha256']
    if not findings.exists():shutil.copyfile(download,findings)
    out=BASE/'evaluated_20261014';out.mkdir(exist_ok=True)
    with zipfile.ZipFile(findings) as z:
        assert z.testzip() is None
        expected=json.loads(z.read('CONTENTS_SHA256.json'))
        for n,digest in expected.items():
            payload=z.read(n);assert hashlib.sha256(payload).hexdigest()==digest
            dest=out/n;assert dest.resolve().is_relative_to(out.resolve())
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():dest.write_bytes(payload)
            assert hashlib.sha256(dest.read_bytes()).hexdigest()==digest
    files=[]
    for p,expected in [(target,manifest['archive_sha256']),(findings,f['sha256'])]:
        dest=PRIVATE/p.name
        if not dest.exists():
            with p.open('rb') as source,dest.open('xb') as output:shutil.copyfileobj(source,output,8*1024**2)
        assert dest.stat().st_size==p.stat().st_size and sha(dest)==expected
        with zipfile.ZipFile(dest) as z:assert z.testzip() is None
        files.append({'host':str(p),'private_Drive_path':str(dest),'bytes':p.stat().st_size,'sha256':expected,'full_byte_and_CRC_readback_verified':True})
    receipt={'host_raw_transfer_complete':True,'private_Drive_full_readback_verified':True,
        'provider_metadata_verified':False,'private_parent_id':'19v-Rjj0WMfxmHiFssqwviLDiTdAxyAev','files':files,
        'new_model_calls':0,'GPU_allocations':0,'initial_failed_zero_forward_attempt_preserved_separately':True}
    (BASE/'FINAL_BACKUP_RECEIPT_20261014.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
